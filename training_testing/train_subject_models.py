import torch
import gc
from torch.utils.data import DataLoader, Dataset
import warnings
from eois import *
from datasets import *
import argparse, json
import numpy as np
import mne
import torch
import torch.multiprocessing as mp
import os, math, copy, random
from collections import defaultdict
from collections import deque
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Global variables for workers (backed by shared memory)
X_ALL = None
SUBJECT_IDS = None
OUT_DIR = None
SUBJECT_CODES = None

def init_worker(x_all, subject_ids, subject_codes, out_dir):
    global X_ALL, SUBJECT_IDS, OUT_DIR, SUBJECT_CODES
    X_ALL = x_all
    SUBJECT_IDS = subject_ids
    OUT_DIR = out_dir
    SUBJECT_CODES = subject_codes
    os.makedirs(OUT_DIR, exist_ok=True)

def ncycles_piecewise(freqs):
    n = np.empty_like(freqs, dtype=int)
    n[freqs < 6] = 3             # < 6 Hz → 3 cycles
    n[(freqs >= 6) & (freqs < 12)] = 4   # 6–10 Hz → 4 cycles
    n[(freqs >= 12) & (freqs < 20)] = 5  # 12–18 Hz → 5 cycles
    n[freqs >= 20] = 6           # ≥20 Hz → 6 cycles
    return n


def train(
    test_subj,                #test subject
    X_all,                    #all data n_trials x n_times
    subject_ids,              #subject ids for each trial
    device,
    pre_samples,              #pre-stimulus sample range
    post_samples,             #post-stimulus sample range
    epochs=300,               # maximum number of training epochs
    val_frac=0.3,             # % of training subjects for validation
    initial_lr=5e-4,          # starting learning rate
    weight_decay=1e-5,        # was 1e-4: slightly looser regularization
    out_dir=None,             # directory to save to
    run_name=None,            # name of the run
    monitor_key="total",      # which validation key to monitor
    min_lr=1e-7,              # floor learning rate

    # training stability knobs
    train_batch_size=200,     # if None: full subject as one batch; else minibatch within subject
    clip_grad_norm=5.0,       # allowed gradient magnitude
    shuffle_subjects=True,    # shuffle training subject order each epoch

    # early stopping knobs
    early_stopping_patience=30,  # patience
    min_delta=1e-4,              # threshold meaningful improvement

    # warmup
    warmup_epochs=10,         # warmup
    fs=1000,                  # sampling rate
    freqs=np.arange(4.0, 40.0, 2.0), # frequency bins
    n_cycles=None,
    weights=None
):
    if n_cycles is None:
        n_cycles = ncycles_piecewise(freqs)
        print(f"using cycles {n_cycles} for freqs {freqs}")
        n_cycles = n_cycles.astype(float)

        
    def _init_sums_from(loss_dict):
        sums = {k: 0.0 for k in loss_dict.keys()}
        if "total" not in sums:
            sums["total"] = 0.0
        return sums

    def _accumulate(loss_sums, loss_dict, loss_scalar, bs):
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                v = v.detach()
                val = v.item() if v.numel() == 1 else v.float().mean().item()
            else:
                val = float(v)
            loss_sums[k] += val * bs
        loss_sums["total"] += float(loss_scalar.detach().item()) * bs

    def _finalize(loss_sums, total_samples):
        return {k: (v / max(1, total_samples)) for k, v in loss_sums.items()}

    # -------------------------
    # Split train / val / test subjects
    # -------------------------
    test_subj = int(test_subj.item()) if torch.is_tensor(test_subj) else int(test_subj)

    train_mask = subject_ids != test_subj
    X_pool = X_all[train_mask]
    y_pool = subject_ids[train_mask]

    # a few guards to ensure no accidental leaking etc via types
    #------------------------------------
    assert X_all.shape[0] == len(subject_ids), (
    f"Row/label count mismatch: X_all.shape[0]={X_all.shape[0]} vs len(subject_ids)={len(subject_ids)}")

    assert X_pool.shape[0] == len(y_pool), (
    f"Row/label count mismatch: X_pool.shape[0]={X_pool.shape[0]} vs len(y_pool)={len(y_pool)}")
    
    assert not (y_pool == test_subj).any(), "Leakage: test subject present after masking"

    unique_subjs = torch.unique(y_pool)
    n_subj = len(unique_subjs)
        
    if len(torch.unique(subject_ids)) - n_subj != 1:
        raise ValueError("Test subject leaks, was not removed from data.")
        
    if n_subj == 0:
        raise ValueError("No subjects left for training/eval after excluding test_subj.")

    if n_subj > 1:
        n_val = max(1, int(round(val_frac * n_subj)))
    else:
        raise ValueError("Too little subjects (n_subj < 1)")
    #------------------------------------
    
    g = torch.Generator()
    g.manual_seed(42 +  test_subj)  # fixed for reproducibility
    perm = torch.randperm(n_subj, generator=g)
    val_subjs = unique_subjs[perm[:n_val]]
    trn_subjs = unique_subjs[perm[n_val:]]

    # -------------------------
    # Per-subject loaders
    # -------------------------
    train_loaders, val_loaders = [], []

    for subj_id in trn_subjs:
        subj_mask = (y_pool == subj_id)
        X_subj = X_pool[subj_mask]
        y_subj = y_pool[subj_mask]
        subj_dataset = NeuroDataset(X_subj, y_subj)
        bs = min(train_batch_size, X_subj.shape[0])
        loader = DataLoader(
            subj_dataset,
            batch_size=bs,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        train_loaders.append(loader)

    for subj_id in val_subjs:
        subj_mask = (y_pool == subj_id)
        X_subj = X_pool[subj_mask]
        y_subj = y_pool[subj_mask]
        subj_dataset = NeuroDataset(X_subj, y_subj)
        loader = DataLoader(
            subj_dataset,
            batch_size=X_subj.shape[0],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        val_loaders.append(loader)

    # -------------------------
    # Model, optimizer, scheduler
    # -------------------------
    model = EvokedOngoingInducedSeparator().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr, weight_decay=weight_decay,
        betas=(0.9, 0.98), eps=1e-8, amsgrad=True
    )

    # Warmup setup
    post_warmup_epochs = epochs - warmup_epochs

    # Less aggressive, with thresholds and cooldown
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=5,
        min_lr=min_lr
    )
    # Smooth the validation metric and delay scheduler stepping after warmup
    val_hist = deque(maxlen=5)
    extra_delay = 10  # epochs after warmup before stepping scheduler

    # -------------------------
    # Tracking / checkpoints
    # -------------------------
    history = []
    best_score = None
    best_epoch = -1
    best_state = None
    no_improve_epochs = 0  # early stopping counter

    code = SUBJECT_CODES[test_subj]
    if run_name is None: # set the run name
        run_name = f"subj_{code}_{test_subj}"

    if out_dir is not None: # set the out dir
        os.makedirs(out_dir, exist_ok=True)
        last_ckpt_path = os.path.join(out_dir, f"{run_name}_last.pt")
        best_ckpt_path = os.path.join(out_dir, f"{run_name}_best.pt")
    else:
        last_ckpt_path = best_ckpt_path = None

    # Optimizer starts at 0 LR for warmup
    if warmup_epochs and warmup_epochs > 0:
        for pg in optimizer.param_groups:
            pg["lr"] = 0.0
    else:
        for pg in optimizer.param_groups:
            pg["lr"] = initial_lr

    # -------------------------
    # Epoch loop with warmup + early stopping
    # -------------------------
    for epoch in range(1, epochs + 1):
        # ----- LR warmup step (set LR at epoch start) -----
        if warmup_epochs and epoch <= warmup_epochs:
            warmup_factor = float(epoch) / float(max(1, warmup_epochs))
            new_lr = initial_lr * warmup_factor
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        elif scheduler is not None and epoch == warmup_epochs + 1:
            # Ensure transition starts with initial_lr post-warmup
            for pg in optimizer.param_groups:
                pg["lr"] = initial_lr

        # ===== TRAIN =====
        model.train()
        train_sums, train_n = None, 0

        epoch_loaders = train_loaders
        if shuffle_subjects:
            import random as _random
            epoch_loaders = _random.sample(train_loaders, len(train_loaders))

        for subj_loader in epoch_loaders:
            for X_batch, _ in subj_loader:
                X_batch = X_batch.to(device)

                optimizer.zero_grad()
                E, U = model(X_batch)

                loss, loss_dict = model.compute_loss(
                    X_batch, E, U,
                    pre_samples,
                    post_samples,
                    fs, freqs, n_cycles,weights,
                )
                print(loss_dict, test_subj, "train")
                if not torch.isfinite(loss):
                    print("Skipping non-finite TRAIN batch:", {k: float(v) if np.isfinite(float(v)) else float('nan')
                                                            for k, v in loss_dict.items()})
                    continue

                if train_sums is None:
                    train_sums = _init_sums_from(loss_dict)
                bs = X_batch.size(0)
                _accumulate(train_sums, loss_dict, loss, bs)
                train_n += bs

                loss.backward()
                if clip_grad_norm and clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                optimizer.step()

        train_epoch = _finalize(train_sums, train_n) if train_sums is not None else {}

        # ===== VALIDATION =====
        model.eval()
        val_sums, val_n = None, 0
        with torch.no_grad():
            for subj_loader in val_loaders:
                for X_batch, _ in subj_loader:
                    X_batch = X_batch.to(device)

                    E, U = model(X_batch)

                    loss, loss_dict = model.compute_loss(
                        X_batch, E, U,
                        pre_samples,
                        post_samples,
                        fs, freqs, n_cycles, weights
                    )
                    print(loss_dict, test_subj, "val")
                    if not torch.isfinite(loss):
                        print("Skipping non-finite VAL batch:", {k: float(v) if np.isfinite(float(v)) else float('nan')
                                                                for k, v in loss_dict.items()})
                        continue

                    if val_sums is None:
                        val_sums = _init_sums_from(loss_dict)
                    bs = X_batch.size(0)
                    _accumulate(val_sums, loss_dict, loss, bs)
                    val_n += bs

        val_epoch = _finalize(val_sums, val_n) if val_sums is not None else {}
        history.append({"train": train_epoch, "val": val_epoch})

        # ===== LOG + LR STEP =====
        t_total = train_epoch.get("total", float("nan"))
        v_total = val_epoch.get("total", float("nan"))
        val_hist.append(v_total)
        v_total_smooth = float(np.mean(val_hist))
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | LR: {curr_lr:.6g} | train total: {t_total:.6f} | "
              f"val total: {v_total:.6f} (smoothed {v_total_smooth:.6f})")

        # Step scheduler AFTER warmup + extra delay using smoothed metric
        if scheduler is not None and epoch >= (warmup_epochs + extra_delay):
            scheduler.step(v_total_smooth)

        # save "last" checkpoint
        if last_ckpt_path is not None:
            state_last = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({"model": state_last, "losses": history, "epoch": epoch}, last_ckpt_path)

        # ===== EARLY STOPPING / BEST TRACKING (minimize monitor_key) =====
        # Use smoothed monitor as well to reduce noise-triggered resets
        monitored_raw = val_epoch.get(monitor_key, None)
        monitored = v_total_smooth if monitor_key == "total" else monitored_raw
        improved = False
        if monitored is not None and not math.isnan(monitored):
            if (best_score is None) or (monitored < best_score - float(min_delta)):
                improved = True
                best_score = monitored
                best_epoch = epoch
                best_state = copy.deepcopy(model.module.state_dict() if hasattr(model, "module") else model.state_dict())
                no_improve_epochs = 0
                if best_ckpt_path is not None:
                    torch.save(
                        {"model": best_state, "losses": history, "epoch": epoch,
                         "best_score": best_score, "monitor": monitor_key},
                        best_ckpt_path
                    )
                    print(f"  -> New best at epoch {epoch}: {monitor_key}={best_score:.6f}")
            else:
                no_improve_epochs += 1

        if val_loaders and monitored is not None:
            print(f"  EarlyStopping: no_improve_epochs={no_improve_epochs}/{early_stopping_patience} "
                  f"(min_delta={min_delta})")

        if val_loaders and (early_stopping_patience is not None) and (early_stopping_patience > 0):
            if (best_score is not None) and (no_improve_epochs >= early_stopping_patience):
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(no improvement in '{monitor_key}' for {early_stopping_patience} epochs).")
                break

    # -------------------------
    # Restore best (if any)
    # -------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model from epoch {best_epoch} with {monitor_key}={best_score:.6f}")
    else:
        print("No valid validation score recorded; using final model state.")

    # -------------------------
    # Final save
    # -------------------------
    if out_dir is not None:
        final_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        final_path = os.path.join(out_dir, f"{run_name}_final.pt")
        torch.save(
            {"model": final_state, "losses": history,
             "best_epoch": best_epoch, "best_score": best_score, "monitor": monitor_key,
             "val_subjects": val_subjs.cpu().tolist()},
            final_path,
        )
        print(f"Saved final checkpoint to: {final_path}")

    return model, history, best_epoch, val_subjs.cpu().tolist()



def run_subject_processing_worker(args):
    test_subj, gpu_id, pre_samples, post_samples, freqs, weights = args

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    try:
        model, history, best_epoch, val_subjs = train(
            test_subj,
            X_ALL,
            SUBJECT_IDS,
            device,
            pre_samples,
            post_samples,
            freqs=freqs,
            out_dir=OUT_DIR,
            weights=weights)
        if best_epoch > 0:
            best_rec = history[best_epoch - 1]
        else:
            best_rec = history[-1] if history else {}

        return test_subj, best_rec
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(stimulation_type, croptimes_all):
    warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"This filename ")
    
    for croptimes in croptimes_all: #different time crops for model inputs
        
        X_list = []
        subject_ids_list = []
        subkeys = {}
        indx = 0
        basepath = "/scratch/work/aholao1/"
        weights = None
        
        if stimulation_type == "visual":
            freqs = np.arange(4.0, 30.0, 2.0)
            root = os.path.join(basepath, "camcan_processed")
            for subject in sorted(os.listdir(root)):
                data, evoked_snr, fs, times = read_preprocessed_meg_epochs(subject, root, [None, 0], croptimes, [0.1, 0.3], [None, 0], 0.1,['vis/checker'], 'grad')
                if evoked_snr < 15:
                    continue   
                    
                X_list.append(data)
                subject_ids_list.append(np.full(data.shape[0], indx, dtype=np.int64))
                subkeys[indx]=subject
                indx += 1
                
        elif stimulation_type == "auditory":
            freqs = np.arange(4.0, 30.0, 2.0)
            root = os.path.join(basepath, "camcan_processed")
            for subject in sorted(os.listdir(root)):
                data, evoked_snr, fs, times = read_preprocessed_meg_epochs(subject, root, [None, 0], croptimes, [0.1, 0.3], [None, 0], 0.1,['audio/300Hz'], 'grad')
                if evoked_snr < 10:
                    continue   
                    
                X_list.append(data)
                subject_ids_list.append(np.full(data.shape[0], indx, dtype=np.int64))
                subkeys[indx]=subject
                indx += 1
                
        elif stimulation_type == "TMS":
            freqs = np.arange(4.0, 40.0, 2.0)
            root = os.path.join(basepath, "reftep_processed")
            for subject in sorted(os.listdir(root)):
                data, fs, times = read_preprocessed_tep_epochs(subject, root, "both", croptimes, "C3", max_trials=None)
                X_list.append(data)
                subject_ids_list.append(np.full(data.shape[0], indx, dtype=np.int64))
                subkeys[indx]=subject
                indx += 1
                
        elif stimulation_type == "TMS_strt":
            freqs = np.arange(4.0, 40.0, 2.0)
            root = os.path.join(basepath, "reftep_processed_strt")
            for subject in sorted(os.listdir(root)):
                data, fs, times = read_preprocessed_tep_epochs_rt(subject, root, croptimes, "C3")
                X_list.append(data)
                subject_ids_list.append(np.full(data.shape[0], indx, dtype=np.int64))
                subkeys[indx]=subject
                indx += 1
                
        elif stimulation_type == "simulation":
            freqs = np.arange(4.0, 40.0, 2.0)
            root = os.path.join(basepath, "reftep_processed")
            subjects_data = []

            for subject in sorted(os.listdir(root)):
                
                out = prep_simulated_data(subject, root, [-1.053,-0.053], [-0.5, 0.5], "C3", threshold1=5*1e-6, threshold2=10)
                
                if out is not False:
                    data, added_evoked, prestim_data, evoked_template, displacement_samples, amp_envelopes, amp_globals, fs, times, rms = out
                else:
                    continue
                
                subject_data = {
                "data": data/rms,
                "added_evoked": added_evoked/rms,
                "prestim_data": prestim_data/rms,
                "evoked_template":evoked_template/rms,
                "displacement_samples": displacement_samples,
                "amp_envelopes": amp_envelopes,
                "amp_globals": amp_globals,
                "fs": fs,
                "times":times,
                "rms": rms,
                "subject": subject}
                data = data/rms
                X_list.append(data)
                subject_ids_list.append(np.full(data.shape[0], indx, dtype=np.int64))
                subkeys[indx]="sub-" + str(indx+1)
                indx += 1
                subjects_data.append(subject_data)
                
            np.save("models/simulated_data.npy", subjects_data) #save the data
            del subjects_data

        #ablated variants
        elif "simulation" in stimulation_type and "_no" in stimulation_type:
            freqs = np.arange(4.0, 40.0, 2.0)
            subjects_data = np.load("models/simulated_data.npy", allow_pickle=True)
            times = subjects_data[0]["times"]

            if "nophase" in stimulation_type:
                weights = dict(e_ppc=0.0, u_ppc=0.0)
            elif "nomean" in stimulation_type:
                weights = dict(evoked_reco=0.0, ongoing_induced_reco=0.0)
            elif "nosingletrial" in stimulation_type:
                weights = dict(trial_reco=0.0, pre_reco=0.0)
            X_list = [s['data'] for s in subjects_data]
            for indx in range(len(X_list)):
                subkeys[indx]="sub-" + str(indx+1)
                subject_ids_list.append(np.full(X_list[indx].shape[0], indx, dtype=np.int64))
                
            del subjects_data
        else:
            raise ValueError("invalid stimulation_type")
            
        # free the builders ASAP to release RAM
        X_all = np.concatenate(X_list, axis=0)
        subject_ids = np.concatenate(subject_ids_list)
        del X_list, subject_ids_list
        
        X_all = torch.from_numpy(X_all).to(torch.float32).share_memory_()
        subject_ids = torch.from_numpy(subject_ids).to(torch.int64).share_memory_()
    
        unique_subs = torch.unique(subject_ids).tolist()
    
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        num_workers = max(1, num_gpus)
        print(f"cuda {torch.cuda.is_available()} | gpus: {num_gpus}")
    
        out_dir = f"models/{stimulation_type}_time_{croptimes}_models_v"
        while os.path.exists(out_dir):
            out_dir += "1"
        
        pre_samples = [0, np.max(np.where(times <= 0)).astype(int)]
        post_samples = [pre_samples[1], X_all.shape[-1]]

        tasks = [(s_ind, i % num_gpus, pre_samples, post_samples, freqs, weights) for i, s_ind in enumerate(unique_subs)]
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(X_all, subject_ids, subkeys, out_dir),
            maxtasksperchild=1,
        ) as pool:
            for subj, avg_loss in pool.imap_unordered(run_subject_processing_worker, tasks, chunksize=1):
                print(f"Completed subject {subj}: avg_loss={avg_loss}", flush=True)



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    p = argparse.ArgumentParser()
    p.add_argument("--stimulationtype", required=True, type=str)
    p.add_argument("--croptimes", required=True, type=str)
    args = p.parse_args()
    croptimes_all = json.loads(args.croptimes)
    
    main(args.stimulationtype, croptimes_all)









