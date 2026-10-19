import os
import mne
import torch
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings(
    "ignore",
    message="This filename.*",
    category=RuntimeWarning,
)
from datasets import *
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from eois import *
from tslearn.metrics import dtw_path
from sklearn.metrics import r2_score, explained_variance_score
from scipy.stats import zscore
from scipy.stats import pearsonr


def load_model_state(model, state, device):
    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        clean_state[k] = v
    
    # Load weights, move to device, set evaluation mode
    _ = model.load_state_dict(clean_state, strict=True)
    model = model.to(device).eval()
    return model

def apply_model_trial_by_trial(test_loader, model, device):
    es = []
    us = []
    xs = []
    with torch.no_grad():
        for i, (xb, sb) in enumerate(test_loader):
            xb = xb.to(device, dtype=torch.float32)
            sb = sb.to(device)
            E, U = model(xb)
            es.append(E.detach())
            us.append(U.detach())
            xs.append(xb.detach())
        es = torch.stack(es, dim=0).squeeze()
        us = torch.stack(us, dim=0).squeeze()
        xs = torch.stack(xs, dim=0).squeeze()
        E = es.cpu().numpy()
        U = us.cpu().numpy()
        X = xs.cpu().numpy()
    return E, U, X


def test_subject(subject, model_path, datatype, model_datatype, freqs, pre_end, post_end, n_cycles, eps=1e-8):
    # load data
    model = EvokedOngoingInducedSeparator()
    if datatype == "vis/checker" or datatype == "audio/300Hz":
        datatype_list = [datatype]
        data, evoked_snr, fs, t = read_preprocessed_meg_epochs(subject, "/scratch/work/aholao1/camcan_processed", [None, 0],
                                                               [None, None], [0.1, 0.3], [None, 0], 0.1, datatype_list, 'grad')
        evoked_snr_lim = 15 if datatype == "vis/checker" else 10 #different snr limits due to differing number of trials
        if evoked_snr < evoked_snr_lim:
            return False
        data = torch.tensor(data, dtype=torch.float32)
        subject_id = torch.tensor(np.full(data.shape[0], int(model_path.split("_")[-2]) , dtype=np.int64))
        test_dataset  = NeuroDataset(data, subject_id)
        test_loader = DataLoader(test_dataset, batch_size=1)

    elif "TMS" in datatype:
        if "_st" in datatype:
            root = "/scratch/work/aholao1/reftep_processed_strt"
            data, fs, t = read_preprocessed_tep_epochs_rt(subject, root, [-0.5, None], "C3")
        else:
            root = "/scratch/work/aholao1/reftep_processed"
            data, fs, t = read_preprocessed_tep_epochs(subject, root, 'both', [-0.5, 0.5], "C3")
        data = torch.tensor(data, dtype=torch.float32)
        subject_id = torch.tensor(np.full(data.shape[0], int(model_path.split("_")[-2]) , dtype=np.int64))
        test_dataset  = NeuroDataset(data, subject_id)
        test_loader = DataLoader(test_dataset, batch_size=1)
            
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    bundle = torch.load(model_path, map_location="cpu")
    state = bundle["model"]
    model = load_model_state(model, state, device) #set the trained model features

    # test on all of the data of the test subject at once
    E, U, X_batch = apply_model_trial_by_trial(test_loader, model, device)

    E = torch.as_tensor(E, dtype=torch.float32, device=device)
    U = torch.as_tensor(U, dtype=torch.float32, device=device)
    X_batch = torch.as_tensor(X_batch, dtype=torch.float32, device=device)

    Xc = morlet_cwt(X_batch, freqs, fs, n_cycles)
    Uc = morlet_cwt(U, freqs, fs, n_cycles)
    Ec = morlet_cwt(E, freqs, fs, n_cycles)

    #absolute means across trials
    X_mean_abs = X_batch.mean(0).abs().cpu().numpy()
    E_mean_abs = E.mean(0).abs().cpu().numpy()
    U_mean_abs = U.mean(0).abs().cpu().numpy()
    
    #pairwise phase consistencies
    mask_lo = freqs <= 15.0
    mask_hi = (freqs > 15.0) & (freqs <= 30.0)
    
    ppc_u = ppc_mean(Uc[:,:,pre_end:post_end], eps)
    ppc_u_pre = ppc_mean(Uc[:,:,:pre_end], eps)
    ppc_e = ppc_mean(Ec[:,:,pre_end:post_end], eps)
    ppc_x_post = ppc_mean(Xc[:,:,pre_end:post_end], eps)
    
    ppc_u_lo = ppc_mean(Uc[:,mask_lo,pre_end:post_end], eps)
    ppc_u_pre_lo = ppc_mean(Uc[:,mask_lo,:pre_end], eps)
    ppc_e_lo = ppc_mean(Ec[:,mask_lo,pre_end:post_end], eps)
    ppc_x_post_lo = ppc_mean(Xc[:,mask_lo,pre_end:post_end], eps)

    ppc_u_hi = ppc_mean(Uc[:,mask_hi,pre_end:post_end], eps)
    ppc_u_pre_hi = ppc_mean(Uc[:,mask_hi,:pre_end], eps)
    ppc_e_hi = ppc_mean(Ec[:,mask_hi,pre_end:post_end], eps)
    ppc_x_post_hi = ppc_mean(Xc[:,mask_hi,pre_end:post_end], eps)
    
    # trial recovery
    y    = X_batch[:, pre_end:post_end].cpu().numpy()
    yhat = (E + U)[:, pre_end:post_end].cpu().numpy()
    trial_reco = np.median([r2_score(yt, yp) for yt, yp in zip(y, yhat)])

    # recovery of average evoked components
    evoked_reco = r2_score(X_batch.mean(0).cpu().numpy()[pre_end:post_end], E.mean(0).cpu().numpy()[pre_end:post_end])

    #print(evoked_reco, ppc_u, ppc_u_pre, ppc_e)
    #return
        

    #create a summary dict
    if "vis" in datatype:
        datatype = "visual" 
    if "audio" in datatype:
        datatype = "auditory"
    summary = {"subject": subject, "stimulation_type_test": datatype, "stimulation_type_train": model_datatype,
               "x_mean_abs":X_mean_abs, "e_mean_abs":E_mean_abs, "u_mean_abs":U_mean_abs,
              "u_ppc_post":ppc_u, "u_ppc_pre": ppc_u_pre, "e_ppc_post": ppc_e, "ppc_x_post": ppc_x_post,
               "u_ppc_post_lo":ppc_u_lo, "u_ppc_pre_lo": ppc_u_pre_lo, "e_ppc_post_lo": ppc_e_lo, "ppc_x_post_lo": ppc_x_post_lo,
               "u_ppc_post_hi":ppc_u_hi, "u_ppc_pre_hi": ppc_u_pre_hi, "e_ppc_post_hi": ppc_e_hi, "ppc_x_post_hi": ppc_x_post_hi,
               "trial_reco_med": np.median(trial_reco), "evoked_reco": evoked_reco,
              "times": t,
              "freqs": freqs,
              "E":E.cpu().numpy(),
              "U":U.cpu().numpy(),
               "X":X_batch.cpu().numpy(),
              "x_tfr_mag_mean":Xc.abs().mean(0).cpu().numpy(),
              "e_tfr_mag_mean":Ec.abs().mean(0).cpu().numpy(),
              "u_tfr_mag_mean":Uc.abs().mean(0).cpu().numpy(),
              "x_tfr_mag_std":Xc.abs().std(0).cpu().numpy(),
              "e_tfr_mag_std":Ec.abs().std(0).cpu().numpy(),
              "u_tfr_mag_std":Uc.abs().std(0).cpu().numpy(),
               "pre_end":pre_end,
              "n_trials": len(data)}

    return summary


def test_sim_subjects(simulated_data_path, models_paths):

    subjects_data = np.load(simulated_data_path,allow_pickle=True)
    r2s_e = {"eois": [], "nomean":[], "nophase":[], "nosingletrial":[], "lse":[], "dtw":[]}
    r2s_u = {"eois": [], "nomean":[], "nophase":[], "nosingletrial":[], "lse":[], "dtw":[]}
    snrs = []
    
    for index, models_path_now in enumerate(models_paths):
        modeltype = models_path_now.split("_")[1]
        #print(modeltype)
        if "no" not in modeltype:
            modeltype="eois"
        for k in range(len(subjects_data)):
            model_path = os.path.join(models_path_now, f"subj_sub-{k+1}_{k}_final.pt")
            sdat = subjects_data[k]
            data = sdat["data"]
            ongoing = sdat["prestim_data"]
            evoked = sdat["added_evoked"]
            U_true = np.asarray(ongoing, dtype=np.float32)
            E_true = np.asarray(evoked, dtype=np.float32)
            t = sdat["times"]
            data = torch.tensor(data, dtype=torch.float32)
            subject_id = torch.tensor(np.full(data.shape[0], int(model_path.split("_")[-2]) , dtype=np.int64))
            test_dataset  = NeuroDataset(data, subject_id)
            test_loader = DataLoader(test_dataset, batch_size=1)
    
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # load model
            bundle = torch.load(model_path, map_location="cpu")
            state = bundle["model"]
            model = EvokedOngoingInducedSeparator()
            model = load_model_state(model, state, device) #set the trained model features and set to eval
            pre_end = np.max(np.where(t <= 0)) #end of pre-stim
            pre_sl  = slice(0, pre_end)
            post_sl = slice(pre_end, len(t))
            #evaluate
            E, U, _ = apply_model_trial_by_trial(test_loader, model, device)
            E = np.asarray(E, dtype=np.float32)
            U = np.asarray(U, dtype=np.float32)
    
            #calc single trial accuracies
            r2_e = np.median([r2_score(et, ep) for et, ep  in zip(E_true[:, post_sl], E[:, post_sl])])
            r2_u_pre = np.median([r2_score(ut, up) for ut, up  in zip(U_true[:, pre_sl], U[:, pre_sl])])
            r2_u_post = np.median([r2_score(ut, up) for ut, up  in zip(U_true[:, post_sl], U[:, post_sl])])

            #general evoked average accuracy
            r2_em = r2_score(E_true[:, pre_end:].mean(0), E[:, pre_end:].mean(0))


            #print(sdat['subject'])
            #print(r2_e, r2_u_pre, r2_u_post, r2_em)
            r2s_e[modeltype].append(r2_e)
            r2s_u[modeltype].append(r2_u_post)
            if index == 0: #only eval these on the first run because they don't depend on eois
                metrics, est = template_fit_metrics_dtw(sdat,
                subjects_data,
                target_index=k,        # exclude this subject from donors 
                dtw_band=None,   
                eps=0)
                r2s_e['lse'].append(metrics['r2_e_lse'])
                r2s_u['lse'].append(metrics['r2_u_post_lse'])
                r2s_e['dtw'].append(metrics['r2_e_dtw'])
                r2s_u['dtw'].append(metrics['r2_u_post_dtw'])
                e_rms = np.sqrt(np.mean(E_true[:, pre_end:]**2))
                u_rms = np.sqrt(np.mean(U_true[:, pre_end:]**2))
                snrs.append(e_rms/u_rms)
    snrs = np.array(snrs)
    #check model performance against the template post-stim RMS 
    models_to_report_corr = ['eois', 'lse', 'dtw']
    for r2, compname in zip([r2s_e, r2s_u], ["e", "u"]):
        for modeltype_now in models_to_report_corr:
            noise_now = snrs if compname == "e" else 1/snrs
            r, p = pearsonr(noise_now, r2[modeltype_now])
            print(f"{compname}, {modeltype_now}: r = {r}, p = {p}")
    
    def med_mad(x):
        x = np.asarray(x, dtype=float)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return med, mad
    
    def pm(x, y, prec=2):
        return f"{x:.{prec}f} ± {y:.{prec}f}"
    
    def stars(p):
        # different in the article
        return '***'   if p < 1e-3 else \
           '**'    if p < 1e-2 else \
           '*'     if p < 5e-2 else ''
    
    def adjust_family_with_multipletests(pdict, alpha=0.05, method="holm"):
        keys = list(pdict.keys())
        pvals = np.array([pdict[k] for k in keys], dtype=float)
        reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=method)
        return dict(zip(keys, p_adj)), dict(zip(keys, reject))
    
    def to_array(x):
        return np.asarray(x, dtype=float)
    
    r2e = {k: to_array(v) for k, v in r2s_e.items()}
    r2u = {k: to_array(v) for k, v in r2s_u.items()}
    
    # Models to compare against eois (include the "no..." ones)
    comparators = ['nophase', 'nosingletrial', 'nomean', 'lse', 'dtw']
    labels = {
        'nophase': 'NoPhase',
        'nosingletrial': 'NoSingleTrial',
        'nomean': 'NoMean',
        'lse': 'LSE',
        'dtw': 'DTW'
    }
    
    # Summary: med ± MAD for all models
    def summarize_all():
        print("R² summaries (median ± MAD)")
        for k in ['eois'] + comparators:
            e_med, e_mad = med_mad(r2e[k])
            u_med, u_mad = med_mad(r2u[k])
            print(f"{labels.get(k, k).upper():>13s} | E: {pm(e_med, e_mad)} | U: {pm(u_med, u_mad)}")
    
    summarize_all()
    
    # Paired comparisons vs eois
    def paired_wilcoxon_vs_ref(arr_ref, arr_cmp):
        ref = np.asarray(arr_ref, float)
        cmp = np.asarray(arr_cmp, float)
        stat, p = wilcoxon(ref, cmp, alternative='two-sided', zero_method='wilcox', mode='auto')
        med_diff = np.median(ref - cmp)
        return stat, p, med_diff
    
    # Compute all p-values and medians vs eois
    results = []  # list of dicts for printing
    pvals_E = {}
    pvals_U = {}
    
    for k in comparators:
        # E metric
        stat_e, p_e, meddiff_e = paired_wilcoxon_vs_ref(r2e['eois'], r2e[k])
        # U metric
        stat_u, p_u, meddiff_u = paired_wilcoxon_vs_ref(r2u['eois'], r2u[k])
        results.append({
            'model': k,
            'label': labels[k],
            'stat_e': stat_e, 'p_e': p_e, 'meddiff_e': meddiff_e,
            'stat_u': stat_u, 'p_u': p_u, 'meddiff_u': meddiff_u,
        })
        pvals_E[(labels[k], 'E')] = p_e
        pvals_U[(labels[k], 'U')] = p_u
    
    # Holm adjustment within E and within U
    adj_p_E, rej_E = adjust_family_with_multipletests(pvals_E)
    adj_p_U, rej_U = adjust_family_with_multipletests(pvals_U)
    
    # Print detailed lines
    print("\nWilcoxon paired tests vs eois (raw p and within-metric Holm-adjusted p)")
    for r in results:
        lab = r['label']
        # E
        p_e_adj = adj_p_E[(lab, 'E')]
        e_star = stars(p_e_adj)
        print(f"eois vs {lab:>13s} (E): statistic={r['stat_e']:.3f}, p={r['p_e']:.3g}, Holm(E) p={p_e_adj:.3g}, "
              f"median diff={r['meddiff_e']:.3f} (eois - {lab}) {e_star}")
        # U
        p_u_adj = adj_p_U[(lab, 'U')]
        u_star = stars(p_u_adj)
        print(f"eois vs {lab:>13s} (U): statistic={r['stat_u']:.3f}, p={r['p_u']:.3g}, Holm(U) p={p_u_adj:.3g}, "
              f"median diff={r['meddiff_u']:.3f} (eois - {lab}) {u_star}")
    
    # Build a simple LaTeX-friendly median table with stars
    def star_for(model_key, metric):  # metric in {'E','U'}
        lab = labels[model_key]
        if metric == 'E':
            return stars(adj_p_E[(lab, 'E')])
        else:
            return stars(adj_p_U[(lab, 'U')])
    
    median_table = []
    median_table.append(f"eois & {pm(*med_mad(r2e['eois']))} & {pm(*med_mad(r2u['eois']))} \\\\")
    for k in comparators:
        e_med, e_mad = med_mad(r2e[k])
        u_med, u_mad = med_mad(r2u[k])
        median_table.append(
            f"{labels[k]:<13s} & {pm(e_med, e_mad)} {star_for(k,'E')} & {pm(u_med, u_mad)} {star_for(k,'U')} \\\\"
        )
    
    print("\n".join(median_table))

def _warp_template_to_trial_post(template_post, trial_post, band=None):
    """
    Warp template_post to trial_post timeline using tslearn's DTW with a
    Sakoe–Chiba constraint. Returns y_warp with len(trial_post).
    band: None | int (samples) | float in (0,1] fraction of min length
    """
    x = np.asarray(template_post, dtype=np.float32)
    y = np.asarray(trial_post, dtype=np.float32)
    nx, ny = len(x), len(y)
    if band is None:
        radius = 40
    elif isinstance(band, float):
        radius = max(1, int(round(band * min(nx, ny))))
    else:
        radius = int(band)

    path, _ = dtw_path(x, y, global_constraint="sakoe_chiba",
                       sakoe_chiba_radius=radius)

    # template samples aligned to each trial index j
    sums = np.zeros(ny, dtype=np.float64)
    cnts = np.zeros(ny, dtype=np.int32)
    for i, j in path:
        sums[j] += x[i]
        cnts[j] += 1
        
    if np.any(cnts == 0):
        raise ValueError("DTW path did not cover all target indices (cnts has zeros).")

    y_warp = (sums / cnts).astype(np.float32)
    return y_warp

def template_fit_metrics_dtw(
    sdat,
    all_sdats,
    target_index=None,     # exclude this subject from donors
    dtw_band=None,         # None, int (samples), or float (fraction of min length)
    eps=0.0
):
    """
    Aligns the template to each trial's post-stim via 1. LSE and 2. DTW (tslearn, Sakoe–Chiba),
    then fits a per-trial gain.
    """
    X = np.asarray(sdat["data"], dtype=np.float32)     # (n_trials, n_times)
    t = np.asarray(sdat["times"])
    n_trials, n_times = X.shape

    # Pre/post slices
    pre_end = int(np.max(np.where(t <= 0)))
    pre_sl  = slice(0, pre_end)
    post_sl = slice(pre_end, n_times)

    # Build donor-based template (mean of subjects' means)
    donors = []
    for si, s in enumerate(all_sdats):
        if target_index is not None and si == int(target_index):
            continue
        template_now = np.asarray(s["data"]*s["rms"], dtype=np.float32).mean(axis=0)
        template_now =  template_now / np.sqrt(np.mean(template_now**2))
        donors.append(template_now)
    donors = np.vstack(donors)
    template = np.mean(donors, axis=0).astype(np.float32)

    t_pre  = template[pre_sl]
    t_post = template[post_sl]
    t_mu  = t_post.mean()
    t_sd  = t_post.std(ddof=0)
    t_post_z = zscore(t_post, ddof=0)


    # LSE gain (post only)
    denom = float(np.dot(t_post, t_post) + eps)
    num = (X[:, post_sl] @ t_post)  # (n_trials,)
    a_lse = num / denom
    E_hat_lse = (a_lse[:, None] * template[None, :]).astype(np.float32)

    # DTW per trial
    E_hat_dtw = np.zeros_like(X, dtype=np.float32)
    a_dtw = np.zeros(n_trials, dtype=np.float32)
    
    for k in range(n_trials):
        y_post = X[k, post_sl]
        y_post_z = zscore(y_post, ddof=0)
    
        # Warp z-scored template to z-scored trial
        t_post_warp_z = _warp_template_to_trial_post(t_post_z, y_post_z, band=dtw_band)
    
        # De-standardize warped template back to original units
        t_post_warp = t_mu + t_sd * t_post_warp_z

        denom_w = float(np.dot(t_post_warp, t_post_warp) + eps)
        num_w = float(np.dot(y_post, t_post_warp))
        a = num_w / denom_w
        a_dtw[k] = a

        E_hat_dtw[k, pre_sl]  = a * t_pre
        E_hat_dtw[k, post_sl] = a * t_post_warp

    # Residuals
    U_hat_lse  = X - E_hat_lse
    U_hat_dtw  = X - E_hat_dtw

    # Ground truth
    E_true = np.asarray(sdat["added_evoked"], dtype=np.float32)
    U_true = np.asarray(sdat["prestim_data"], dtype=np.float32)


    # Convenience slicers
    Et, El, Ed = E_true[:, post_sl], E_hat_lse[:, post_sl], E_hat_dtw[:, post_sl]
    Ut, Ul, Ud = U_true[:, post_sl], U_hat_lse[:, post_sl], U_hat_dtw[:, post_sl]
    
    # Per-trial R² 
    r2_e_lse  = np.median([r2_score(t, p) for t, p in zip(Et, El)])
    r2_e_dtw  = np.median([r2_score(t, p) for t, p in zip(Et, Ed)])
    
    r2_u_post_lse  = np.median([r2_score(t, p) for t, p in zip(Ut, Ul)])
    r2_u_post_dtw  = np.median([r2_score(t, p) for t, p in zip(Ut, Ud)])


    metrics = {
        "r2_e_lse": r2_e_lse,
        "r2_e_dtw": r2_e_dtw,
        "r2_u_post_lse": r2_u_post_lse,
        "r2_u_post_dtw": r2_u_post_dtw,
    }
   #print(metrics)

    estimates = dict(
        E_hat_lse=E_hat_lse,
        E_hat_dtw=E_hat_dtw,
        U_hat_lse=U_hat_lse,
        U_hat_dtw=U_hat_dtw,
        template=template,
        scales_lse=a_lse.astype(np.float32),
        scales_dtw=a_dtw.astype(np.float32)
    )
    return metrics, estimates





        

        
        
        
    
    









