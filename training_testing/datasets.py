import mne
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter1d
import os
from scipy.ndimage import shift as nd_shift

class NeuroDataset(Dataset):
    def __init__(self, X, subject_ids):
        """
        X: tensor (n_trials, n_channels, n_times)
        subject_ids: tensor (n_trials,)
        """
        self.X = X
        self.subject_ids = subject_ids
        
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.subject_ids[idx]

# ---------- read preprocessed epoched data ----------

def read_preprocessed_meg_epochs(subject, root, baseline, croptimes, post_croptimes, pre_croptimes, post_end_shift, stimtypes, ch_type):
    fp = f"{root}/{subject}/{subject}_processed_passive-epo.fif" #define filepath
    # Events are {'audio/300Hz': 6, 'audio/600Hz': 7, 'audio/1200Hz': 8, 'vis/checker': 9}
    # Read epochs, select event types, apply baseline correction and crop the data
    epochs = mne.read_epochs(fp, verbose=False).pick([ch_type])[stimtypes]
    
    t = baseline[1]
    mask = epochs.times < t
    idx = np.searchsorted(epochs.times, t, side='left') - 1
    bl_end = epochs.times[idx]
    epochs.apply_baseline((baseline[0], bl_end), verbose=False).crop(croptimes[0],croptimes[1])
    """
    #alternative way ot determining a good channel
    cropped_evoked_data = epochs.copy().crop(post_croptimes[0], post_croptimes[1]).average().data
    vals = np.trapezoid(np.abs(cropped_evoked_data), axis=1)  # or auc of the data in the time range
    highest_arg = np.argmax(vals) #channel ind with highest auc
    epochs.pick([epochs.ch_names[highest_arg]]) #pick the best channel

    #check that the snr constraint is fullfilled
    evoked = epochs.copy().average()
    data_pre = evoked.copy().crop(pre_croptimes[0], pre_croptimes[1]).data
    data_post = evoked.copy().crop(post_croptimes[0], post_croptimes[1]).data
    rms_pre = np.sqrt((data_pre**2).mean())
    rms_post = np.sqrt((data_post**2).mean())
    evoked_snr = rms_post / rms_pre
    """
    # evoked response as the average of epochs
    evoked = epochs.copy().average()
    
    # crop to pre/post windows
    evoked_pre = evoked.copy().crop(pre_croptimes[0], pre_croptimes[1])
    evoked_post = evoked.copy().crop(post_croptimes[0], post_croptimes[1])
    
    # compute per-channel RMS in each window
    data_pre = evoked_pre.data
    data_post = evoked_post.data
    rms_pre = np.sqrt(np.mean(data_pre ** 2, axis=1))
    rms_post = np.sqrt(np.mean(data_post ** 2, axis=1))
    
    # SNR per channel
    snr = rms_post / rms_pre
    
    # pick the channel with the highest SNR
    best_idx = np.argmax(snr)
    best_ch = evoked.ch_names[best_idx]
    epochs.pick([best_ch])
    
    evoked_snr = snr[best_idx] # SNR of the chosen channel
    polarity = np.sign(np.trapezoid(evoked_post.pick([best_ch]).crop(None, evoked_post.times[-1]-post_end_shift).data.squeeze()))
    data = epochs.get_data(copy=True).squeeze()  # n_trials x n_times (squeeze out singleton channel dim)
    rms = np.sqrt((data**2).mean()) # normalization factor
    data = data / rms # normalize by rms
    
    return data*polarity, evoked_snr, evoked.info['sfreq'], evoked.times
    

def read_preprocessed_tep_epochs(subject, basepath, data_type, croptimes, channel, normalize=True, max_trials=None):
    if data_type not in ["calibration", "application", "both"]:
        raise ValueError("Bad data_type!")
    fp = os.path.join(basepath, f"{subject}/{subject}_calibration_post.fif")
    epochs = mne.read_epochs(fp, verbose=False).crop(croptimes[0], croptimes[1]) # read data and crop
    epochs.pick([channel]) # pick the channel of interest
    data = epochs.get_data(copy=True).squeeze()  # squeeze out the singleton dimension
    rms = np.sqrt((data**2).mean())      # normalization factor
    
    if data_type == "calibration":
        if normalize:
            data = data / rms
        return data, epochs.info['sfreq'], epochs.times #return normalized data, sampling freq and times
    else:
        fp = os.path.join(basepath, f"{subject}/{subject}_fixedprep_post.fif")
        epochs = mne.read_epochs(fp, verbose=False).crop(croptimes[0], croptimes[1]) # read data and crop
        epochs.pick([channel]) # pick the channel of interest
        data2 = epochs.get_data(copy=True).squeeze()  # squeeze out the singleton dimension
        # normalize by the rms calculated from the calibration data and return the normalized data along with sampling freq and times
        if data_type == "both":
            data = np.concatenate([data, data2], axis=0) # contacenate calibration and application data in that order
        else:
            data = data2
        if normalize:
            data = data / rms
        if max_trials is not None:
            if len(data) > max_trials:
                data = data[:max_trials]
        return data, epochs.info['sfreq'], epochs.times


def read_preprocessed_tep_epochs_rt(subject, basepath, croptimes, channel):
    fp = os.path.join(basepath, f"{subject}/{subject}_rt-epo.fif")
    epochs = mne.read_epochs(fp, verbose=False).crop(croptimes[0], croptimes[1]) # read data and crop
    epochs.pick([channel]) # pick the channel of interest
    data = epochs.get_data(copy=True).squeeze()  # squeeze out the singleton dimension
    rms = np.sqrt(data[0,:]**2).mean()      # normalization factor
    return data / rms, epochs.info['sfreq'], epochs.times #return normalized data, sampling freq and times

        
def read_preprocessed_prestim_epochs(subject, basepath, data_type, croptimes, channel, normalize=True):
    if data_type not in ["calibration", "application", "both"]:
        raise ValueError("Bad data_type!")
    fp = os.path.join(basepath, f"{subject}/{subject}_calibration_pre.fif")
    epochs = mne.read_epochs(fp, verbose=False).crop(croptimes[0], croptimes[1]) # read data and crop
    epochs.pick([channel]) # pick the channel of interest
    data = epochs.get_data(copy=True).squeeze()  # squeeze out the singleton dimension
    rms = np.sqrt((data**2).mean())      # normalization factor
    
    if data_type == "calibration":
        if normalize:
            data = data / rms
        return data, epochs.info['sfreq'], epochs.times #return normalized data, sampling freq and times
    else:
        fp = os.path.join(basepath, f"{subject}/{subject}_fixedprep_pre.fif")
        epochs = mne.read_epochs(fp, verbose=False).crop(croptimes[0], croptimes[1]) # read data and crop
        epochs.pick([channel]) # pick the channel of interest
        data2 = epochs.get_data(copy=True).squeeze()  # squeeze out the singleton dimension
        # normalize by the rms calculated from the calibration data and return the normalized data along with sampling freq and times
        if data_type == "both":
            data = np.concatenate([data, data2], axis=0) # contacenate calibration and application data in that order
        else:
            data = data2
        if normalize:
            data = data / rms
        data -= data.mean(axis=1, keepdims=True) #baseline correct
        return data, epochs.info['sfreq'], epochs.times

def add_evoked_with_smooth_time_warp_and_amp_envelope(
    prestim_data,
    evoked_template,
    sfreq,
    # jitter controls
    jitter_mode="none",           # "none" (in the paper) | "shift" (global latency, not used) | "warp" (smooth local, not used)
    jitter_sd_ms=5.0,              # target SD of timing jitter (ms). For "shift": SD of latency; for "warp": RMS of displacement field
    warp_smooth_sd_ms=20.0,        # smoothness of local warp (only used if jitter_mode="warp")
    max_jitter_ms=None,            # optional clamp for |latency| (shift) or |displacement(t)| (warp)
    # amplitude params (unchanged)
    amp_global_sd=0.5,
    amp_mod_sd=1.0,
    amp_mod_smooth_sd_ms=40.0,
    amp_envelope_mode="lognormal",  # "additive" or "lognormal"
    min_amp_scale=0.3,
    max_amp_scale=None,
    # onset control
    onset=501,                     # envelope applied at/after this sample
    random_state=42,
):
    """
    Adds the evoked template with amplitude modulation and optional timing jitter.

    jitter_mode:
      - "none": no timing jitter (original behavior).
      - "shift": per-trial constant latency shift ~ N(0, jitter_sd_ms) [ms].
      - "warp": smooth per-trial displacement field with RMS ≈ jitter_sd_ms, smoothed by warp_smooth_sd_ms.

    Timing jitter is applied to the template before amplitude modulation.
    """
    rng = np.random.default_rng(random_state)

    prestim_data = np.asarray(prestim_data, float)
    template = np.asarray(evoked_template, float)
    n_trials, n_times = prestim_data.shape
    if template.shape != (n_times,):
        raise ValueError("evoked_template must have shape (n_times,)")

    # Precompute scales
    amp_smooth_samp = max(1e-12, amp_mod_smooth_sd_ms / 1000.0 * sfreq)
    warp_smooth_samp = max(1e-12, warp_smooth_sd_ms / 1000.0 * sfreq)
    jitter_sd_samp   = float(jitter_sd_ms) / 1000.0 * sfreq
    max_jitter_samp  = None if max_jitter_ms is None else float(max_jitter_ms) / 1000.0 * sfreq

    # Outputs
    added_evoked = np.zeros((n_trials, n_times), float)
    displacement_samples = np.zeros((n_trials, n_times), float)  # shift (constant) or local displacement
    amp_envelopes = np.ones((n_trials, n_times), float)
    # Global trial-wise amplitude scales
    amp_globals = rng.normal(loc=1.0, scale=amp_global_sd, size=n_trials) if amp_global_sd > 0 else np.ones(n_trials)
    if min_amp_scale is not None:
        amp_globals = np.maximum(amp_globals, min_amp_scale)

    # Time axis in samples
    t = np.arange(n_times, dtype=float)

    for i in range(n_trials):
        # 1) Build jittered template
        if jitter_mode == "none" or jitter_sd_ms <= 0:
            warped = template.copy()
            disp = np.zeros(n_times, float)
        else:
            raise ValueError("jitter_mode must be 'none', others not implemented")

        displacement_samples[i] = disp

        # 2) Amplitude envelope (post-onset only)
        envelope = np.ones(n_times, float)
        if onset < n_times and amp_mod_sd > 0:
            post_len = n_times - onset
            mod = rng.normal(0.0, amp_mod_sd, size=post_len)
            mod = gaussian_filter1d(mod, sigma=amp_smooth_samp, mode="reflect")
            # Renormalize to target SD
            cur_sd = np.std(mod)
            if cur_sd > 0:
                mod = mod / cur_sd * amp_mod_sd
            else:
                mod = np.zeros_like(mod)
            if amp_envelope_mode == "lognormal":
                # mean ~ 1 in linear domain
                env_post = np.exp(mod - 0.5 * (amp_mod_sd ** 2))
            else:
                raise ValueError("amp_envelope_mode must be 'lognormal'")
            envelope[onset:] = env_post

        # Apply global gain
        envelope *= amp_globals[i]

        amp_envelopes[i] = envelope
        added_evoked[i] = envelope * warped

    data_with_evoked = prestim_data + added_evoked
    return data_with_evoked, added_evoked, displacement_samples, amp_envelopes, amp_globals


def prep_simulated_data(subject, basepath, croptimes_pre, croptimes_post, channel, max_trials=np.inf, threshold1=-np.inf, threshold2=-np.inf):
    subject_path = os.path.join(basepath, subject) 
    epo_post1 = mne.read_epochs(os.path.join(subject_path, f"{subject}_calibration_post.fif"), verbose=False).pick(channel)
    post1_data = epo_post1.copy().average().pick([channel]).crop(0.02, 0.15).data
    pre1_data = epo_post1.copy().average().pick([channel]).crop(-0.5, -0.1).data
    rms1 = np.sqrt(np.mean(post1_data**2))
    rms2 = np.sqrt(np.mean(pre1_data**2))
    min_max_diff = np.max(post1_data) - np.min(post1_data)
    if min_max_diff < threshold1:
        return False
    ratio = rms1 / rms2
    if ratio < threshold2:
        return False
    
    epo_post2 = mne.read_epochs(os.path.join(subject_path, f"{subject}_fixedprep_post.fif"), verbose=False).pick(channel)
    epo_pre1 = mne.read_epochs(os.path.join(subject_path, f"{subject}_calibration_pre.fif"), verbose=False).pick(channel)
    epo_pre2 = mne.read_epochs(os.path.join(subject_path, f"{subject}_fixedprep_pre.fif"), verbose=False).pick(channel)
    epo_post = mne.concatenate_epochs([epo_post1, epo_post2]).crop(croptimes_post[0],croptimes_post[1])
    epo_pre = mne.concatenate_epochs([epo_pre1, epo_pre2]).crop(croptimes_pre[0],croptimes_pre[1])
    if len(epo_pre) > max_trials:
        epo_pre = epo_pre[:max_trials]
        epo_post = epo_post[:max_trials]
    template = np.mean(epo_post.get_data(copy=True), 0).squeeze()
    prestim_data = epo_pre.get_data(copy=True).squeeze()
    template[epo_post.times <= 0] = 0
    pre_mean = prestim_data[:, :501].mean(axis=1, keepdims=True)
    prestim_data = prestim_data - pre_mean
    onset = np.max(np.where(epo_post.times <= 0)).astype(int)
    data_with_evoked, added_evoked, displacement_samples, amp_envelopes, amp_globals = add_evoked_with_smooth_time_warp_and_amp_envelope(
    prestim_data,
    template,
    epo_post.info['sfreq'], onset=onset)
    if template.shape[0] != prestim_data.shape[-1]:
        raise ValueError("template.shape[0] should be equal to prestim_data.shape[-1]")
    rms = np.sqrt((data_with_evoked[0]**2).mean()) 
    return data_with_evoked, added_evoked, prestim_data, template, displacement_samples, amp_envelopes, amp_globals, epo_post.info['sfreq'], epo_post.times, rms
    
    
















