import numpy as np
import mne

def make_ocular_templates(common_channels):

    # create MNE info + montage (standard_1005) and get positions
    montage = mne.channels.make_standard_montage('standard_1005')
    info = mne.create_info(ch_names=common_channels, sfreq=1, ch_types='eeg')
    info.set_montage(montage)
    
    # Get 3D positions in head coordinates for each channel in order
    pos_dict = montage.get_positions()['ch_pos']
    ch_pos = np.array([pos_dict[ch] for ch in common_channels])
    x = ch_pos[:, 0] 
    y = ch_pos[:, 1] 
    z = ch_pos[:, 2] 
    
    # Normalize coordinates to [0,1] where helpful
    def _norm01(v):
        vmin, vmax = np.min(v), np.max(v)
        return (v - vmin) / (vmax - vmin)
    
    x_abs_n = _norm01(np.abs(x))
    y_n = _norm01(y)
    
    # smooth VEOG (blink) template
    # Strongest at anterior (high y), decays smoothly posteriorly.
    # Mostly symmetric left-right (VEOG), mild lateral emphasis near the eyes.
    # Small opposite polarity at far posterior.
    gamma_ap = 1.6  # curvature of anterior emphasis (higher => sharper frontal peak)
    alpha_back = 0.08  # magnitude of opposite polarity at posterior
    beta_back = 1.0    # shape of posterior term
    
    veog = (y_n ** gamma_ap) - alpha_back * ((1.0 - y_n) ** beta_back)
    
    # mild lateral emphasis to reflect proximity to eyes near Fp1/Fp2
    veog *= (0.95 + 0.15 * x_abs_n)
    
    # small boost at Fp1/Fp2/Fpz
    boost = 0.05
    for ch in ['Fp1', 'Fp2', 'Fpz']:
        if ch in common_channels:
            veog[common_channels.index(ch)] += boost
    
    
    # HEOG (horizontal) template (left-right antisymmetric)
    
    # HEOG is strongest at lateral anterior temporal/frontal sites with opposite signs.
    # smooth antisymmetric map based on x (left/right) and y (anterior) proximity.
    x_signed_n = (x - np.mean(x)) / (np.max(np.abs(x)))
    # emphasize anterior and lateral regions
    heog = x_signed_n * (0.3 + 0.7 * (y_n ** 1.2)) * (0.5 + 0.5 * (np.abs(x_signed_n) ** 0.7))
    # slight posterior taper
    heog *= (0.8 + 0.2 * y_n)
    
    # normalize and average-reference
    def _l2norm(xv):
        n = np.linalg.norm(xv)
        return xv if n == 0 else xv / n
    
    def average_reference_map(xv, renorm=True):
        xv = xv - np.mean(xv)
        if renorm:
            xv = _l2norm(xv)
        return xv
    
    veog_ar = average_reference_map(veog, renorm=True)
    heog_ar = average_reference_map(heog, renorm=True)
    
    return veog_ar, heog_ar

