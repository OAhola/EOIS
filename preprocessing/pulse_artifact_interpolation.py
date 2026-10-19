import numpy as np
from scipy.interpolate import PchipInterpolator

def interpolate_PCHIP(data, s_start, s_end):
    n_trials, n_channels, n_times = data.shape
    t = np.arange(n_times)
    #create mask
    mask = np.ones(n_times, dtype=bool)
    mask[s_start:s_end] = False
    #create interpolator
    interp = PchipInterpolator(t[mask], data[:,:, mask], axis=2)
    data[:, :, ~mask] = interp(t[~mask]) #interpolate data in the mask complement
    return data

def interpolate_PCHIP_evoked(data, s_start, s_end):
    n_channels, n_times = data.shape
    t = np.arange(n_times)
    #create mask
    mask = np.ones(n_times, dtype=bool)
    mask[s_start:s_end] = False
    #create interpolator
    interp = PchipInterpolator(t[mask], data[:, mask], axis=1)
    data[:, ~mask] = interp(t[~mask]) #interpolate data in the mask complement
    return data