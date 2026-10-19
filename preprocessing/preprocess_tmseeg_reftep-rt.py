import mne
import numpy as np
import scipy
from scipy.stats import zscore
from ssp_sir_python import *
from pulse_artifact_interpolation import *
from ocular_templates import *
from data_filtering import *
from sklearn.metrics import r2_score
from mne.bem import _fit_sphere_to_headshape
from sound_modified import *
import time
import os

def is_good_trial_pre(data, n_bad_lim, z_thresh_autocorr, z_thresh_std):
    n_ch = data.shape[0]
    channel_autocorrs = np.array([autocorr(data[ch_ind,:], 1) for ch_ind in range(data.shape[0])]) #lagged autocorrelation across samples
    channel_stds = np.std(data, axis=1) #std across samples

    #find bad channels if any
    bads_autocorr = np.asarray(iter_channel_drops(channel_autocorrs, z_thresh_autocorr), dtype=int)
    
    keep_idx = np.setdiff1d(np.arange(n_ch), bads_autocorr)
    
    if keep_idx.size:
        bads_std_sub = np.asarray(iter_channel_drops(channel_stds[keep_idx], z_thresh_std), dtype=int)
        bads_std = keep_idx[bads_std_sub]
    else:
        bads_std = np.array([], dtype=int)

    # Combine bads
    bads = np.union1d(bads_autocorr, bads_std)

    # Threshold on number of bad channels
    is_good = len(bads) <= (n_bad_lim * n_ch)
    return is_good, bads

def is_good_trial_post(data, n_bad_lim, z_thresh_auc):
    channel_aucs = np.array([np.trapezoid(np.abs(data[ch_ind,:])) for ch_ind in range(data.shape[0])]) #aucs for channels
    bads = iter_channel_drops(channel_aucs, z_thresh_auc)
    
    if len(bads) > n_bad_lim*data.shape[0]:
        return False, bads 
    else:
        return True, bads

def iter_channel_drops(ch_data, z_thresh):
    goods_idx = np.ones(len(ch_data), dtype=bool)

    while True:
        ch_data_z_good = zscore(ch_data[goods_idx])
    
        ch_data_z = np.zeros(len(ch_data))
    
        good_ind = 0
        for idx in range(len(ch_data)):
            if goods_idx[idx]:
                ch_data_z[idx] = ch_data_z_good[good_ind]
                good_ind += 1
    
        bad_indices = np.where(
            (ch_data_z < z_thresh[0]) | (ch_data_z > z_thresh[1]))[0]
    
        if len(bad_indices) > 0:
            goods_idx[bad_indices] = False
        else:
            break
    
    return np.where(~goods_idx)[0]

def is_on_ocular_activity(evoked, templates, sample_ranges, threshold1, threshold2, ptp_thresh):
    r2s = np.full((len(templates), len(sample_ranges)), np.nan, dtype=float)
    for t_ind, template in enumerate(templates):
        for s_ind, sample_range in enumerate(sample_ranges):
            data_now = evoked.data[:, sample_range[0]:sample_range[1]]
            data = np.mean(data_now, axis=1)
            x, residuals, rank, s = np.linalg.lstsq(template[:, None], data, rcond=None)
            x = x.item()  # extract scalar
            yhat = x * template
            r2 = r2_score(data, yhat)
            r2s[t_ind, s_ind] = r2
            spatial_ptp = np.ptp(data_now, axis=0)  # shape (n_times,)
            if r2 > threshold1:
                return True, r2s
            if r2 > threshold2 and np.any(spatial_ptp > ptp_thresh):
                return True, r2s
    return False, r2s
            
        

def recover_bads(evoked, bads, origin):
    evoked.info['bads'] = [evoked.ch_names[ind] for ind in bads]
    evoked.interpolate_bads(reset_bads=True, mode='accurate', origin=origin, method='spline')
    return evoked
    
    
def autocorr(data, lag=1):
    #compute autocorrelation for a given lag
    return np.corrcoef(data[:-lag],data[lag:])[0,1]
    

def preprocess_prestim_and_poststim(evoked,
                                    baseline_samples, pre_range_samples, post_range_samples, pulse_artifact_samples1, pulse_artifact_samples2,
                                    rejection_options, ssp_sir_options, L, filter_options, crop_times, resample_to, origin_head, sound_options, save_inter=False):
    
    preprocessing_info = {} #initialize dict for pre-processing information
    evoked.data -= np.mean(evoked.data[:, baseline_samples[0]:baseline_samples[1]], axis=1, keepdims=True) #baseline correct
    evoked.data = interpolate_PCHIP_evoked(evoked.data, pulse_artifact_samples1[0], pulse_artifact_samples1[1]) #interpolate pulse artifact times

    #detect bad channels separately from pre- and post-stimulus periods, interpolate them using MNE, and then set EEG to average reference
    is_good_pre, bad_channels_pre = is_good_trial_pre(evoked.data[:, pre_range_samples[0]: pre_range_samples[1]],
                                                      rejection_options['bad_proportion'],
                                                      rejection_options['z_autocorr'], rejection_options['z_std'])
    preprocessing_info['bad_channels_pre'] = bad_channels_pre
    
    if not is_good_pre:
        return False, preprocessing_info
        
    keep_idx = np.setdiff1d(np.arange(evoked.data.shape[0]), bad_channels_pre)
    
    is_good_post_rel, bad_channels_post_sub = is_good_trial_post(evoked.data[keep_idx, post_range_samples[0]: post_range_samples[1]],
                                                         rejection_options['bad_proportion'], rejection_options['z_auc'])
    # Map sub-indices back to true channel indices
    bad_channels_post = keep_idx[np.asarray(bad_channels_post_sub, dtype=int)]
    preprocessing_info['bad_channels_post'] = bad_channels_post

    if not is_good_post_rel: #check
        return False, preprocessing_info
        
    bad_channels = list(np.union1d(bad_channels_pre, bad_channels_post)) #combine information on bad channels

    #update combined bad channel information to the preprocessing info dictionary
    preprocessing_info['bad_channels'] = bad_channels
    if len(bad_channels) > rejection_options['bad_proportion']*evoked.data.shape[0]:
        return False, preprocessing_info
    if bad_channels: #interpolate bad channels if any were found
        evoked = recover_bads(evoked, np.array(bad_channels), origin_head)
    
    evoked.data -= np.mean(evoked.data, axis=0) #average reference with only good channels
        
    if sound_options['use_sound']:
        n_channels = evoked.data.shape[0]
        sigmas_init = np.ones((n_channels, 1))
        SOUND_filter, sigmas, n_iters, convergence_reached = sound(evoked.data[:, baseline_samples[1]:].T, 0, np.ones((n_channels, 1)), n_channels, L, sound_options['max_iterations'], sound_options['lambda0'], sound_options['convergence_boundary'], sound_options['fixed_iters'])
        if save_iter:
            preprocessing_info['SOUND_filter'] = SOUND_filter
            preprocessing_info['SOUND_sigmas'] = sigmas
            preprocessing_info['SOUND_n_iters'] = n_iters
            preprocessing_info['SOUND_convergence_reached'] = convergence_reached
        evoked.data = np.matmul(SOUND_filter, evoked.data)
        evoked.data -= np.mean(evoked.data, axis=0)
        
    if rejection_options['ocular']['check_ocular']:
        is_ocular, r2s = is_on_ocular_activity(evoked, rejection_options['ocular']['templates'], rejection_options['ocular']['sample_ranges_ocular'], rejection_options['ocular']['threshold1'], rejection_options['ocular']['threshold2'], rejection_options['ocular']['ptp_thresh'])
        if save_inter:
            preprocessing_info['is_ocular'] = is_ocular
            preprocessing_info['ocular_templates'] = rejection_options['ocular']['templates']
            preprocessing_info['ocular_sample_ranges'] = rejection_options['ocular']['sample_ranges_ocular']
            preprocessing_info['ocular_template_r2s'] = r2s
        if is_ocular:
            return False, preprocessing_info

    # Apply SSP-SIR
    # get the filter kernel/envelope, number of removed components, the artifact suppression matrices
    preprocessing_info['sspsir_svd_converged'] = None
    try:
        data, artifact_topographies, data_suppressed, filter_kernel, P, M, PL, sir_projmat_suppr, sir_projmat_orig, n_pcs_removed = ssp_sir_to(
            evoked.data, L, evoked.info['sfreq'], ssp_sir_options['ranges'], method=ssp_sir_options['method'])
    except np.linalg.LinAlgError:
        preprocessing_info['sspsir_svd_converged'] = False
        return False, preprocessing_info
    evoked.data = data

    #save SSP--SIR information
    if save_inter:
        preprocessing_info['sspsir_suppression_matrix_P'] = P
        preprocessing_info['sspsir_suppression_matrix_PL'] = PL
        preprocessing_info['sspsir_filter_kernel'] = filter_kernel
        preprocessing_info['sspsir_M'] = M
        preprocessing_info['sspsir_n_pcs_removed'] = n_pcs_removed
        preprocessing_info['sspsir_artifact_topographies'] = artifact_topographies
        preprocessing_info['sspsir_data_suppressed_ave'] = data_suppressed
        preprocessing_info['sspsir_sir_projmat_suppr'] = sir_projmat_suppr
        preprocessing_info['sspsir_sir_projmat_orig'] = sir_projmat_orig

    #interpolate again the artifact time window (can be different)
    evoked.data = interpolate_PCHIP_evoked(evoked.data, pulse_artifact_samples2[0], pulse_artifact_samples2[1])
    if rejection_options['check_ptp']:
        # check for exceeding ptp threshols
        # spatial peak-to-peak at each time point: max-min across channels
        spatial_ptp = np.ptp(evoked.data, axis=0)  # shape (n_times,)
        no_times = rejection_options['ptp_notimes']
        # Build exclusion mask: union of all excluded windows and times before t1
        exclude = np.zeros(evoked.times.shape, dtype=bool)
        exclude |= (evoked.times >= rejection_options['ptp_notimes'][0]) & (evoked.times <= rejection_options['ptp_notimes'][1])
        exclude |= (evoked.times < evoked.times[pre_range_samples[0]])  # exclude anything before crop_times[0]
        
        # Violations outside the excluded times
        viol_mask = (spatial_ptp > rejection_options['ptp_thresh']) & (~exclude)
        preprocessing_info['ptp_threshold_exceeded'] = False
        if np.any(viol_mask):
            preprocessing_info['ptp_threshold_exceeded'] = True
            return False, preprocessing_info

    evoked.data, _ = butter_filter(evoked.data, filter_options['cutoff'], 'bandpass',
                                        evoked.info['sfreq'], filter_options['order'], filter_options['pad_time'])
    b_start = evoked.times[baseline_samples[0]]
    b_end = evoked.times[baseline_samples[1]]
    evoked.resample(resample_to, method='polyphase')
    b_start_new = np.ceil((b_start - evoked.times[0]) * evoked.info['sfreq']).astype(int)
    b_end_new = np.floor((b_end - evoked.times[0]) * evoked.info['sfreq']).astype(int)
    evoked.data -= np.mean(evoked.data[:, b_start_new:b_end_new], axis=1, keepdims=True) #re-baseline correct
    evoked.data -= np.mean(evoked.data, axis=0) #ensure average reference

    evoked.crop(crop_times[0], crop_times[1]) #crop away edges
    
    return evoked, preprocessing_info
    

def run_subject_processing(site_id, subject_id, all_data_path, processed_path):
    # Time ranges
    baseline = [-0.5, -0.025] #baseline correction
    pre_range_times = [-0.5, -0.005] #pre trial rejection
    #pulse artifact interpolation times
    pulse_artifact_time1 = [-0.004, 0.006]
    pulse_artifact_time2 = [-0.006, 0.015]
    post_range_times = [0.02, 0.05] #used for rejection
    ocular_times = [
        [-0.5, -0.3],
        [-0.4, -0.2],
        [-0.3, -0.1],
        [-0.2,  0.0],
        [-0.1,  0.1],
        [ 0.0,  0.2],
        [ 0.1,  0.3],
        [ 0.2,  0.4],
        [ 0.3,  0.5]] #used for rejecting trials based on ocular artifacts

    # Channel and forward model setup
    common_channels = ['AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5',
                        'CP6', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5',
                          'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5',
                            'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8']
    
    montage_name = 'standard_1005'
    montage = mne.channels.make_standard_montage(montage_name)
    forward = mne.read_forward_solution("/scratch/work/aholao1/eos/preprocess_tmseeg/subjects_dir_fsaverage/fsaverage/fsaverage_eeg-fwd.fif")
    channel_order = forward.ch_names
    
    # filter options
    filter_options = {'cutoff': [2, 40], 'btype': 'bandpass', 'order': 2, 'pad_time': 0.1} #filter options

    # processing the subject
    print(f"Processing {subject_id}")
    start_time_total = time.time()

    subject_path = os.path.join(all_data_path, site_id, subject_id)
    subject_path_processed = os.path.join(processed_path, subject_id)
    subject_path_processed_epochs = os.path.join(subject_path_processed, f'{subject_id}_rt-epo.fif')
    subject_path_processed_info = os.path.join(subject_path_processed, f'{subject_id}_rt-processing_info.npy')

    # Create output directory
    os.mkdir(subject_path_processed)

    # load data and set montage
    epochs = mne.read_epochs_eeglab(os.path.join(subject_path, f'{subject_id}_task-tep_all_eeg.set'))
    epochs.pick(common_channels) #restrict channel space
    epochs.reorder_channels(channel_order) #reorder channels to match the forward solution
    epochs.set_montage(None) #set None montage to clear existing vals
    epochs.set_montage(montage) #set montage
    epochs.crop(-1.0, None) #crop from -1s before TMS onward
    radius, origin_head, origin_device = _fit_sphere_to_headshape(epochs[0].info, dig_kinds="auto")
    block_identifiers = scipy.io.loadmat(os.path.join(subject_path, f'{subject_id}_block_identifiers.mat'), simplify_cells=True)['block_identifiers_trials'] #load block idents
    ocular_templates = make_ocular_templates(common_channels) #blink and eye movement estimate topographies

    # define sample ranges to use in baseline correction, pulse artifact interpolation, and post range times
    times_to_use = np.where((epochs.times >= pulse_artifact_time1[0]) & (epochs.times <= pulse_artifact_time1[1]))[0]
    pulse_artifact_samples1 = (np.min(times_to_use), np.max(times_to_use + 1))
    times_to_use = np.where((epochs.times >= pulse_artifact_time2[0]) & (epochs.times <= pulse_artifact_time2[1]))[0]
    pulse_artifact_samples2 = (np.min(times_to_use), np.max(times_to_use + 1))
    times_to_use  = np.where((epochs.times >= baseline[0]) & (epochs.times <= baseline[1]))[0]
    baseline_samples = (np.min(times_to_use), np.max(times_to_use + 1))
    times_to_use = np.where((epochs.times >= post_range_times[0]) & (epochs.times <= post_range_times[1]))[0]
    post_range_samples = (np.min(times_to_use), np.max(times_to_use + 1))
    times_to_use  = np.where((epochs.times >= pre_range_times[0]) & (epochs.times <= pre_range_times[1]))[0]
    pre_range_samples = (np.min(times_to_use), np.max(times_to_use + 1))
    crop_times = [epochs.times[0] + 0.2, epochs.times[-1] - 0.1] #crop away edges after filtering

    ocular_samples = []
    for s, e in ocular_times:
        i0 = np.searchsorted(epochs.times, s, side="left")
        i1 = np.searchsorted(epochs.times, e, side="right")
        if i0 < i1:  # ensure non-empty
            ocular_samples.append([i0, i1])

    # Trial and channel rejection options
    rejection_options = {'z_autocorr':[-3, 10], 'z_std':[-10, 3], 'z_auc':[-np.inf, 3.5], 'bad_proportion':0.15, 
                         'ocular':{'templates': ocular_templates, 'sample_ranges_ocular': ocular_samples , 'threshold1':0.85, 'threshold2':0.75,'ptp_thresh':100*1e-6, 'check_ocular':False},
                         'ptp_thresh':300*1e-6, 'ptp_notimes':[-0.02, 0.03], 'check_ptp':False}

    # SSP-SIR options
    ssp_sir_options = {'timerange': ['automatic', 40], 'method': ['threshold', 0.99], 'selected_times':[0.007, 0.035]}
    selected_times = ssp_sir_options['selected_times']
    samples_to_use = np.where((epochs.times >= selected_times[0]) & (epochs.times <= selected_times[1]))[0]
    time_ranges_sample = (np.min(samples_to_use), np.max(samples_to_use) + 1)
    ssp_sir_ranges = [ssp_sir_options['timerange'][0], ssp_sir_options['timerange'][1], time_ranges_sample]
    ssp_sir_options['ranges'] = ssp_sir_ranges

    resample_to = 1000 #resampling freq

    sound_options = {'max_iterations':10, 'lambda0':0.01, 'convergence_boundary':1e-9, 'fixed_iters':False, 'use_sound':False}
    L = forward['sol']['data']-np.mean(forward['sol']['data'], axis=0) #leadfield
    epochs_pre_list = []
    epochs_post_list = []
    bad_trials_calibrated = []
    n_channels = len(epochs.ch_names)
    n_trials = len(epochs)
    n_times = len(epochs.times)

    preprocessing_informations = []
    epochs_list = []
    print(f"Processing subject {subject_id}")
    
    for trial_ind in range(n_trials):
        print(f"Processing trial {trial_ind}")
        epoch = epochs[trial_ind]
        evoked = mne.EvokedArray(epoch.get_data(copy=True).reshape(n_channels, n_times), info=epoch.info, tmin=epoch.times[0], kind='single_epoch')
        start = time.perf_counter() #timer for processing
        evoked, preprocessing_info = preprocess_prestim_and_poststim(evoked, baseline_samples, pre_range_samples, post_range_samples,
                                                 pulse_artifact_samples1, pulse_artifact_samples2,
                                                 rejection_options, ssp_sir_options, L, filter_options, crop_times, resample_to, origin_head, sound_options)
        elapsed = time.perf_counter() - start #end timer
        # save processing times for good and bad trials
        if evoked:
            preprocessing_info['good_time'] = elapsed
            preprocessing_info['bad_time'] = None
            epochs_list.append(mne.EpochsArray(evoked.data.reshape(1,n_channels, len(evoked.times)), info=evoked.info, tmin=evoked.times[0], baseline=baseline))
        else:
            preprocessing_info['bad_time'] = elapsed
            preprocessing_info['good_time'] = None
        preprocessing_informations.append(preprocessing_info)


    epochs = mne.concatenate_epochs(epochs_list)
    epochs.save(subject_path_processed_epochs)
    np.save(subject_path_processed_info, preprocessing_informations)
    del epochs
    del epochs_list
    del preprocessing_informations
    #save results
    

# %%
import multiprocessing as mp
def main():
    all_data_path = r"/scratch/nbe/c2b-shared/reftep_oskari"
    processed_path = r"/scratch/work/aholao1/reftep_processed_strt"
    os.mkdir(processed_path)
    args = []
    for site_id in ['Data_Tuebingen', 'Data_Aalto']:
        site_dir = os.path.join(all_data_path, site_id)
        subjects = os.listdir(site_dir)
        subjects = [subject for subject in subjects if "_rep" not in subject]
        args += [(site_id, subject_id, all_data_path, processed_path) for subject_id in subjects]
    print("Real-time compatible processing without a calibration period")
    with mp.Pool(processes=17) as pool:
        pool.starmap(run_subject_processing, args)
    #for site_id, subject_id, all_data_path, processed_path in args:
        #run_subject_processing(site_id, subject_id, all_data_path, processed_path)
        

if __name__ == "__main__":
    main()