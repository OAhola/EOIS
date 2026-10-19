#%%
import mne #for useful functions and structures
import numpy as np #for useful linear algebra functions
#import neat methods from scipy
import scipy
from scipy.stats import median_abs_deviation
from scipy.stats import zscore
from scipy.signal import butter, sosfiltfilt
import warnings
import matplotlib.pyplot as plt
from ica_calibrator import *
from ssp_sir_python import *
from channel_interpolations import *
from pulse_artifact_interpolation import *
from data_filtering import *
import time
import os


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*ICA\.apply\(\) was baseline-corrected.*"
)

#----------------------------- Finding and reconstructing bad channels EEG from epoched data ----------------------------- ----------------------------- 
def get_bad_channels_epoched(epochs, z_score_thresh_mad, z_score_thresh_power, fmin_fmax, z_score_autocorr, z_score_auc):
    bad_channels = get_bad_channels_epoched_run(epochs.copy().set_eeg_reference('average', projection=False, verbose=False), z_score_thresh_mad, z_score_thresh_power,fmin_fmax, z_score_autocorr, z_score_auc)
    if len(bad_channels) > 0: #if one or more bad channels were found then re-run the channel rejection
        while True:
            epochs_without_bad_channels = epochs.copy().drop_channels(bad_channels).set_eeg_reference('average', projection=False, verbose=False) #do not analyze the dropped channels anymore
            #get new bad channels
            bad_channels_new = get_bad_channels_epoched_run(epochs_without_bad_channels, z_score_thresh_mad, z_score_thresh_power,fmin_fmax, z_score_autocorr, z_score_auc)
            if len(bad_channels_new) == 0:
                break
            bad_channels = list(np.union1d(bad_channels, bad_channels_new)) #list of all bad channels
    return bad_channels

def get_bad_channels_epoched_run(epochs, z_score_thresh_mad, z_score_thresh_power, fmin_fmax, z_score_autocorr, z_score_auc):
    data = epochs.get_data(copy=True) # get the epoched data
    n_trials, n_channels, _ = data.shape #number of trials and channels
    bad_channel_indices = np.array([]) #initialize an empty array for later unions with bad channels detected by various metrics

    if z_score_thresh_mad:
        data_flattened = data.swapaxes(0, 1).reshape(n_channels, -1)
        #get the median absolute deviation values (more robust and not as sensitive to outliers compared to standard deviation)
        mad_values = median_abs_deviation(data_flattened, axis=1)
        #z-score the values and get the absolute values
        mad_z_scores = zscore(mad_values)
        #compare absolute z-scored values to the threshold
        bad_channel_indices_deviation = np.where((mad_z_scores > z_score_thresh_mad[1]) | (mad_z_scores < z_score_thresh_mad[0]))[0]
        print_out_channel_rejection_info(epochs, bad_channel_indices_deviation, "median absolute deviation")
        bad_channel_indices = np.union1d(bad_channel_indices, bad_channel_indices_deviation)

    if z_score_thresh_power:
        # check if channel has a lot of high-frequency power
        # get the power spectral densities
        trials_channels_psds, _ = get_psds(data,epochs.info['sfreq'],fmin_fmax[0],fmin_fmax[1])
        channel_psds = np.mean(trials_channels_psds,axis=0) #average psds for each channel
        channel_psds = np.mean(channel_psds,axis=-1) #average psds for each channel
        psd_z_scores = zscore(channel_psds) #z-scored average psds
        #which channels go over the threshold?
        bad_channel_indices_power = np.where(psd_z_scores > z_score_thresh_power)[0]
        # combine the information of bad channels
        print_out_channel_rejection_info(epochs, bad_channel_indices_power, "high frequency power")
        bad_channel_indices = np.union1d(bad_channel_indices,bad_channel_indices_power)

    if z_score_autocorr:
        #detect flat and noisy channels
        # compute autocorrelation the mean autocorrelation for each channel
        ch_autocorrelations = [np.mean([autocorr(data[trial_ind, ch_ind,:],lag=1)
                                for trial_ind in range(n_trials)]) for ch_ind in range(n_channels)]
        autocorrelation_z_scores = np.abs(zscore(ch_autocorrelations)) #z-scored mean autocorrelations
        #which channels go over the threshold?
        bad_channel_indices_autocorrelation = np.where(autocorrelation_z_scores > z_score_autocorr)[0]
        print_out_channel_rejection_info(epochs, bad_channel_indices_autocorrelation, "autocorrelation")
        # combine the information of bad channels
        bad_channel_indices = np.union1d(bad_channel_indices,bad_channel_indices_autocorrelation)
    
    if z_score_auc:
        #check if a channel is bad by calculating area under the curve distribution
        #this should be only used for post-stimulus intervals, where large decay artifacts can substantially contaminate the signal
        evoked_data = np.mean(data, axis=0)
        auc_values = [np.trapezoid(np.abs(evoked_data[ch_ind,:])) for ch_ind in range(n_channels)]
        auc_z_scores = zscore(auc_values) #z-scored aucs
        #which channels go over the threshold?
        bad_channel_indices_auc = np.where(auc_z_scores > z_score_auc)[0]
        # combine the information of bad channels
        print_out_channel_rejection_info(epochs, bad_channel_indices_auc, "area under the curve")
        bad_channel_indices = np.union1d(bad_channel_indices,bad_channel_indices_auc)
    bad_channel_indices = bad_channel_indices.astype(int)
    # get the names of the bad channels
    bad_channels = [epochs.ch_names[bad_ch_ind] for bad_ch_ind in bad_channel_indices]
    return bad_channels

def autocorr(data, lag=1):
    #compute autocorrelation for a given lag
    return np.corrcoef(data[:-lag],data[lag:])[0,1]

def get_psds(data, sfreq, fmin, fmax):
    #get the power spectral densities for all channels and trials
    psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq, fmin=fmin, fmax=fmax, verbose=False)
    return psds, freqs

def print_out_channel_rejection_info(structure, rejected_channel_indices, type_of_rejection):
    if len(rejected_channel_indices) > 0 and type_of_rejection:
        print(f"Rejected channels {[structure.ch_names[bad_ch_ind] for bad_ch_ind in rejected_channel_indices]} due to deviation in {type_of_rejection}.")
    if not type_of_rejection and len(rejected_channel_indices) == 0:
        print(f"Rejected channels {[structure.ch_names[bad_ch_ind] for bad_ch_ind in rejected_channel_indices]}.")

def get_bad_channels_pre(epochs, rejection_options, filter_options):
    epochs_data = epochs.get_data(copy=True) #get the epoched data
    epochs_data, epochs_pre_filter = butter_filter(epochs_data, filter_options['cutoff'], filter_options['btype'], epochs.info['sfreq'], filter_options['order'], filter_options['pad_time'])
    epochs_pre_filtered = mne.EpochsArray(epochs_data, epochs.info, epochs.events, tmin=epochs.times[0]) #re-create the epochs object for channel rejection
    bad_channels_pre = get_bad_channels_epoched(epochs_pre_filtered, rejection_options['z_score_thresh_mad'],
                                                 rejection_options['z_score_thresh_power'],rejection_options['fmin_fmax'], rejection_options['z_score_thresh_autocorr'], False)
    return bad_channels_pre, epochs_pre_filter

def get_bad_channels_post(epochs, rejection_options, post_range):
    #apply baseline correction using a post-stimulus time (responses should not be on here or artifacts, such as recharging artifacts)
    if post_range:
        epochs_post_true = epochs.copy().crop(post_range[0], post_range[1])
    else:
        epochs_post_true = epochs.copy()
    bad_channels_post = get_bad_channels_epoched(epochs_post_true, False, False, False, False, rejection_options['z_score_thresh_auc']) #reject channels in post-stimulus period based on area under the curve
    return bad_channels_post

def reconstruct_bad_channels(epochs, bad_channels, interpolation_info):
    epochs.info['bads'] = bad_channels #set the bad channel information to info
    if interpolation_info is None:
        interpolation_matrix, goods_idx, bads_idx = custom_get_interpolation_matrix(epochs, exclude=None, ecog=False)
        interpolation_info = {'interpolation_matrix':interpolation_matrix, 'goods_idx':goods_idx, 'bads_idx':bads_idx}
    apply_channel_interpolation(epochs, interpolation_info) #interpolate bad channels
    epochs.info['bads'] = [] #reset bad channels after interpolation

    return epochs, interpolation_info

#---------------- Functions for finding bad trials from pre- or post-stimulus EEG data -------------------------------------------------------------
def find_bad_trials(epochs, global_zscore_threshold, local_zscore_threshold, psd_trial_threshold, psd_freq_range):
    bad_trials, good_trial_stats = find_bad_trials_run(epochs, global_zscore_threshold, local_zscore_threshold, psd_trial_threshold, psd_freq_range, np.array([]))
    if len(bad_trials) > 0:
        while True:
            bad_trials_new, good_trial_stats = find_bad_trials_run(epochs, global_zscore_threshold, local_zscore_threshold, psd_trial_threshold, psd_freq_range, bad_trials)
            if len(bad_trials_new) == 0:
                break
            else:
                bad_trials = np.union1d(bad_trials, bad_trials_new)
    return list(bad_trials.astype(int)), good_trial_stats

def find_bad_trials_run(epochs, global_zscore_threshold, local_zscore_threshold, psd_trial_threshold, psd_freq_range, current_bad_trials):
    data =  epochs.get_data(copy=True) #data is n_trials x n_channels x n_samples
    n_trials = data.shape[0] #number of all trials
    trial_indices = np.arange(0, n_trials, 1).astype(int)
    good_trial_indices = np.array([ind for ind in trial_indices if ind not in current_bad_trials])
    data = data[good_trial_indices,:,:] #only the good trial indices

    if psd_trial_threshold:
        psds, _ = get_psds(data, epochs.info['sfreq'],psd_freq_range[0],psd_freq_range[1])
        psds_trials = np.mean(psds, axis=(1,2)) #mean across frequencies and channels
        z_score_psd_trials_adjusted = []
        z_score_psd_trials = zscore(psds_trials)

    mad_trials = median_abs_deviation(data, axis=(1,2)) #median abs deviation across channels and samples
    z_score_global_trials = zscore(mad_trials)


    mad_trials_channels = median_abs_deviation(data, axis=2) #median abs deviation across samples
    z_score_local_trials = zscore(mad_trials_channels, axis=1)
    z_score_global_trials_adjusted = []
    z_score_local_trials_adjusted = []
    good_ind = 0
    for i in trial_indices:
        if i in good_trial_indices:
            z_score_global_trials_adjusted.append(z_score_global_trials[good_ind])
            z_score_local_trials_adjusted.append(z_score_local_trials[good_ind])
            if psd_trial_threshold:
                z_score_psd_trials_adjusted.append(z_score_psd_trials[good_ind])
            good_ind += 1
        else: #do not re-reject bad trials
            z_score_global_trials_adjusted.append(0)
            z_score_local_trials_adjusted.append(np.zeros(z_score_local_trials.shape[1]))
            if psd_trial_threshold:
                z_score_psd_trials_adjusted.append(0)
    
    if len(z_score_local_trials_adjusted):                          # at least one good trial left
        z_score_local_trials_adjusted = np.vstack(z_score_local_trials_adjusted)
    else:                                                           # corner case: nothing to stack
        z_score_local_trials_adjusted = np.zeros(
            (len(trial_indices), mad_trials_channels.shape[1])
        )

    z_score_global_trials_adjusted = np.asarray(z_score_global_trials_adjusted)
    if psd_trial_threshold:
        z_score_psd_trials_adjusted = np.asarray(z_score_psd_trials_adjusted)


    # check which trials exceed global_zscore_threshold
    bad_trials_local = np.where(np.any(np.abs(z_score_local_trials_adjusted) > local_zscore_threshold, axis=1))[0]
    bad_trials = np.where((np.array(z_score_global_trials_adjusted) > global_zscore_threshold[1]) | (np.array(z_score_global_trials_adjusted) < global_zscore_threshold[0]))[0]

    bad_trials_mad_current = np.where((z_score_global_trials > global_zscore_threshold[1]) | (z_score_global_trials < global_zscore_threshold[0]))[0]
    good_indices_mad = np.array([index for index in range(len(mad_trials)) if index not in bad_trials_mad_current])
    good_trial_stats = {'mads': mad_trials[good_indices_mad], 'mads_std': np.std(mad_trials[good_indices_mad]), 'mads_mean': np.mean(mad_trials[good_indices_mad])} #compute statistics of good trials

    bad_trials = np.union1d(bad_trials, bad_trials_local) #add the bad "globally and locally" detected bad trials
    if psd_trial_threshold:
        #compute statistics of good trials
        bad_trials_psd = np.where(np.array(z_score_psd_trials_adjusted) > psd_trial_threshold)[0]
        bad_trials_psd_current = np.where(z_score_psd_trials > psd_trial_threshold)[0]
        good_indices_psd = np.array([index for index in range(len(psds_trials)) if index not in bad_trials_psd_current])
        good_trial_stats['psds'] = psds_trials[good_indices_psd] #accepted PSDs of the trials
        good_trial_stats['psds_std'] = np.std(psds_trials[good_indices_psd])
        good_trial_stats['psds_mean'] = np.mean(psds_trials[good_indices_psd])

        bad_trials = np.union1d(bad_trials, bad_trials_psd)

    return bad_trials, good_trial_stats


def get_bad_ocular_trials(ica, epochs, tmin, tmax, ocular_artifact_indices, z_thresh):
    ica_source_tcs = ica.get_sources(epochs) #get the ica sources
    ica_tcs = ica_source_tcs.get_data(copy=True) #the the ica source time courses

    #check if tmin and tmax have been defined to something else than None and set them to values that wont cause problems
    if tmax is None:
        tmax = np.inf
    if tmin is None:
        tmin = -np.inf

    time_indices_of_interest = np.where((ica_source_tcs.times >= tmin) & (ica_source_tcs.times <= tmax))[0] #time indices to check for ocular artifacts for
    bad_trials = np.array([]) #init array for bad trials
    ic_thresholds = {} #store distribution information of time courses here

    for comp_ind in ocular_artifact_indices: #go through the ocular artifact indices
        abs_ica_tcs = np.abs(ica_tcs[:, comp_ind, :])          # shape: (n_trials, n_times)
        n_trials, n_times = abs_ica_tcs.shape                  # cache dims once
        ocular_comp_ts = abs_ica_tcs.ravel()                       # ravel
        z_scored = zscore(ocular_comp_ts).reshape(n_trials, n_times) #z-scored and reshaped time courses (n_good_trials x n_times)

        assert z_scored.shape == (n_trials, n_times), (
            f"Shape mismatch after z-scoring: got {z_scored.shape}, "
            f"expected {(n_trials, n_times)}"
        )

        z_scores_in_range = z_scored[:, time_indices_of_interest] #z-scores within the time range of interest (n_good_trials x n_times_in_time_range_of_interest (=len(time_indices_of_interest)))
        median_z_scores = np.median(z_scores_in_range, axis=1) #check whether the median z-score exceeds the threshold
        #adjust the size of median_z_scores to match the number of trials for rejection
        bad_trial_inds_now = np.where(median_z_scores > z_thresh)[0]
        ic_thresholds[comp_ind] = {'std':np.std(ocular_comp_ts), 'mean': np.mean(ocular_comp_ts), 'time_indices_of_interest':time_indices_of_interest} #the maximum accepted median
        bad_trials = np.union1d(bad_trials, bad_trial_inds_now)

    return list(bad_trials.astype(int)), ic_thresholds #return the bad ocular trials and the number of iterations the method took to converge


def drop_trials_from_structs(epochs_objects, bad_trials):
    for epochs in epochs_objects:
        epochs.drop(bad_trials) #directly drop bad trials
    return epochs_objects


def preprocess_prestim_and_poststim(epochs, pre_range_times, post_range_times, channel_rejection_options, ica_options, trial_rejection_options,
                                     ssp_sir_options, L, resample_to, filter_options, bandstop_options_pre, filter_options_post, bandstop_options_post):
    
    preprocessing_info = {} #initialize dict for pre-processing information
    pre_timerange_ica = ica_options['pre_timerange']
    pre_range = pre_range_times['pre_range_final']
    #extract the post-stimulus time range information from the dict
    post_range = post_range_times['post_range']
    post_range_baseline = post_range_times['baseline']
    post_range_reject_trials = post_range_times['post_range_reject_trials']
    post_range_reject_channels = post_range_times['post_range_reject_channels']

    #crop epochs separately for pre- and post-time ranges
    epochs_pre = epochs.copy().crop(pre_range[0], pre_range[1])
    epochs_pre_ica = epochs.copy().crop(pre_timerange_ica[0], pre_timerange_ica[1])
    epochs_post = epochs.copy().crop(post_range[0], post_range[1])
    del epochs # not needed anymore because pre- and post-stim for EEG are preprocessed separately

    #epochs_pre.resample(resample_to, method='polyphase') #downsample data
    #epochs_pre_ica.resample(resample_to, method='polyphase') #downsample data
    #epochs_post.resample(resample_to, method='polyphase') #downsample data

    epochs_post.apply_baseline(post_range_baseline)
    epochs_post_data = epochs_post.get_data(copy=True)
    epochs_post_data = interpolate_PCHIP(epochs_post_data, post_range_times['pulse_artifact_samples1'][0], post_range_times['pulse_artifact_samples1'][1])
    epochs_post = mne.EpochsArray(epochs_post_data, epochs_post.info, events=epochs_post.events, tmin=epochs_post.times[0])
    #epochs_post = mne.preprocessing.fix_stim_artifact(epochs_post, tmin=pulse_artifact_time1[0], tmax=pulse_artifact_time1[1], mode='window') #reconstruct pulse artifact window

    #detect bad channels separately from pre- and post-stimulus periods, interpolate them using MNE, and then set EEG to average reference
    bad_channels_pre, pre_stim_filter = get_bad_channels_pre(epochs_pre, channel_rejection_options['pre'], filter_options)
    preprocessing_info['pre_stim_filter'] = pre_stim_filter #save the filter used for pre-stim
    bad_channels_post = get_bad_channels_post(epochs_post, channel_rejection_options['post'], post_range_reject_channels)
    bad_channels = list(np.union1d(bad_channels_pre, bad_channels_post)) #combine information on bad channels

    #update bad channel information to the preprocessing info dictionary
    preprocessing_info['channels_before_rejection'] = epochs_post.ch_names
    preprocessing_info['bad_channels_pre'] = bad_channels_pre
    preprocessing_info['bad_channels_post'] = bad_channels_post
    preprocessing_info['bad_channels'] = bad_channels
    
    if bad_channels: #interpolate bad channels if any were found
        epochs_pre, interpolation_info = reconstruct_bad_channels(epochs_pre, bad_channels, interpolation_info=None)
        epochs_pre_ica, _ = reconstruct_bad_channels(epochs_pre_ica, bad_channels, interpolation_info=interpolation_info)
        epochs_post, _ = reconstruct_bad_channels(epochs_post, bad_channels,interpolation_info=interpolation_info)
    else:
        interpolation_info = None
    preprocessing_info['channel_interpolation_info'] = interpolation_info
    
    #Run ICA only on pre-stim that has been prepared for icalabel and apply the ica filters to the pre-stim and post-stim
    epochs_pre_ica_data = epochs_pre_ica.get_data(copy=True)
    epochs_pre_ica_data, epochs_pre_ica_filter = butter_filter(epochs_pre_ica_data, ica_options['filtering']['cutoff'], 'bandpass', epochs_pre_ica.info['sfreq'], ica_options['filtering']['order_bandpass'], ica_options['filtering']['pad_time_bandpass'])
    preprocessing_info['epochs_pre_ica_filter'] = epochs_pre_ica_filter #save the filter used for pre-stim
    #create an mne.Epochs structure of the filtered data (note that the info structure is now inconsistent due to filtering outside of MNE Python)
    epochs_pre_ica_filtered = mne.EpochsArray(epochs_pre_ica_data, info=epochs_pre_ica.info, events=epochs_pre_ica.events, tmin=epochs_pre_ica.times[0])

    #set EEGs reference to average
    epochs_post.set_eeg_reference('average', projection=False, verbose=False)
    epochs_pre_ica_filtered.set_eeg_reference('average', projection=False, verbose=False)

    #get the optimal number of ICS to use in ICA
    #Get the ica structure fitted to epochs_pre that has been filtered with n_components
    ica, components_excluded, ic_label_dict = get_ica(epochs_pre_ica_filtered, ica_options['pc_threshold'], None, ica_options['bad_component_thresholds'], ica_options['n_min_comps_to_reject'], ica_options['thresh_min_comps_to_reject'])
    preprocessing_info['ica_comps_excluded'] = components_excluded #save the excluded components and their labels
    preprocessing_info['ic_label_dict'] = ic_label_dict #iclabel dictionary
    del epochs_pre_ica_filtered #not needed anymore after ICA calibration, save memory 

    #reference and filter epochs_pre
    epochs_pre_data = epochs_pre.get_data(copy=True)
    epochs_pre_data = apply_filter_to_data(epochs_pre_data, pre_stim_filter, filter_options['pad_time'], epochs_pre.info['sfreq']) #corrected order of arguments
    if bandstop_options_pre['apply_filter']:
        epochs_pre_data, epochs_pre_filter_bandstop = butter_filter(epochs_pre_data, bandstop_options_pre['cutoff'], 'bandstop',
                                                                        epochs_pre.info['sfreq'], bandstop_options_pre['order'], bandstop_options_pre['pad_time'])
        preprocessing_info['pre_stim_filter_bandstop'] = epochs_pre_filter_bandstop #save the bandstop filter used for pre-stim
    epochs_pre = mne.EpochsArray(epochs_pre_data, info=epochs_pre.info, events=epochs_pre.events, tmin=epochs_pre.times[0])
    epochs_pre.set_eeg_reference('average', projection=False, verbose=False)
    del epochs_pre_data #save memory by deleting

    #bad_ocular_trials_post, ocular_thresholds_post = get_bad_ocular_trials(ica, epochs_post, trial_rejection_options['ocular']['post_timerange'][0],
                                                    #trial_rejection_options['ocular']['post_timerange'][1], components_excluded['eye blink'], trial_rejection_options['ocular']#['z_thresh'])
    
    #combine and save information of trials with ocular artifacts near the TMS pulse
    #preprocessing_info['ocular_thresholds_post'] = ocular_thresholds_post
    #preprocessing_info['bad_trials_ocular'] = bad_ocular_trials_post
    #preprocessing_info['trials_before_ocular_rejection'] = epochs_pre.get_data(copy=True).shape[0]

    #now that the bad trials (regarding ocular artifacts) have been identified, suppress the bad components with ica
    ica.apply(epochs_pre) #apply ica to epochs_pre
    ica.apply(epochs_post) #apply ica to epochs_post

    #drop the bad ocular trials
    #epochs_pre, epochs_post = drop_trials_from_structs([epochs_pre, epochs_post], preprocessing_info['bad_trials_ocular'])
    epochs_pre.resample(resample_to, method='polyphase') #downsample data
    epochs_pre.set_eeg_reference('average', projection=False, verbose=False) #set pre-stimulus EEG reference to average

    #find bad trials from the pre-stim period
    bad_trials_pre, good_trial_stats_pre = find_bad_trials(epochs_pre, trial_rejection_options['pre']['global_zscore_threshold'],
                                                            trial_rejection_options['pre']['local_zscore_threshold'], False, False)
    preprocessing_info['trials_before_pre_eeg_rejection'] = epochs_pre.get_data(copy=True).shape[0]
    preprocessing_info['bad_trials_pre'] = bad_trials_pre
    preprocessing_info['good_trial_stats_pre'] = good_trial_stats_pre #add statistics on accepted good trials from pre-stim
    
    epochs_pre, epochs_post = drop_trials_from_structs([epochs_pre, epochs_post], bad_trials_pre)



    epochs_post.apply_baseline(post_range_baseline).set_eeg_reference('average', projection=False, verbose=False) #baseline correct the data after ICA has been applied and set to average reference
    # Apply SSP-SIR
    # get the filter kernel, number of removed components, the artifact suppression matrices, ...
    selected_times = ssp_sir_options['selected_times']
    samples_to_use = np.where((epochs_post.times >= selected_times[0]) & (epochs_post.times <= selected_times[1]))[0]
    time_ranges_sample = (np.min(samples_to_use), np.max(samples_to_use) + 1)
    ranges = [ssp_sir_options['timerange'][0], ssp_sir_options['timerange'][1], time_ranges_sample]
    epochs_post_data = epochs_post.get_data(copy=True)
    evoked_post_data = np.mean(epochs_post_data, axis=0) #mean across trials
  
    data_corrected, artifact_topographies, data_suppressed, filter_kernel, P, M, PL, sir_projmat_suppr, sir_projmat_orig, n_pcs_removed = ssp_sir_to(
        evoked_post_data, L, epochs_post.info['sfreq'], ranges, method=ssp_sir_options['method'])
   
    epochs_post_data = ssp_sir_trials(epochs_post_data, P, sir_projmat_suppr, sir_projmat_orig, filter_kernel) #apply SSP-SIR for single trials
    #save SSP--SIR information
    preprocessing_info['sspsir_suppression_matrix_P'] = P
    preprocessing_info['sspsir_suppression_matrix_PL'] = PL
    preprocessing_info['sspsir_filter_kernel'] = filter_kernel
    preprocessing_info['sspsir_M'] = M
    preprocessing_info['sspsir_n_pcs_removed'] = n_pcs_removed
    preprocessing_info['sspsir_data_corrected_ave'] = data_corrected
    preprocessing_info['sspsir_artifact_topographies'] = artifact_topographies
    preprocessing_info['sspsir_data_suppressed_ave'] = data_suppressed
    preprocessing_info['sspsir_sir_projmat_suppr'] = sir_projmat_suppr
    preprocessing_info['sspsir_sir_projmat_orig'] = sir_projmat_orig

    #interpolate again the artifact time window (can be different)
    epochs_post_data = interpolate_PCHIP(epochs_post_data, post_range_times['pulse_artifact_samples2'][0], post_range_times['pulse_artifact_samples2'][1])
    
    #re-create the post-stimulus epochs structure
    epochs_post = mne.EpochsArray(epochs_post_data, epochs_post.info, events=epochs_post.events, tmin=epochs_post.times[0])
    
    if filter_options_post['apply_filter']:
        epochs_post_data = epochs_post.get_data(copy=True)
        epochs_post_data, epochs_post_filter_bandpass = butter_filter(epochs_post_data, filter_options_post['cutoff'], 'bandpass',
                                                                       epochs_post.info['sfreq'], filter_options_post['order'], filter_options_post['pad_time'])
        preprocessing_info['epochs_post_bandpass'] = epochs_post_filter_bandpass
        if bandstop_options_post['apply_filter']:
            epochs_post_data, epochs_post_filter_bandstop = butter_filter(epochs_post_data, bandstop_options_post['cutoff'], 'bandstop',
                                                                        epochs_post.info['sfreq'], bandstop_options_post['order'], bandstop_options_post['pad_time'])
            preprocessing_info['epochs_post_bandstop'] = epochs_post_filter_bandstop
        epochs_post = mne.EpochsArray(epochs_post_data, info=epochs_post.info, events=epochs_post.events, tmin=epochs_post.times[0])
    epochs_post.resample(resample_to, method='polyphase') #downsample data
    epochs_post.apply_baseline(post_range_baseline).set_eeg_reference('average', projection=False, verbose=False) #set the EEG to average reference

    #find bad trials from the post-stimulus period and drop them from the pre- and post-stimulus epochs
    bad_trials_post, good_trial_stats_post = find_bad_trials(epochs_post.copy().crop(post_range_reject_trials[0],post_range_reject_trials[1]), trial_rejection_options['post']['global_zscore_threshold'],
                                       trial_rejection_options['post']['local_zscore_threshold'], False, False)
    preprocessing_info['trials_before_post_eeg_rejection'] = epochs_post.get_data(copy=True).shape[0]
    preprocessing_info['bad_trials_post'] = bad_trials_post
    preprocessing_info['good_trial_stats_post'] = good_trial_stats_post
    #drop the same trials from all structures
    epochs_pre, epochs_post = drop_trials_from_structs([epochs_pre, epochs_post], bad_trials_post)

    return epochs_pre, epochs_post, ica, preprocessing_info


def get_z_scores_of_val(val, mean, std):
    return (val - mean)/std #return the z-score of the value given an existing distribution

def preprocess_pre_trial_with_calibrated_data(epoch_pre, preprocessing_info, trial_rejection_options, ica, filter_options, bandstop_options, resample_to):
    """
    Preprocesses a single pre-stimulus trial using calibrated parameters,
    with detailed timing for each step.
    """
    performance_times = {}
    total_start_time = time.perf_counter()

    # Reconstruct Bad Channels
    if preprocessing_info['bad_channels']:
        start_time = time.perf_counter()
        epoch_pre, _ = reconstruct_bad_channels(
            epoch_pre,
            preprocessing_info['bad_channels'],
            preprocessing_info['channel_interpolation_info']
        )
        duration = (time.perf_counter() - start_time) * 1e3
        performance_times['Reconstruct Bad Channels'] = duration

    # Get Data
    epoch_pre_data = epoch_pre.get_data(copy=True)

    # Apply Filter
    start_time = time.perf_counter()
    epoch_pre_data = apply_filter_to_data(
        epoch_pre_data,
        preprocessing_info['pre_stim_filter'],
        filter_options['pad_time'],
        epoch_pre.info['sfreq']
        
    )
    if bandstop_options['apply_filter']:
        epoch_pre_data = apply_filter_to_data(
            epoch_pre_data,
            preprocessing_info['pre_stim_filter_bandstop'],
            bandstop_options['pad_time'],
            epoch_pre.info['sfreq']
            
        )
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Apply Filter(s)'] = duration
    #print(f"Step 2. 'Apply Filter(s): {duration:.3f}ms")

    epoch_pre = mne.EpochsArray(
        epoch_pre_data,
        info=epoch_pre.info,
        events=epoch_pre.events,
        tmin=epoch_pre.times[0],
        verbose=False
    )
    start_time = time.perf_counter()
    epoch_pre.set_eeg_reference('average', projection=False, verbose=False)
    ica.apply(epoch_pre)
    epoch_pre.resample(resample_to, method='polyphase') #downsample data
    epoch_pre.set_eeg_reference('average', projection=False, verbose=False)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Apply average refs and ica refs and ica'] = duration
    epoch_pre_data = epoch_pre.get_data(copy=True)
    
    # Global MAD Check
    start_time = time.perf_counter()
    mad_trial = median_abs_deviation(epoch_pre_data, axis=(1, 2))[0]
    
    z_mad = get_z_scores_of_val(
        mad_trial,
        preprocessing_info['good_trial_stats_pre']['mads_mean'],
        preprocessing_info['good_trial_stats_pre']['mads_std']
    )
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['4. Global MAD Check'] = duration
    #print(f"Step 4. 'Global MAD Check': {duration:.3f}ms")
    if z_mad < trial_rejection_options['pre']['global_zscore_threshold'][0] or z_mad > trial_rejection_options['pre']['global_zscore_threshold'][1]:
        print(f"rejected because of {z_mad}")
        #print(f"Rejected trial due to global MAD z-score {z_mad:.2f} outside threshold {trial_rejection_options['pre']['global_zscore_threshold']}")
        performance_times['Total (rejected)'] = (time.perf_counter() - total_start_time) * 1e3
        return False, performance_times

    # Local MAD Check
    start_time = time.perf_counter()
    mad_trials_channels = median_abs_deviation(epoch_pre_data, axis=2)
    z_score_local_trial = zscore(mad_trials_channels, axis=1)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Local MAD Check'] = duration
    #print(f"Step 5. 'Local MAD Check': {duration:.3f}ms")
    if np.any(np.abs(z_score_local_trial) > trial_rejection_options['pre']['local_zscore_threshold']):
        #print(f"Rejected trial due to local MAD z-score min/max {np.min(z_score_local_trial):.2f}/{np.max(z_score_local_trial):.2f} outside ±{trial_rejection_options['pre']['local_zscore_threshold']}")
        performance_times['Total (rejected)'] = (time.perf_counter() - total_start_time) * 1e3
        return False, performance_times

    # Final total time
    total_duration = (time.perf_counter() - total_start_time) * 1e3
    performance_times['Total (accepted)'] = total_duration

    return epoch_pre, performance_times



def preprocess_post_trial_with_calibrated_data(epoch_post, preprocessing_info, trial_rejection_options, ica, post_range_baseline, pulse_artifact_samples1, pulse_artifact_samples2, post_range_reject_trials,filter_options_post,bandstop_options_post, resample_to):
    #print("\n--- Starting single trial post-stimulus processing ---")
    performance_times = {}
    total_start_time = time.perf_counter()

    #  Baseline Correction
    start_time = time.perf_counter()
    epoch_post.apply_baseline(post_range_baseline)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Baseline Correction'] = duration

    # Interpolate Pulse Artifact (First Window)
    start_time = time.perf_counter()
    #epoch_post = mne.preprocessing.fix_stim_artifact(epoch_post, tmin=pulse_artifact_time1[0], tmax=pulse_artifact_time1[1], mode='window')
    epoch_post_data = epoch_post.get_data(copy=True)
    epoch_post_data = interpolate_PCHIP(epoch_post_data, pulse_artifact_samples1[0], pulse_artifact_samples1[1])
    epoch_post = mne.EpochsArray(epoch_post_data, epoch_post.info, events=epoch_post.events, tmin=epoch_post.times[0])
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Pulse Artifact Interpolation 1'] = duration

    # Reconstruct Bad Channels
    if preprocessing_info['bad_channels']:
        start_time = time.perf_counter()
        epoch_post, _ = reconstruct_bad_channels(epoch_post, preprocessing_info['bad_channels'], preprocessing_info['channel_interpolation_info'])
        duration = (time.perf_counter() - start_time) * 1e3
        performance_times['Reconstruct Bad Channels'] = duration

    # Set EEG Reference
    start_time = time.perf_counter()
    epoch_post.set_eeg_reference('average', projection=False, verbose=False)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Set EEG Reference'] = duration
    
    """
    # Check ICA for Ocular Artifact
    start_time = time.perf_counter()
    ica_source_tc = ica.get_sources(epoch_post)
    ica_tc = ica_source_tc.get_data(copy=True)
    for comp_ind in preprocessing_info['ocular_thresholds_post']:
        comp_ind_dict = preprocessing_info['ocular_thresholds_post'][comp_ind]
        z_score_comp = get_z_scores_of_val(np.abs(ica_tc[0, comp_ind, comp_ind_dict['time_indices_of_interest']]), comp_ind_dict['mean'], comp_ind_dict['std'])
        if np.median(z_score_comp) > trial_rejection_options['ocular']['z_thresh']:
            #print(f"Rejected trial due to near-TMS ocular activity in post: {np.median(z_score_comp)} exceeded threshold {trial_rejection_options['ocular']['z_thresh']}")
            performance_times['Total (rejected)'] = (time.perf_counter() - total_start_time) * 1e3
            #print(f"--- Total trial processing time (rejected): {performance_times['Total (rejected)'] / 1e3:.4f}s ---")
            return False, performance_times
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['5. ICA Ocular Check'] = duration
    #print(f"Step 5 'ICA Ocular Check': {duration:.3f}ms")
    """
    # Apply ICA
    start_time = time.perf_counter()
    ica.apply(epoch_post)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Apply ICA'] = duration

    # Re-baseline and Crop
    start_time = time.perf_counter()
    epoch_post.apply_baseline(post_range_baseline).set_eeg_reference('average', projection=False, verbose=False)
    #epoch_post.crop(preprocessing_info['post_mintime'], None)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Re-baseline and reference'] = duration

    # Apply SSP-SIR
    start_time = time.perf_counter()
    epoch_post_data = epoch_post.get_data(copy=True)
    epoch_post_data = ssp_sir_single_trial(epoch_post_data[0,:,:], preprocessing_info['sspsir_suppression_matrix_P'], preprocessing_info['sspsir_sir_projmat_suppr'],
                    preprocessing_info['sspsir_sir_projmat_orig'], preprocessing_info['sspsir_filter_kernel'])
    #epoch_post = mne.EpochsArray(epoch_post_data.reshape(1, epoch_post_data.shape[0], epoch_post_data.shape[1]), epoch_post.info, events=epoch_post.events, tmin=epoch_post.times[0])
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Apply SSP-SIR'] = duration

    # Interpolate Pulse Artifact (Second Window)
    start_time = time.perf_counter()
    epoch_post_data = interpolate_PCHIP(epoch_post_data.reshape(1, epoch_post_data.shape[0], epoch_post_data.shape[1]), pulse_artifact_samples2[0], pulse_artifact_samples2[1])
    epoch_post = mne.EpochsArray(epoch_post_data, epoch_post.info, events=epoch_post.events, tmin=epoch_post.times[0])
    #epoch_post = mne.preprocessing.fix_stim_artifact(epoch_post, tmin=pulse_artifact_time2[0], tmax=pulse_artifact_time2[1], mode='window')
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Pulse Artifact Interpolation 2'] = duration

    #filter the data if the option is set to True
    start_time = time.perf_counter()
    if filter_options_post['apply_filter']:
        epoch_post_data = epoch_post.get_data(copy=True)
        epoch_post_data = apply_filter_to_data(epoch_post_data,preprocessing_info['epochs_post_bandpass'], filter_options_post['pad_time'], epoch_post.info['sfreq'])
        if bandstop_options_post['apply_filter']:
            epoch_post_data = apply_filter_to_data(epoch_post_data, preprocessing_info['epochs_post_bandstop'], bandstop_options_post['pad_time'], epoch_post.info['sfreq']) 
        epoch_post = mne.EpochsArray(epoch_post_data, info=epoch_post.info, events=epoch_post.events, tmin=epoch_post.times[0])
    epoch_post.resample(resample_to, method='polyphase') #downsample data
    epoch_post.apply_baseline(post_range_baseline).set_eeg_reference('average', projection=False, verbose=False) #set the EEG to average reference
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Filtering (if done) and baseline with second average ref 2'] = duration

    # Global MAD Check
    start_time = time.perf_counter()
    epoch_post_data = epoch_post.copy().crop(post_range_reject_trials[0], post_range_reject_trials[1]).get_data(copy=True)
    mad_trial = median_abs_deviation(epoch_post_data, axis=(1,2))[0]
    z_mad = get_z_scores_of_val(mad_trial, preprocessing_info['good_trial_stats_post']['mads_mean'], preprocessing_info['good_trial_stats_post']['mads_std'])
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Global MAD Check'] = duration
    #print(f"Step 12 'Global MAD Check': {duration:.3f}ms")
    if z_mad < trial_rejection_options['post']['global_zscore_threshold'][0] or z_mad > trial_rejection_options['post']['global_zscore_threshold'][1]:
        print(f"Rejected trial due to median absolute deviation in post: {z_mad} exceeded thresholds {trial_rejection_options['post']['global_zscore_threshold']}")
        performance_times['Total (rejected)'] = (time.perf_counter() - total_start_time) * 1e3
        #print(f"--- Total trial processing time (rejected): {performance_times['Total (rejected)'] / 1e3:.4f}s ---")
        return False, performance_times

    # Local MAD Check
    start_time = time.perf_counter()
    mad_trials_channels = median_abs_deviation(epoch_post_data, axis=2)
    z_score_local_trial = zscore(mad_trials_channels, axis=1)
    duration = (time.perf_counter() - start_time) * 1e3
    performance_times['Local MAD Check'] = duration
    #print(f"Step 13 'Local MAD Check': {duration:.3f}ms")
    if np.any(np.abs(z_score_local_trial) > trial_rejection_options['post']['local_zscore_threshold']):
        #print(f"Rejected trial due to local median absolute deviation in post: {np.min(z_score_local_trial)} or {np.max(z_score_local_trial)} exceeded thresholds ±{trial_rejection_options['post']['local_zscore_threshold']}")
        performance_times['Total (rejected)'] = (time.perf_counter() - total_start_time) * 1e3
        #print(f"--- Total trial processing time (rejected): {performance_times['Total (rejected)'] / 1e3:.4f}s ---")
        return False, performance_times

    # Final total time
    total_duration = (time.perf_counter() - total_start_time) * 1e3
    performance_times['Total (accepted)'] = total_duration
    print(f"--- Total trial processing time (accepted): {total_duration / 1e3:.4f}s ---")

    return epoch_post, performance_times

    


def run_subject_processing(site_id, subject_id, all_data_path, processed_path, save_results):
    """
    Main logic to preprocess data for a single subject.
    """

    # Time ranges
    pre_range = [-1.1, -0.005]
    post_range = [-1.1, None]
    post_range_baseline = [-0.5, -0.025]
    post_range_reject_trials = [0.02, 0.3]
    post_range_reject_channels = [0.02, 0.05]
    pulse_artifact_time1 = [-0.004, 0.006]
    pulse_artifact_time2 = [-0.006, 0.015]
    post_range_times = {'post_range': post_range, 'baseline': post_range_baseline,
                         'post_range_reject_trials': post_range_reject_trials, 'post_range_reject_channels': post_range_reject_channels}
    pre_range_times = {'pre_range_final': pre_range}

    # Frequency ranges
    freq_range_suspect_channels = [None, None]
    freq_range_suspect_trials = [None, None]
    
    # Channel rejection options#
    channel_rejection_options_pre = {'z_score_thresh_mad': [-6, 4], 'z_score_thresh_power': False, 'fmin_fmax': freq_range_suspect_channels, 'z_score_thresh_autocorr': 4}
    channel_rejection_options_post = {'z_score_thresh_auc': 2.5}
    channel_rejection_options = {'pre': channel_rejection_options_pre, 'post': channel_rejection_options_post}

    # ICA options
    ica_options = {'pc_threshold': 0.99, 'bad_component_thresholds': {'eye blink': 0.9}, 'n_min_comps_to_reject': {'eye blink': 2},
                    'thresh_min_comps_to_reject': {'eye blink': 0.7}, 'pre_timerange': [-1.1, -0.005],
                      'filtering': {'order_bandpass': 2, 'order_bandstop': 2, 'pad_time_bandpass': 0.15, 'cutoff': [2, 90]}}

    # SSP-SIR options
    ssp_sir_options = {'timerange': ['automatic', 40], 'method': ['threshold', 0.999], 'selected_times':[0.0, 0.04]}

    # Trial rejection options
    trial_rejection_options = {'ocular': {'z_thresh': 2, 'pre_timerange_min': False, 'post_timerange': [0.011, 0.2]},
                                'pre': {'global_zscore_threshold': [-8, 5], 'local_zscore_threshold': 5, 'psd_zscore_threshold': False,
                                         'psd_freq_range': freq_range_suspect_trials}, 'post': {'global_zscore_threshold': [-np.inf, 6],
                                                                                                 'local_zscore_threshold': 5}}


    # Channel and forward model setup
    common_channels = ['AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5',
                        'CP6', 'CPz', 'Cz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5',
                          'FC6', 'FT7', 'FT8', 'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 'O2', 'Oz', 'P1', 'P2', 'P3', 'P4', 'P5',
                            'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz', 'T7', 'T8', 'TP7', 'TP8']
    montage_name = 'standard_1005'
    montage = mne.channels.make_standard_montage(montage_name)
    forward = mne.read_forward_solution("/scratch/work/aholao1/eos/preprocess_tmseeg/subjects_dir_fsaverage/fsaverage/fsaverage_eeg-fwd.fif")
    L = forward['sol']['data'] - np.mean(forward['sol']['data'], axis=0) #leadfield in average reference
    channel_order = forward.ch_names
    
    # filter options
    filter_options_pre = {'cutoff': [2, 40], 'btype': 'bandpass', 'order': 2, 'pad_time': 0.1} #pre-stimulus filter options
    bandstop_options_pre = {'cutoff': [48, 52], 'btype': 'bandstop', 'order': 2, 'pad_time': 0.1, 'apply_filter':False} #for powerline noise
    bandstop_options_post = {'cutoff': [48, 52], 'btype': 'bandstop', 'order': 2, 'pad_time': 0.1, 'apply_filter':False} #for powerline noise
    filter_options_post = {'cutoff': [2, 40], 'btype': 'bandpass', 'order': 2, 'pad_time': 0.1, 'apply_filter': True} #"post-stimulus" filter options and whether to apply them at all
    
    # resample to
    resample_to = 1000
    
    #n trials used in calibration (from the start, with respect to the current epochs structure)
    n_trials_use = 200

    # processing the subject
    print(f"Processing {subject_id}")
    start_time_total = time.time()

    subject_path = os.path.join(all_data_path, site_id, subject_id)
    subject_path_processed = os.path.join(processed_path, subject_id)

    # Create output directory
    os.mkdir(subject_path_processed)

    # load data and set montage
    epochs = mne.read_epochs_eeglab(os.path.join(subject_path, f'{subject_id}_task-tep_all_eeg.set'))
    epochs.pick(common_channels) #restrict channel space
    epochs.reorder_channels(channel_order) #reorder channels
    epochs.set_montage(None) #set None montage
    epochs.set_montage(montage) #set montage
    block_identifiers = scipy.io.loadmat(os.path.join(subject_path, f'{subject_id}_block_identifiers.mat'), simplify_cells=True)['block_identifiers_trials'] #load block idents


    times_post_range = epochs.copy().crop(post_range[0], post_range[1]).times
    times_to_use = np.where((times_post_range >= pulse_artifact_time1[0]) & (times_post_range <= pulse_artifact_time1[1]))[0]
    pulse_artifact_samples1 = (np.min(times_to_use), np.max(times_to_use + 1))
    times_to_use = np.where((times_post_range >= pulse_artifact_time2[0]) & (times_post_range <= pulse_artifact_time2[1]))[0]
    pulse_artifact_samples2 = (np.min(times_to_use), np.max(times_to_use + 1))
    post_range_times['pulse_artifact_samples1'] = pulse_artifact_samples1
    post_range_times['pulse_artifact_samples2'] = pulse_artifact_samples2

    start = time.time()
    epochs_now = epochs[0:n_trials_use]
    epochs_pre_calibration, epochs_post_calibration, ica, preprocessing_info = preprocess_prestim_and_poststim(epochs_now, pre_range_times, post_range_times,
                                                channel_rejection_options, ica_options, trial_rejection_options,
                                                ssp_sir_options, L, resample_to, filter_options_pre, bandstop_options_pre, filter_options_post, bandstop_options_post)
    
    end = time.time()
    
    processing_times = {'pre': {'good': [], 'bad': [], 'all': []}, 'post': {'good': [], 'bad': [], 'all': []},
                        'calibration_processing_time': end - start, 'trials_in_calibration': n_trials_use}
    

    preprocessing_info['n_trials_calibration'] = len(epochs_pre_calibration)
    preprocessing_info['n_trials_used_in_calibration'] = n_trials_use
    preprocessing_info['n_trials_original'] = len(epochs)
    
    if save_results:
        # Save epoch structures
        for epochs_structure, name_attr in zip([epochs_pre_calibration, epochs_post_calibration], ['pre', 'post']):
            filename = f"{subject_id}_calibration_{name_attr}.fif"
            filepath = os.path.join(subject_path_processed, filename)
            epochs_structure.save(filepath, overwrite=True)
        ica.save(os.path.join(subject_path_processed, f'{subject_id}_ica.fif'), overwrite=True)
    
    
    epochs_pre_list = []
    epochs_post_list = []
    bad_trials_calibrated = []
    preprocessing_info['n_trials_to_process_with_calibrated_filters'] = len(epochs) - n_trials_use
    
    for trial_ind in range(n_trials_use, len(epochs)):
        epoch = epochs[trial_ind]

        epoch_pre_now = epoch.copy().crop(pre_range_times['pre_range_final'][0], pre_range_times['pre_range_final'][1])
        start_pre = time.perf_counter()
        epoch_pre, processing_times_pre = preprocess_pre_trial_with_calibrated_data(epoch_pre_now, preprocessing_info, trial_rejection_options, ica, filter_options_pre, bandstop_options_pre, resample_to)
        end_pre = time.perf_counter()
        processing_times['pre']['all'].append(processing_times_pre)

        epoch_post_now = epoch.copy().crop(post_range_times['post_range'][0], post_range_times['post_range'][1])
        start_post = time.perf_counter()
        epoch_post, processing_times_post = preprocess_post_trial_with_calibrated_data(
            epoch_post_now, preprocessing_info, trial_rejection_options,
            ica, post_range_baseline, pulse_artifact_samples1, pulse_artifact_samples2, post_range_reject_trials,
            filter_options_post, bandstop_options_post, resample_to)
        end_post = time.perf_counter()
        processing_times['post']['all'].append(processing_times_post)

        if epoch_pre and epoch_post: #check if the trial was accepted in both conditions
            epochs_pre_list.append(epoch_pre)
            epochs_post_list.append(epoch_post)
            processing_times['pre']['good'].append(end_pre - start_pre)
            processing_times['post']['good'].append(end_post - start_post)
        else:
            bad_trials_calibrated.append(trial_ind)
            processing_times['pre']['bad'].append(end_pre - start_pre)
            processing_times['post']['bad'].append(end_post - start_post)
            
    preprocessing_info['bad_trials_calibrated'] = bad_trials_calibrated
    preprocessing_info['processing_times'] = processing_times

    if epochs_pre_list and epochs_post_list:
        epochs_pre = mne.concatenate_epochs(epochs_pre_list)
        epochs_post = mne.concatenate_epochs(epochs_post_list)

        # Metadata handling
        bad_keys = ['bad_trials_pre', 'bad_trials_post']
        block_idents = block_identifiers[:n_trials_use]
        for key in bad_keys:
            current_good_trials = np.array([index for index in range(len(block_idents)) if index not in preprocessing_info[key]])
            block_idents = block_idents[current_good_trials]
        good_trials_calibrated = np.array([index for index in range(n_trials_use, len(epochs)) if index not in preprocessing_info['bad_trials_calibrated']])
        block_idents2 = block_identifiers[good_trials_calibrated]
        block_identifiers = np.concatenate((block_idents, block_idents2))
        preprocessing_info['block_identifiers'] = block_identifiers
        preprocessing_info['n_trials_final'] = len(epochs_pre)

        # save results
        if save_results:
            # save epoch structs
            for epochs_structure, name_attr in zip([epochs_pre, epochs_post], ['pre', 'post']):
                filename = f"{subject_id}_fixedprep_{name_attr}.fif"
                filepath = os.path.join(subject_path_processed, filename)
                epochs_structure.save(filepath, overwrite=True)
            
            # save other data
            np.savez(os.path.join(subject_path_processed, f'{subject_id}_preprocessing_info.npz'), **preprocessing_info)
            np.save(os.path.join(subject_path_processed, f'{subject_id}_block_identifiers.npy'), block_identifiers)
            
            # save plots
            fig = epochs_post.average().plot_joint(show=False)
            fig.savefig(os.path.join(subject_path_processed, f'{subject_id}_post_joint.png'))
            plt.close(fig)
            del epochs_pre, epochs_post
    else:
        print(f"No fixed prep trials for {subject_id}")
        np.savez(os.path.join(subject_path_processed, f'{subject_id}_preprocessing_info.npz'), **preprocessing_info)
        
    del epochs
    del epochs_pre_list, epochs_post_list
    del epochs_pre_calibration, epochs_post_calibration
    del preprocessing_info, processing_times, ica
    print(f"Finished processing {subject_id}. Total time: {time.time() - start_time_total:.2f}s")

# %%
import multiprocessing as mp
def main():
    all_data_path = r"/scratch/nbe/c2b-shared/reftep_oskari"
    processed_path = r"/scratch/work/aholao1/reftep_processed"
    os.mkdir(processed_path)
    save_results = True
    args = []
    for site_id in ['Data_Tuebingen', 'Data_Aalto']:
        site_dir = os.path.join(all_data_path, site_id)
        subjects = os.listdir(site_dir)
        subjects = [subject for subject in subjects if "_rep" not in subject]
        args += [(site_id, subject_id, all_data_path, processed_path, save_results) for subject_id in subjects]
        
    with mp.Pool(processes=25) as pool:
        pool.starmap(run_subject_processing, args)

if __name__ == "__main__":
    main()