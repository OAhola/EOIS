import mne
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import zscore
import multiprocessing as mp

mne.set_log_level('ERROR')

def preprocess_subject(subject, subjects_directory_meg, preprocessing_options, task="passive"):
    #define filepaths and create a directory for the processed data
    subject_processing_directory = os.path.join(subjects_directory_meg, subject)

    #define the path of the recording
    raw_task_path = rf"/scratch/shareddata/camcan/cc700/meg/pipeline/release005/BIDSsep/derivatives_passive/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003/{subject}/mf2pt2_{subject}_ses-passive_task-{task}_megtransdef.fif"

    #define paths to save to
    processed_epochs_path = os.path.join(subject_processing_directory, f"{subject}_processed_{task}-epo.fif")
    ica_path = os.path.join(subject_processing_directory, f"{subject}_{task}-ica.fif")
    subject_preprocessing_info_path = os.path.join(subject_processing_directory, f"{subject}_{task}_preprocessing_info.npy")

    raw_task = mne.io.read_raw_fif(raw_task_path, preload=True)

    event_ids = {'audio/300Hz': 6, 'audio/600Hz': 7, 'audio/1200Hz': 8, 'vis/checker': 9}
    try:
        events = mne.find_events(raw_task)
    except:
        print(f"{subject}: Problem with events in, skipping...")
        return
    unique_ids = np.unique(events[:, 2])
    if list(unique_ids) != [6, 7, 8, 9]:
        print(f"{subject} events unique ids (listed) are not equal to [6, 7, 8, 9], skipping...")
        return

    #read the data

    subject_preprocessing_info = {'subject': subject} #initialize preprocessing info directory for the subject
    subject_preprocessing_info['events_original'] = events
    subject_preprocessing_info['event_ids_original'] = event_ids
    subject_preprocessing_info.update(preprocessing_options) #copy over the parameters from preprocessing options

    raw_task.filter(l_freq=preprocessing_options['fmin'],h_freq=preprocessing_options['fmax']) #filter the raw data
    #calibrate ICA filters
    ica = mne.preprocessing.ICA(n_components=preprocessing_options['ica_n_components'], method='fastica', random_state=42) #initialize the ICA instance

    ica.fit(raw_task) #fit ica
    exclude_inds, exclude_comp_names, corr_matrix, ica_contrast_channel_names = find_bad_ics(raw_task, ica, preprocessing_options['ica_contrast_channels'], preprocessing_options['ica_correlation_z_thresh'], subject) #find bad IC indices
    subject_preprocessing_info['ica_exclude_inds'] = exclude_inds
    subject_preprocessing_info['ica_exclude_comp_names'] = exclude_comp_names
    subject_preprocessing_info['ica_corr_matrix'] = corr_matrix
    subject_preprocessing_info['ica_corr_matrix_contrast_channel_names'] = ica_contrast_channel_names
    #apply ica
    ica.exclude = exclude_inds #tell the ica instance that these indices should be excluded
    ica.apply(raw_task)

    #epoch the data around the selected events, do not apply baseline correction here yet
    epochs = mne.Epochs(raw_task, events, event_id=event_ids, tmin=preprocessing_options['epochs_tmin'], tmax=preprocessing_options['epochs_tmax'], baseline=None, preload=True)
    epochs_meg = epochs.copy().pick(['meg']) # MEG data
    epochs_eog = epochs.copy().pick(preprocessing_options['eog_channels_reject']['ch_names']) # pick EOG channels

    #find bad trials from the epoched data structure
    bad_trials = find_bad_trials(epochs_meg, epochs_eog, preprocessing_options['z_thresh_trials'], subject, always_reject=[], eog_channels_reject=preprocessing_options['eog_channels_reject'])
    if bad_trials: #drop (reject) the bad trials, if any
        epochs_meg.drop(bad_trials)
        
    os.mkdir(subject_processing_directory)
    #save the the processed data, ica and cov instances, and the preprocessing info
    epochs_meg.save(processed_epochs_path)
    ica.save(ica_path)
    np.save(subject_preprocessing_info_path, subject_preprocessing_info)
    print(f"Processed {subject}")


def find_bad_ics(raw, ica, contrast_channels, z_thresh, subject):
    print(f"{subject}: Finding ICs to exclude...")
    #pick only the ecg and eog channels from the data
    contrast_channel_raw = raw.copy().pick(list(contrast_channels.values()))
    contrast_channel_data = contrast_channel_raw.get_data()
    contrast_channel_names = contrast_channel_raw.info['ch_names']

    ica_raw = ica.get_sources(raw) #time courses of the ICs (Raw object)
    ica_activations = ica_raw.get_data() #time courses of the Raw object representing the ICs

    #get the correlations between the contrast channel data and the IC time courses
    corr_matrix = np.zeros((contrast_channel_data.shape[0],ica_activations.shape[0]))
    for i in range(contrast_channel_data.shape[0]):
        for j in range(ica_activations.shape[0]):
            corr_matrix[i,j], _ = pearsonr(contrast_channel_data[i,:], ica_activations[j,:])

    #evaluate the correlations
    exclude_inds = []
    exclude_comp_names = []
    for chan_corrs, chan_name in zip(corr_matrix, contrast_channel_names):
        chan_corrs = np.abs(chan_corrs) # take abs of the correlations to find positively and possible negatively related components
        best_match = np.argmax(chan_corrs) #ic index of the best matching component (corresponding to the highest correlation)
        z_scores = zscore(chan_corrs) #zscore across all component correlations
        if z_scores[best_match] > z_thresh: #check if the best one exceeds the threshold
            all_matches = np.where(z_scores > z_thresh)[0] #all indices that exceed the threshold (only if the best one exceeds it)
            #update the exclusion lists
            exclude_inds += list(all_matches)
            exclude_comp_names.append((chan_name, all_matches))
    exclude_inds = list(np.unique(exclude_inds)) #take only unique values to be excluded and transfer back to list format for ica.exclude later
    print(f"{subject}: Found artifact IC indices matching with channels: {exclude_comp_names}")
    return exclude_inds, exclude_comp_names, corr_matrix, contrast_channel_names


def find_bad_trials(epochs, epochs_eog, z_thresh, subject, always_reject=[], eog_channels_reject=None):
    #check for trials with:

    #1. anomaly-like data (z-score-based rejection) separetely in magneto- and gradiometers
    for type_of_channel_index, type_of_channel in enumerate(['mag','grad']):
        meg_data = epochs.copy().pick([type_of_channel]).get_data(copy=True)
        trial_scores = np.std(meg_data, axis=(1,2)) #stds for each trial
        z_scores = zscore(trial_scores) #z-score stds
        bad_trials_now = np.where(z_scores > z_thresh)[0] #check which trials exceed the threshold
        if type_of_channel_index == 0:
            bad_trials = bad_trials_now
        else:
            bad_trials = np.union1d(bad_trials, bad_trials_now)
            print(f"{subject}: Found bad trials based on deviation: {bad_trials}")
            
    # 2. if specific trials, such as the first ones, should always be rejected
    if len(always_reject) > 0:
        bad_trials = np.union1d(bad_trials, always_reject)
    
    # 3. where the EOG channels are highly active at the time of the stimulus onset
    if eog_channels_reject is not None:
        bad_trials_eog = []
        eog_data = np.abs(epochs_eog.get_data(copy=True)) #epoched absolute eog data
        samples_of_interest = np.where((epochs_eog.times >= eog_channels_reject['tmin']) & (epochs_eog.times <= eog_channels_reject['tmax']))[0] #evaluate only the samples of interest near the stimulus onset
        z_scored_eogs = (eog_data - np.mean(eog_data, axis=(0,2), keepdims=True)) / np.std(eog_data, axis=(0,2), keepdims=True)
        for trial_index, trial_scores in enumerate(z_scored_eogs):
            if np.any(trial_scores[:,samples_of_interest] > eog_channels_reject['z_thresh']):
                bad_trials_eog.append(trial_index) #add the trial index
        bad_trials = np.union1d(bad_trials, np.array(bad_trials_eog))
        print(f"{subject}: Found bad eog trials: {bad_trials_eog}")

    return list(bad_trials) #return bad trials


def process_subject(subject):
    print(f'Processing {subject}')

    preprocessing_options = {
    'fmin':1,'fmax':30, 'epochs_tmin':-0.2, 'epochs_tmax':0.7,

    'ica_n_components':0.99, 'ica_contrast_channels':{'ecg_1':'ECG063','eog_1':'EOG061', 'eog_2':'EOG062'}, 
        
    'ica_correlation_z_thresh':3, 'z_thresh_trials':2.5,
        
    'eog_channels_reject':{'tmin':-0.1, 'tmax':0.1, 'ch_names':['EOG061','EOG062'], 'z_thresh':3}}
    
    preprocessed_dir = "/scratch/work/aholao1/camcan_processed"

    preprocess_subject(subject, preprocessed_dir, preprocessing_options, task="passive")

# run the processing
if __name__ == "__main__":
    subjects_dir = "/scratch/shareddata/camcan/cc700/meg/pipeline/release005/BIDSsep/derivatives_passive/aa/AA_movecomp_transdef/aamod_meg_maxfilt_00003"
    preprocessed_dir = "/scratch/work/aholao1/camcan_processed"
    os.mkdir(preprocessed_dir)
    subjects = [folder_name for folder_name in os.listdir(subjects_dir) if "sub-" in folder_name]
    with mp.Pool(processes=50) as pool:
        pool.map(process_subject, subjects)