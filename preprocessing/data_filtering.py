from scipy.signal import butter, sosfiltfilt

def butter_filter(data, cutoff, btype, fs, order, pad_time):
    nyquist = fs/2 #nyquist rate
    if isinstance(cutoff, list):
        if len(cutoff)==2:
            cutoff_normalized = [cutoff[0]/nyquist, cutoff[1]/nyquist] #for bandpass and bandstop filters (two frequency edges)
        else:
            raise ValueError("cutoff should be a single value or a list of length 2.")
    else:
        cutoff_normalized = cutoff/nyquist #low-pass or high-pass style filter has only one edge frequency
    sos = butter(order, cutoff_normalized, btype=btype, analog=False, output='sos') #get the filter coefficients
    filtered_data = apply_filter_to_data(data, sos, pad_time, fs)
    return filtered_data, sos #return the filtered data without pad samples

def apply_filter_to_data(data, coefs, pad_time, fs):
    pad_samples = int(pad_time*fs) #number of samples to pad
    filtered_data = sosfiltfilt(coefs, data, padlen=pad_samples, padtype='odd') #apply filter to the data
    return filtered_data