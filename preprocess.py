
import numpy as np
from scipy.signal import butter, filtfilt, resample
import pywt

def preprocess(signal, fs=32768):
    # Downsample only when needed, matching the paper's 4,096 Hz target.
    target_fs = 4096
    if fs != target_fs:
        signal = resample(signal, int(len(signal) * target_fs / fs))
    b,a = butter(3, 10/(4096/2), btype='high')
    signal = filtfilt(b,a,signal)
    coeffs = pywt.wavedec(signal, 'db6', level=4)
    # Universal threshold based on MAD from the finest-detail coefficients.
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(len(signal) + 1e-12))
    # Apply soft-thresholding only to detail coefficients; keep approximation.
    coeffs = [coeffs[0]] + [pywt.threshold(c, thresh, mode='soft') for c in coeffs[1:]]
    cleaned = pywt.waverec(coeffs, 'db6')
    return cleaned[:len(signal)]
