
import numpy as np
from scipy.signal import butter, filtfilt, resample
import pywt

def preprocess(signal, fs=32768):
    signal = resample(signal, int(len(signal)*4096/fs))
    b,a = butter(3, 10/(4096/2), btype='high')
    signal = filtfilt(b,a,signal)
    coeffs = pywt.wavedec(signal, 'db6', level=4)
    # Universal threshold based on MAD from the finest-detail coefficients.
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(len(signal) + 1e-12))
    coeffs = [pywt.threshold(c, thresh, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, 'db6')
