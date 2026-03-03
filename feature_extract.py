import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch, hilbert
from scipy.spatial import cKDTree
import pywt


# ---------- Helper Functions ----------

def zero_crossing(x):
    x = np.asarray(x)
    return np.sum((x[:-1] * x[1:]) < 0)


def willison_amplitude(x, threshold=0.01):
    return np.sum(np.abs(np.diff(x)) > threshold)


def turns_count(x, threshold=0.01):
    dx = np.diff(x)
    return np.sum((dx[:-1] * dx[1:] < 0) & (np.abs(dx[:-1]) > threshold) & (np.abs(dx[1:]) > threshold))


def mean_absolute_value_slope(x):
    dx = np.diff(x)
    if dx.size == 0:
        return 0.0
    return np.mean(np.abs(dx))


def slope_sign_changes(x):
    dx = np.diff(x)
    if dx.size < 2:
        return 0
    return np.sum((dx[:-1] * dx[1:]) < 0)


def sample_entropy(x, m=2, r=0.2):
    x = np.asarray(x, dtype=float).ravel()
    N = x.size
    if N <= m + 1:
        return np.nan

    std_x = np.std(x)
    if std_x == 0:
        return 0.0

    r *= std_x

    def _count_matches(dim):
        emb = np.lib.stride_tricks.sliding_window_view(x, dim)
        if emb.shape[0] < 2:
            return 0
        tree = cKDTree(emb)
        return len(tree.query_pairs(r=r, p=np.inf))

    B = _count_matches(m)
    A = _count_matches(m + 1)

    if B == 0:
        return np.inf
    if A == 0:
        return -np.log(np.finfo(float).tiny / B)

    return -np.log(A / B)


def higuchi_fd(x, kmax=5):
    N = len(x)
    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            x_mk = x[m::k]
            if x_mk.size < 2:
                continue
            Lmk = np.sum(np.abs(np.diff(x_mk)))
            denom = ((N - m) // k) * k
            if denom == 0:
                continue
            Lmk = (Lmk * (N - 1)) / denom
            Lk.append(Lmk)
        if not Lk:
            Lk = [0.0]
        L.append(np.mean(Lk))

    L = np.asarray(L, dtype=float)
    L[L <= 0] = np.finfo(float).tiny
    lnL = np.log(L)
    lnk = np.log(1.0 / np.arange(1, kmax + 1))
    return np.polyfit(lnk, lnL, 1)[0]


def lempel_ziv_complexity(x):
    x = (x > np.mean(x)).astype(int)
    s = ''.join(map(str, x))
    if len(s) == 0:
        return 0.0

    i, c, l = 0, 1, 1
    while True:
        if s[i:i + l] not in s[:i]:
            c += 1
            i += l
            l = 1
        else:
            l += 1
        if i + l > len(s):
            break
    return c / len(s)


def hjorth_parameters(x):
    dx = np.diff(x)
    ddx = np.diff(dx)

    var_x = np.var(x)
    var_dx = np.var(dx) if dx.size > 0 else 0.0
    var_ddx = np.var(ddx) if ddx.size > 0 else 0.0

    activity = var_x
    mobility = np.sqrt(var_dx / var_x) if var_x > 0 else 0.0
    complexity = (np.sqrt(var_ddx / var_dx) / mobility) if (var_dx > 0 and mobility > 0) else 0.0
    return activity, mobility, complexity


def get_feature_names():
    return [
        # Time-domain
        "Mean",
        "Variance",
        "Standard Deviation (SD)",
        "RMS",
        "Integrated EMG (IEMG)",
        "Skewness",
        "Kurtosis",
        "Zero Crossings (ZC)",
        "Mean Absolute Value Slope (MAVS)",
        "Turn Count (TC)",
        "Willison Amplitude (WAMP)",
        "Motor Unit Action Potential (MUAP)",
        "Signal Duration (SGD)",
        # Frequency-domain
        "Mean Frequency (MNF)",
        "Median Frequency (MDF)",
        "Total Power (TP)",
        "Peak Frequency (PF)",
        "Variance of Central Frequency (VCF)",
        # Time-Frequency
        "RMS Envelope (RMSE)",
        "Wavelet Entropy (WE)",
        "Max Wavelet Coefficient (MWC)",
        "Slope Sign Changes (SSC)",
        # Nonlinear
        "Sample Entropy (SE)",
        "Fractal Dimension (FD)",
        "Lempel-Ziv Complexity (LZC)",
        "Hjorth Activity (HA)",
        "Hjorth Mobility (HM)",
        "Hjorth Complexity (HC)",
    ]


# ---------- MAIN FEATURE FUNCTION ----------

def extract_features(x, fs=4096):
    x = np.asarray(x, dtype=float).ravel()
    features = []
    # Nonlinear metrics are O(N^2)-like; bound length to avoid memory spikes.
    if x.size > 4096:
        step = int(np.ceil(x.size / 4096.0))
        x_nl = x[::step]
    else:
        x_nl = x

    # ===== Time-domain =====
    mean_x = np.mean(x)
    std_x = np.std(x)
    features.extend([
        mean_x,
        np.var(x),
        std_x,
        np.sqrt(np.mean(x ** 2)),
        np.sum(np.abs(x)),
        skew(x),
        kurtosis(x),
        zero_crossing(x),
        mean_absolute_value_slope(x),
        turns_count(x),
        willison_amplitude(x),
        np.mean(np.abs(x)),
        len(x) / float(fs),
    ])

    # ===== Frequency-domain =====
    f, Pxx = welch(x, fs=fs)
    total_power = np.sum(Pxx)
    if total_power <= 0:
        pxx_norm = np.full_like(Pxx, 1.0 / len(Pxx))
    else:
        pxx_norm = Pxx / total_power

    mnf = np.sum(f * pxx_norm)
    mdf = f[np.where(np.cumsum(pxx_norm) >= 0.5)[0][0]]
    pf = f[np.argmax(Pxx)]
    vcf = np.sum(((f - mnf) ** 2) * pxx_norm)

    features.extend([mnf, mdf, total_power, pf, vcf])

    # ===== Time-frequency =====
    env = np.abs(hilbert(x))
    rmse = np.sqrt(np.mean(env ** 2))

    coeffs = pywt.wavedec(x, 'db6', level=4)
    energy = np.array([np.sum(c ** 2) for c in coeffs], dtype=float)
    energy_sum = np.sum(energy)
    if energy_sum <= 0:
        energy_ratio = np.full_like(energy, 1.0 / len(energy))
    else:
        energy_ratio = energy / energy_sum

    wavelet_entropy = -np.sum(energy_ratio * np.log2(energy_ratio + 1e-12))
    max_wavelet_coeff = np.max(np.abs(np.concatenate(coeffs)))
    ssc = slope_sign_changes(x)

    features.extend([rmse, wavelet_entropy, max_wavelet_coeff, ssc])

    # ===== Nonlinear =====
    features.append(sample_entropy(x_nl))
    features.append(higuchi_fd(x_nl))
    features.append(lempel_ziv_complexity(x_nl))

    ha, hm, hc = hjorth_parameters(x_nl)
    features.extend([ha, hm, hc])

    return features
