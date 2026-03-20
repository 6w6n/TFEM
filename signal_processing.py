import numpy as np
from scipy.signal.windows import hann, flattop


# =====================================================================
# [新增核心引擎] 统一的窗函数生成与能量补偿器
# =====================================================================
def get_window_and_correction(N, window_type):
    if window_type == 'hann':
        w = hann(N)
    elif window_type == 'flattop':
        w = flattop(N)
    else:
        w = np.ones(N)  # 矩形窗

    correction = 1.0 / np.mean(w) if np.mean(w) > 0 else 1.0
    return w, correction


# =====================================================================
# 1. 不叠加傅里叶变换 (单周期直接 FFT)
# =====================================================================
def fft_no_stack(timeseries, sample_rate, window_type='hann'):
    n_sam = timeseries.shape[-1]
    w, corr = get_window_and_correction(n_sam, window_type)

    windowed_ts = timeseries * w
    yf_raw = np.fft.fft(windowed_ts, axis=-1)
    xf = np.fft.fftfreq(n_sam, 1.0 / sample_rate)

    yf = yf_raw * (2.0 / n_sam) * corr
    return xf, yf


# =====================================================================
# 2. 时域叠加 (Time-Domain Stacking)
# =====================================================================
def time_domain_stacking(time_series, cyc_len, cyc_num):
    expected_len = cyc_len * cyc_num
    valid_data = time_series[:expected_len]
    matrix = valid_data.reshape((cyc_num, cyc_len))
    stacked_wave = np.mean(matrix, axis=0)
    return stacked_wave


def fft_short(stacked_wave, sample_rate, pad_factor=1, window_type='hann'):
    n_original = len(stacked_wave)
    w, corr = get_window_and_correction(n_original, window_type)

    windowed_wave = stacked_wave * w
    n_padded = n_original * pad_factor
    freqs = np.fft.fftfreq(n_padded, d=1.0 / sample_rate)
    yf = np.fft.fft(windowed_wave, n=n_padded)

    # 严格统一物理振幅，基准为真实数据点数 n_original
    real_yf = yf * (2.0 / n_original) * corr
    return freqs, real_yf


# =====================================================================
# 3. 频域叠加 (Frequency-Domain Stacking)
# =====================================================================
def fft_freq_stacking(timeseries, sample_rate, cyc_len, cyc_num, window_type='hann'):
    n_chan = timeseries.shape[0] if len(timeseries.shape) > 1 else 1
    expected_len = cyc_len * cyc_num

    if n_chan == 1:
        reshaped_ts = timeseries[:expected_len].reshape((cyc_num, cyc_len))
    else:
        reshaped_ts = timeseries[:, :expected_len].reshape((n_chan, cyc_num, cyc_len))

    w, corr = get_window_and_correction(cyc_len, window_type)
    windowed_ts = reshaped_ts * w
    yf_all_cycles = np.fft.fft(windowed_ts, axis=-1)
    yf_stacked_raw = np.mean(yf_all_cycles, axis=-2 if n_chan == 1 else 1)

    xf = np.fft.fftfreq(cyc_len, 1.0 / sample_rate)
    yf_stacked = yf_stacked_raw * (2.0 / cyc_len) * corr
    return xf, yf_stacked


# =====================================================================
# 4. 加窗全长 FFT
# =====================================================================
def long_fft_with_window(time_series, sample_rate, window_type='hann'):
    freqs, yf_complex = fft_no_stack(time_series, sample_rate, window_type)
    return freqs, np.abs(yf_complex)