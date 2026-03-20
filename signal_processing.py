import numpy as np
from scipy.signal.windows import hann, flattop


# =====================================================================
# [新增核心引擎] 统一的窗函数生成与能量补偿器
# =====================================================================
def get_window_and_correction(N, window_type):
    """
    生成指定长度的窗函数及其能量补偿系数。
    解决加窗后信号整体能量下降的问题。
    """
    if window_type == 'hann':
        w = hann(N)
    elif window_type == 'flattop':
        w = flattop(N)
    else:
        w = np.ones(N)  # 矩形窗 (等同于不加窗)

    # 能量补偿: 比如 Hann 窗均值是 0.5，算出来的振幅要乘以 2 才能还原真实物理值
    correction = 1.0 / np.mean(w) if np.mean(w) > 0 else 1.0
    return w, correction


# =====================================================================
# 1. 不叠加傅里叶变换 (长序列直接 FFT - 选项 1 导出专用)
# =====================================================================
def fft_no_stack(timeseries, sample_rate, window_type='hann'):
    """对整段输入的时间序列直接加窗并进行 FFT 变换"""
    n_sam = timeseries.shape[-1]
    w, corr = get_window_and_correction(n_sam, window_type)

    # 利用 NumPy 的广播机制，自动对 1D 或 2D(多通道) 数组进行加窗
    windowed_ts = timeseries * w

    yf_raw = np.fft.fft(windowed_ts, axis=-1)
    xf = np.fft.fftfreq(n_sam, 1.0 / sample_rate)

    # 严格的物理振幅缩放 + 窗函数能量补偿
    yf = yf_raw * (2.0 / n_sam) * corr
    return xf, yf


# =====================================================================
# 2. 时域叠加 (Time-Domain Stacking - 选项 2 专用)
# =====================================================================
def time_domain_stacking(time_series, cyc_len, cyc_num):
    """时域叠加算法 (同步平均)"""
    expected_len = cyc_len * cyc_num
    valid_data = time_series[:expected_len]
    matrix = valid_data.reshape((cyc_num, cyc_len))
    stacked_wave = np.mean(matrix, axis=0)
    return stacked_wave


def fft_short(stacked_wave, sample_rate, pad_factor=1, window_type='hann'):
    """
    对叠加后的单周期波形先加窗，再补零，最后进行高密度 FFT。
    加窗可以完美消除首尾不连续造成的突变泄露。
    """
    n_original = len(stacked_wave)
    w, corr = get_window_and_correction(n_original, window_type)

    # 1. 先加窗，压平单周期首尾
    windowed_wave = stacked_wave * w

    # 2. 加窗后再补零，实现完美平滑过渡
    n_padded = n_original * pad_factor
    freqs = np.fft.fftfreq(n_padded, d=1.0 / sample_rate)
    yf = np.fft.fft(windowed_wave, n=n_padded)

    # ==========================================
    # 【核心修复】：严格统一物理振幅换算公式！
    # 无论后面补了多少零，真实的信号能量只存在于前 n_original 个点中。
    # 所以必须乘以 (2.0 / n_original) 来还原真实的物理振幅！
    # ==========================================
    real_yf = yf * (2.0 / n_original) * corr

    return freqs, real_yf


# =====================================================================
# 3. 频域叠加 (Frequency-Domain Stacking - 选项 3专用 )
# =====================================================================
def fft_freq_stacking(timeseries, sample_rate, cyc_len, cyc_num, window_type='hann'):
    """先按周期分别加窗并FFT，再在频域内求平均"""
    n_chan = timeseries.shape[0] if len(timeseries.shape) > 1 else 1
    expected_len = cyc_len * cyc_num

    if n_chan == 1:
        valid_ts = timeseries[:expected_len]
        reshaped_ts = valid_ts.reshape((cyc_num, cyc_len))
    else:
        valid_ts = timeseries[:, :expected_len]
        reshaped_ts = valid_ts.reshape((n_chan, cyc_num, cyc_len))

    # 为单个循环周期生成窗函数
    w, corr = get_window_and_correction(cyc_len, window_type)

    # 广播相乘：给每一个独立循环都单独加窗
    windowed_ts = reshaped_ts * w

    # 对加过窗的每个循环独立做 FFT
    yf_all_cycles = np.fft.fft(windowed_ts, axis=-1)

    # 频域叠加均值化
    yf_stacked_raw = np.mean(yf_all_cycles, axis=-2 if n_chan == 1 else 1)

    xf = np.fft.fftfreq(cyc_len, 1.0 / sample_rate)
    yf_stacked = yf_stacked_raw * (2.0 / cyc_len) * corr
    return xf, yf_stacked


# =====================================================================
# 4. 加窗全长 FFT (测底噪专用 - 选项 6 专用)
# =====================================================================
def long_fft_with_window(time_series, sample_rate, window_type='hann'):
    """复用底层的加窗 FFT 引擎，仅返回真实振幅供直接画图使用"""
    freqs, yf_complex = fft_no_stack(time_series, sample_rate, window_type)
    return freqs, np.abs(yf_complex)


# =====================================================================
# 5. 精准提取单点复数
# =====================================================================
def extract_target_frequency(xf, yf_matrix, target_freq):
    """精准提取主频复数结果"""
    idx = np.argmin(np.abs(xf - target_freq))
    if len(yf_matrix.shape) == 1:
        return yf_matrix[idx]
    return yf_matrix[:, idx]