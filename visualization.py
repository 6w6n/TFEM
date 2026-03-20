import numpy as np
from pathlib import Path

# =====================================================================
# 1. Matplotlib 全局环境与字体配置
# =====================================================================
import matplotlib

matplotlib.use('TkAgg')  # 强制指定后端，确保交互窗口和滑块正常弹出

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.ticker import FuncFormatter, LogLocator

# 强制全局支持中文和负号显示，防止由于 \u2212 导致的方块报错
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# [静态图库] 基础波形与频谱可视化
# =====================================================================
def plot_waveform(data, title="Time Domain Waveform", xlabel="Sample Points", ylabel="Amplitude",
                  num_points=None, vlines=None, ylim=None):
    """画出时域的时间序列波形图 (支持强制锁定纵轴范围 ylim)"""
    plot_data = data[:num_points] if (num_points is not None and num_points < len(data)) else data
    x_axis = np.arange(len(plot_data))

    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, plot_data, color='#1f77b4', linestyle='-', linewidth=1.5, marker='.', markersize=3)

    if vlines is not None:
        num_cycles = len(plot_data) // vlines
        for i in range(1, num_cycles + 1):
            plt.axvline(x=i * vlines, color='red', linestyle='--', alpha=0.6)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if ylim is not None:
        plt.ylim(ylim)

    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_compare_spectra(freqs, yf_complex1, yf_complex2, label1="Method 1", label2="Method 2",
                         title="Spectra Comparison", max_freq=None):
    """将两种不同算法（如时域叠加 vs 频域叠加）算出的频谱画在同一张图里对比"""
    positive_idx = freqs > 0
    f = freqs[positive_idx]
    amp1 = np.abs(yf_complex1[positive_idx])
    amp2 = np.abs(yf_complex2[positive_idx])

    if max_freq is not None:
        valid_idx = f <= max_freq
        f, amp1, amp2 = f[valid_idx], amp1[valid_idx], amp2[valid_idx]

    plt.figure(figsize=(12, 5))
    plt.plot(f, amp1, label=label1, color='blue', linewidth=2, alpha=0.7)
    plt.plot(f, amp2, label=label2, color='red', linewidth=2, linestyle='--', alpha=0.7)

    plt.title(title, fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# =====================================================================
# [交互图库] 动态数据查看器 (带滑块与通道切换)
# =====================================================================
def interactive_time_viewer(ts_dir, file_stem):
    """交互式时域波形查看器 (自适应周期 + 鼠标无缝切换通道)"""
    ts_dir_path = Path(ts_dir)
    period_files = list(ts_dir_path.glob(f"{file_stem}_#Period=*_Timeseries.txt"))
    n_periods = len(period_files)

    if n_periods == 0:
        print(f"\n❌ 严重错误: 在 {ts_dir_path} 中没有找到任何周期数据！")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2, left=0.2)

    def load_period_data(p):
        path = ts_dir_path / f"{file_stem}_#Period={p:02d}_Timeseries.txt"
        return np.loadtxt(path, skiprows=2) if path.exists() else None

    initial_data = load_period_data(1)
    if initial_data is None: return

    chan_labels = [f"Ch {i}" for i in range(initial_data.shape[1])]
    state = {'period': 1, 'channel': 0}

    y_data_init = initial_data[:, state['channel']]
    x_data_init = np.arange(len(y_data_init))
    line, = ax.plot(x_data_init, y_data_init, lw=1, color='b')

    ax.set_title(f"Time Series: Period 1 - Channel 0")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, len(x_data_init))

    ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, 'Period', 1, n_periods, valinit=1, valstep=1)

    ax_radio = plt.axes([0.02, 0.4, 0.12, 0.2], facecolor='lightgoldenrodyellow')
    radio = RadioButtons(ax_radio, chan_labels, active=0)

    def update_plot():
        data = load_period_data(state['period'])
        if data is None: return
        new_y = data[:, state['channel']]
        new_x = np.arange(len(new_y))

        line.set_xdata(new_x)
        line.set_ydata(new_y)
        ax.set_xlim(0, len(new_x))

        y_min, y_max = new_y.min(), new_y.max()
        margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_title(f"Time Series: Period {state['period']} - Channel {state['channel']} (Samples: {len(new_x)})")
        fig.canvas.draw_idle()

    slider.on_changed(lambda val: (state.update({'period': int(val)}), update_plot()))
    radio.on_clicked(lambda label: (state.update({'channel': int(label.split(" ")[1])}), update_plot()))
    plt.show(block=True)


def interactive_freq_viewer(fs_dir, file_stem):
    """交互式频域频谱查看器 (自适应周期数量，仅显示正频率幅度谱)"""
    fs_dir_path = Path(fs_dir)
    period_files = list(fs_dir_path.glob(f"{file_stem}_#Period=*_Spectrum.txt"))
    n_periods = len(period_files)

    if n_periods == 0: return

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2)

    def load_freq_data(p):
        path = fs_dir_path / f"{file_stem}_#Period={p:02d}_Spectrum.txt"
        if not path.exists(): return None
        data = np.loadtxt(path, skiprows=1)
        raw_freq = data[:, 0]
        pos_mask = raw_freq >= 0
        amp_pos = np.sqrt(data[:, 1] ** 2 + data[:, 2] ** 2)[pos_mask]
        return raw_freq[pos_mask], amp_pos

    init_data = load_freq_data(1)
    if init_data is None: return
    f_init, a_init = init_data

    line, = ax.semilogy(f_init, a_init, color='r', lw=1.5)
    ax.set_title(f"Frequency Spectrum - Magnitude (Period 1)")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.set_xlabel("Frequency (Hz)"), ax.set_ylabel("Amplitude")
    ax.set_xlim(f_init.min(), f_init.max())

    valid_a = a_init[a_init > 0]
    if len(valid_a) > 0: ax.set_ylim(valid_a.min() * 0.5, valid_a.max() * 2.0)

    slider = Slider(plt.axes([0.2, 0.05, 0.6, 0.03]), 'Period', 1, n_periods, valinit=1, valstep=1)

    def update(val):
        data = load_freq_data(int(val))
        if data is None: return
        line.set_xdata(data[0])
        line.set_ydata(data[1])
        ax.set_xlim(data[0].min(), data[0].max())
        valid = data[1][data[1] > 0]
        if len(valid) > 0: ax.set_ylim(valid.min() * 0.5, valid.max() * 2.0)
        ax.set_title(f"Frequency Spectrum - Magnitude (Period {int(val)})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show(block=True)


# =====================================================================
# [工业级图库] 精密谐波分析 (纯 FFT 局部寻峰版)
# =====================================================================
def plot_analyzed_spectrum(freqs_fft, yf_fft, fundamental_freq, num_harmonics=15, title="频谱精密分析 (纯 FFT)"):
    """使用高密度 FFT 并在理论频点附近进行局部寻峰，放弃 DFT。"""
    fig, ax = plt.subplots(figsize=(16, 8))

    # 1. FFT 全频段背景黑线
    pos_idx = freqs_fft > 0
    f_full, amp_full = freqs_fft[pos_idx], np.abs(yf_fft[pos_idx])
    ax.loglog(f_full, amp_full, color='black', linewidth=0.5, alpha=0.8, label="FFT 轮廓")

    # 2. 纯 FFT 局部寻峰算法标点 (绿点)
    theoretical_harmonics = [fundamental_freq * (2 * i + 1) for i in range(num_harmonics // 2 + 1)]
    for i, th_f in enumerate(theoretical_harmonics):
        if th_f > f_full.max() or th_f < f_full.min(): continue

        # 【核心】：在理论频率附近的正负 1% 范围内寻找真实的局部最大值
        idx_center = np.argmin(np.abs(f_full - th_f))
        search_radius = max(5, int(len(f_full) * 0.01))
        idx_min = max(0, idx_center - search_radius)
        idx_max = min(len(f_full), idx_center + search_radius)

        # 锁定局部最高峰的索引、频率和振幅
        local_max_idx = idx_min + np.argmax(amp_full[idx_min:idx_max])
        peak_f = f_full[local_max_idx]
        peak_amp = amp_full[local_max_idx]

        # 悬浮画法
        dot_y = peak_amp * (2.5 if i % 2 == 0 else 5.0)
        ax.plot([peak_f, peak_f], [peak_amp, dot_y], color='black', linewidth=0.8, zorder=4)
        ax.scatter(peak_f, dot_y, color='#00FF00', s=50, edgecolors='black', zorder=5)

        # 标出找到的真实频率和真实振幅
        ax.text(peak_f, dot_y * 1.2, f"{peak_f:.1f}Hz({2 * i + 1}T)\n{peak_amp:.4f}",
                rotation=30, ha='left', va='bottom', fontsize=9, color='green')

    # ===============================================================
    # 3. 工业级坐标轴设置 (双对数 + 横轴反转 + 密集网格)
    # ===============================================================
    ax.set_xlim(1000, 0.01)

    # 【终极坐标修复】：强制按最简格式显示浮点数，比如把 0.01 就显示为 0.01，绝不会变成 0
    from matplotlib.ticker import FuncFormatter, LogLocator
    formatter = FuncFormatter(lambda y, _: f'{y:g}')

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # 设置极度密集的网格定位器
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0))

    # 开启四面边框的刻度，向内指
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labeltop=True)

    ax.grid(True, which='major', color='#a0a0a0', linestyle='-', linewidth=0.6, alpha=0.8)
    ax.grid(True, which='minor', color='#d3d3d3', linestyle=':', linewidth=0.5, alpha=0.6)

    ax.set_title(title, pad=40, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =====================================================================
# [新增图库] 纯净版整体频谱图 (双对数坐标 + 自然数显示)
# =====================================================================
def plot_overall_spectrum(freqs_fft, yf_fft, title="整体频率图"):
    """
    只画出 FFT 之后的整体频谱轮廓，不进行局部寻峰和标记。
    保留工业级双对数坐标、反转 X 轴以及自然数刻度显示。
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    # 1. 提取正频率部分
    pos_idx = freqs_fft > 0
    f_full, amp_full = freqs_fft[pos_idx], np.abs(yf_fft[pos_idx])

    # 2. 画出双对数曲线 (换个清爽的蓝色)
    ax.loglog(f_full, amp_full, color='#1f77b4', linewidth=1.2, alpha=0.9, label="FFT Spectrum")

    # 3. 工业级坐标轴设置 (与之前保持一致的高级质感)
    ax.set_xlim(1000, 0.01)

    from matplotlib.ticker import FuncFormatter, LogLocator
    # 强制按最简格式显示浮点数 (0.01, 0.1, 1, 10...)
    formatter = FuncFormatter(lambda y, _: f'{y:g}')

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # 设置极其密集的网格线
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0))

    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labeltop=True)
    ax.grid(True, which='major', color='#a0a0a0', linestyle='-', linewidth=0.6, alpha=0.8)
    ax.grid(True, which='minor', color='#d3d3d3', linestyle=':', linewidth=0.5, alpha=0.6)

    ax.set_title(title, pad=40, fontsize=14, fontweight='bold')
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)

    plt.tight_layout()
    plt.show()