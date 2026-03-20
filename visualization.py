import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use('TkAgg')  # 强制指定后端，确保交互窗口正常弹出
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.ticker import FuncFormatter, LogLocator

# 全局强制支持中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# [静态图库] 1. 时域波形可视化
# =====================================================================
def plot_waveform(data, title="Time Domain Waveform", xlabel="Sample Points", ylabel="Amplitude",
                  num_points=None, vlines=None, ylim=None, max_vlines=20):
    """画出时域时间序列波形图，自带智能周期分割线防重叠保护"""
    plot_data = data[:num_points] if (num_points is not None and num_points < len(data)) else data
    x_axis = np.arange(len(plot_data))

    plt.figure(figsize=(12, 5))
    plt.plot(x_axis, plot_data, color='#1f77b4', linestyle='-', linewidth=1.5, marker='.', markersize=3)

    # ==========================================
    # 【核心修复】：智能红线控制机制
    # 如果周期数量超过 max_vlines (默认 20)，则自动隐藏红线，防止变成红色的墙
    # ==========================================
    if vlines is not None:
        num_cycles = len(plot_data) // vlines
        if num_cycles > max_vlines:
            print(f"  [视觉优化] 提示: '{title}' 包含 {num_cycles} 个周期，已自动隐藏红色分割线以保持波形清晰可见。")
        else:
            for i in range(1, num_cycles + 1):
                plt.axvline(x=i * vlines, color='red', linestyle='--', alpha=0.6)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if ylim is not None: plt.ylim(ylim)
    plt.grid(True, which='both', linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


# =====================================================================
# [交互图库] 2. 动态数据查看器
# =====================================================================
def interactive_time_viewer(ts_dir, file_stem):
    ts_dir_path = Path(ts_dir)
    period_files = list(ts_dir_path.glob(f"{file_stem}_#Period=*_Timeseries.txt"))
    if not period_files: return print(f"\n❌ 未找到周期数据！")

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2, left=0.2)

    def load_period_data(p):
        path = ts_dir_path / f"{file_stem}_#Period={p:02d}_Timeseries.txt"
        return np.loadtxt(path, skiprows=2) if path.exists() else None

    initial_data = load_period_data(1)
    if initial_data is None: return

    chan_labels = [f"Ch {i}" for i in range(initial_data.shape[1])]
    state = {'period': 1, 'channel': 0}

    line, = ax.plot(np.arange(len(initial_data)), initial_data[:, 0], lw=1, color='b')
    ax.set_title("Time Series: Period 1 - Channel 0"), ax.grid(True, linestyle='--', alpha=0.6)

    slider = Slider(plt.axes([0.25, 0.05, 0.65, 0.03]), 'Period', 1, len(period_files), valinit=1, valstep=1)
    radio = RadioButtons(plt.axes([0.02, 0.4, 0.12, 0.2], facecolor='lightgoldenrodyellow'), chan_labels, active=0)

    def update_plot():
        data = load_period_data(state['period'])
        if data is None: return
        new_y = data[:, state['channel']]
        line.set_xdata(np.arange(len(new_y))), line.set_ydata(new_y)
        ax.set_xlim(0, len(new_y))
        y_min, y_max = new_y.min(), new_y.max()
        ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
        ax.set_title(f"Period {state['period']} - Channel {state['channel']}")
        fig.canvas.draw_idle()

    slider.on_changed(lambda val: (state.update({'period': int(val)}), update_plot()))
    radio.on_clicked(lambda label: (state.update({'channel': int(label.split(" ")[1])}), update_plot()))
    plt.show(block=True)


def interactive_freq_viewer(fs_dir, file_stem):
    """【功能5】: 交互式频域查看 (双对数固定坐标，自然数显示)"""
    fs_dir_path = Path(fs_dir)
    period_files = list(fs_dir_path.glob(f"{file_stem}_#Period=*_Spectrum.txt"))
    n_periods = len(period_files)

    if n_periods == 0: return print(f"\n❌ 未找到频谱数据！")

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

    line, = ax.loglog(f_init, a_init, color='r', lw=1.5)
    ax.set_title(f"Frequency Spectrum - Magnitude (Period 1)")
    ax.grid(True, which="both", ls="--", alpha=0.6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(1000, 0.01)

    valid_a = a_init[a_init > 0]
    if len(valid_a) > 0: ax.set_ylim(valid_a.min() * 0.5, valid_a.max() * 2.0)

    formatter = FuncFormatter(lambda y, _: f'{y:g}')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0))
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labeltop=True)

    slider = Slider(plt.axes([0.2, 0.05, 0.6, 0.03]), 'Period', 1, n_periods, valinit=1, valstep=1)

    def update(val):
        data = load_freq_data(int(val))
        if data is None: return
        line.set_xdata(data[0])
        line.set_ydata(data[1])
        valid = data[1][data[1] > 0]
        if len(valid) > 0: ax.set_ylim(valid.min() * 0.5, valid.max() * 2.0)
        ax.set_title(f"Frequency Spectrum - Magnitude (Period {int(val)})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show(block=True)


# =====================================================================
# [工业级图库] 3. 总体轮廓与精密分析图 (纯 FFT 版)
# =====================================================================
def plot_overall_spectrum(freqs_fft, yf_fft, title="整体频率图"):
    """纯净版整体频谱轮廓图 (无悬浮标记)"""
    fig, ax = plt.subplots(figsize=(16, 8))

    pos_idx = freqs_fft > 0
    f_full, amp_full = freqs_fft[pos_idx], np.abs(yf_fft[pos_idx])

    ax.loglog(f_full, amp_full, color='#1f77b4', linewidth=1.2, alpha=0.9, label="FFT Spectrum")
    ax.set_xlim(1000, 0.01)

    formatter = FuncFormatter(lambda y, _: f'{y:g}')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
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


def plot_analyzed_spectrum(freqs_fft, yf_fft, fundamental_freq, num_harmonics=15, title="频谱精密分析 (纯 FFT)"):
    """使用高密度 FFT 并在严格限定的频率范围内进行局部寻峰"""
    fig, ax = plt.subplots(figsize=(16, 8))

    pos_idx = freqs_fft > 0
    f_full, amp_full = freqs_fft[pos_idx], np.abs(yf_fft[pos_idx])
    ax.loglog(f_full, amp_full, color='black', linewidth=0.5, alpha=0.8, label="FFT 轮廓")

    # 2. 纯 FFT 局部寻峰算法标点 (绿点)
    theoretical_harmonics = [fundamental_freq * (2 * i + 1) for i in range(num_harmonics // 2 + 1)]

    # 获取真实的频率分辨率 (df)
    df = f_full[1] - f_full[0] if len(f_full) > 1 else 1.0

    for i, th_f in enumerate(theoretical_harmonics):
        if th_f > f_full.max() or th_f < f_full.min(): continue

        idx_center = np.argmin(np.abs(f_full - th_f))

        # ==========================================
        # 【终极自适应寻峰】：根据物理频率分辨率，智能切换三大状态！
        # ==========================================
        if df > 5.0:
            # 状态 1：单周期数据 (功能3，df=15.6Hz)。极其稀疏，死死锁定最近的理论频点，半径为 0。
            search_radius = 0
        else:
            # 状态 2 & 3：长序列或补零数据。
            # 我们放宽到 ±1.5 Hz 的物理搜索范围！
            # 对于功能 2 (df=0.78)，1.5/0.78 = 1，自动退化为最完美的 1 个点半径。
            # 对于功能 6 (df=0.078)，1.5/0.078 = 19，允许寻找偏移的峰，但完美避开 50Hz (距3.1Hz)。
            search_radius = int(1.5 / df)

        idx_min = max(0, idx_center - search_radius)

        # 注意这里的 + 1，即使 radius 是 0，也能保证刚好取到 idx_center 自身
        idx_max = min(len(f_full), idx_center + search_radius + 1)

        # 锁定局部最高峰
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

    ax.set_xlim(1000, 0.01)

    formatter = FuncFormatter(lambda y, _: f'{y:g}')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0))
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labeltop=True)
    ax.grid(True, which='major', color='#a0a0a0', linestyle='-', linewidth=0.6, alpha=0.8)
    ax.grid(True, which='minor', color='#d3d3d3', linestyle=':', linewidth=0.5, alpha=0.6)

    ax.set_title(title, pad=40, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()