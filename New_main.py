import sys
import numpy as np
from pathlib import Path

# 导入我们的核心功能模块
import data_io
import signal_processing
import visualization as vis


def main():
    # ==========================================
    # 1. 全局配置区
    # ==========================================
    tx_filepath = r"D:\资料包\时频电磁\测试数据\current\07-14\C016ST01.DAT"
    rx_filepath = r"D:\资料包\时频电磁\测试数据\data\0714\C016ST521.dat"

    print("\n🚀 欢迎使用 TFEM 时频电磁数据处理平台 🚀")
    print(f"[*] 默认发射机文件: {tx_filepath}")
    print(f"[*] 默认接收机文件: {rx_filepath}")

    # ==========================================
    # 2. 交互式控制主循环
    # ==========================================
    while True:
        print("\n" + "=" * 38)
        print("           📊 主 要 功 能 菜 单           ")
        print("=" * 38)
        print("  1. 时频数据导出 (长序列直接FFT)")
        print("  2. 叠加傅里叶变换-时域 (全部时域->叠加时域->FFT频谱)")
        print("  3. 叠加傅里叶变换-频域 (全部时域->全长FFT总体->频域叠加分析)")
        print("  4. [交互] 查看时域波形")
        print("  5. [交互] 查看全频段频谱 (自然数坐标)")
        print("  6. [测试] 不叠加，直接加窗长序列 FFT")
        print("  0. 退出系统")
        print("=" * 38)

        choice = input("👉 请输入操作对应的数字 [0-6]: ").strip()

        # ---------------------------------------------------------
        if choice == '1':
            print("\n>>> 正在执行: [1] 时频数据导出并进行单周期FFT...")
            data_io.process_and_export_all_data(tx_filepath, rx_filepath)

        # ---------------------------------------------------------
        elif choice == '2':
            print("\n>>> 正在执行: [2] 叠加傅里叶变换 - 时域...")
            stem = Path(rx_filepath).stem
            ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"

            try:
                p_idx = int(input("👉 请输入要测试的周期号 (例如 1): ").strip())
                ch_idx = int(input("👉 请输入要处理的通道号 (例如 0 或 1): ").strip())
            except ValueError:
                print("❌ 输入无效，请输入整数！")
                continue

            target_file = ts_dir / f"{stem}_#Period={p_idx:02d}_Timeseries.txt"
            if not target_file.exists():
                print(f"❌ 找不到文件: {target_file}")
                continue

            print(f"  正在加载数据: {target_file.name}")
            data = np.loadtxt(target_file, skiprows=2)
            time_series = data[:, ch_idx]

            rx_header, _, _, rx_sr, _ = data_io.read_age_binary(rx_filepath)
            cyc_len = int(rx_header['Isw'][30 + (p_idx - 1)])
            cyc_num = int(rx_header['Isw'][60 + (p_idx - 1)])

            # 【提前交互】获取基频
            theory_f0 = rx_sr / cyc_len
            user_f0 = input(f"👉 请输入发射基频 (直接回车使用 {theory_f0:.4f} Hz): ").strip()
            f0 = theory_f0 if not user_f0 else float(user_f0)

            print(f"  [>] 执行参数: 循环长度={cyc_len}, 循环次数={cyc_num}")

            # 算法计算
            stacked_wave = signal_processing.time_domain_stacking(time_series, cyc_len, cyc_num)
            freqs, spectrum = signal_processing.fft_short(stacked_wave, rx_sr)

            print("[√] 时域叠加及 FFT 计算完成！正在出图...")

            # --- 图1：全部时域数据图片 ---
            vis.plot_waveform(time_series, title=f"Raw Data (All Cycles) - Period {p_idx} Ch {ch_idx}", vlines=cyc_len)

            # --- 图2：叠加之后的波形图片 ---
            vis.plot_waveform(stacked_wave,
                              title=f"Time-Domain Stacked Waveform (Averaged {cyc_num} Cycles) - Period {p_idx} Ch {ch_idx}")

            # --- 图3：FFT 之后的图片 ---
            vis.plot_analyzed_spectrum(freqs_fft=freqs, yf_fft=spectrum, fundamental_freq=f0, num_harmonics=15,
                                       title=f"Stacked Spectrum (Time-Domain) - Period {p_idx} Ch {ch_idx}")

        # ---------------------------------------------------------
        elif choice == '3':
            print("\n>>> 正在执行: [3] 叠加傅里叶变换 - 频域...")
            stem = Path(rx_filepath).stem
            ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"

            try:
                p_idx = int(input("👉 请输入要测试的周期号 (例如 1): ").strip())
                ch_idx = int(input("👉 请输入要处理的通道号 (例如 0 或 1): ").strip())
            except ValueError:
                print("❌ 输入无效，请输入整数！")
                continue

            target_file = ts_dir / f"{stem}_#Period={p_idx:02d}_Timeseries.txt"
            if not target_file.exists():
                print(f"❌ 找不到文件: {target_file}")
                continue

            print(f"  正在加载数据: {target_file.name}")
            data = np.loadtxt(target_file, skiprows=2)
            time_series = data[:, ch_idx]

            rx_header, _, _, rx_sr, _ = data_io.read_age_binary(rx_filepath)
            cyc_len = int(rx_header['Isw'][30 + (p_idx - 1)])
            cyc_num = int(rx_header['Isw'][60 + (p_idx - 1)])

            # 【提前交互】获取基频
            theory_f0 = rx_sr / cyc_len
            user_f0 = input(f"👉 请输入发射基频 (直接回车使用 {theory_f0:.4f} Hz): ").strip()
            f0 = theory_f0 if not user_f0 else float(user_f0)

            print(f"  [>] 执行参数: 循环长度={cyc_len}, 循环次数={cyc_num}")

            # 去直流偏置，防止全长 FFT 零频泄漏
            time_series = time_series - np.mean(time_series)

            # 1. 频域叠加计算
            freqs_stacked, spectrum_stacked = signal_processing.fft_freq_stacking(
                timeseries=time_series, sample_rate=rx_sr, cyc_len=cyc_len, cyc_num=cyc_num, window_type='rect'
            )

            # 2. 长序列全长 FFT (为了给出和功能6一样的全部频率图片)
            freqs_all, amp_all = signal_processing.long_fft_with_window(
                time_series, rx_sr, window_type='rect'
            )

            print("[√] 频域叠加计算完成！正在出图...")

            # --- 图1：全部时域数据图片 ---
            vis.plot_waveform(time_series, title=f"Raw Data (All Cycles) - Period {p_idx} Ch {ch_idx}", vlines=cyc_len)

            # --- 图2：FFT之后的全部频率图片 ---
            vis.plot_overall_spectrum(freqs_fft=freqs_all, yf_fft=amp_all,
                                      title=f"All Frequency Spectrum (Unstacked Long FFT) - Period {p_idx} Ch {ch_idx}")

            # --- 图3：频域叠加之后的结果图片 ---
            vis.plot_analyzed_spectrum(freqs_fft=freqs_stacked, yf_fft=spectrum_stacked, fundamental_freq=f0,
                                       num_harmonics=15,
                                       title=f"Stacked Spectrum (Frequency-Domain) - Period {p_idx} Ch {ch_idx}")

        # ---------------------------------------------------------
        elif choice == '4':
            print("\n>>> 开启交互式时域查看器...")
            inp = input("👉 要查看哪一端的时域数据 [T/R]: ").strip().upper()
            if inp == 'T':
                stem = Path(tx_filepath).stem
                ts_dir = Path(tx_filepath).parent / f"{stem}_Timeseries"
                vis.interactive_time_viewer(ts_dir, stem)
            elif inp == 'R':
                stem = Path(rx_filepath).stem
                ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"
                vis.interactive_time_viewer(ts_dir, stem)
            else:
                print("❌ 输入无效！")

        # ---------------------------------------------------------
        elif choice == '5':
            print("\n>>> 开启交互式频域查看器...")
            inp = input("👉 要查看哪一端的频谱数据 [T/R]: ").strip().upper()
            if inp == 'T':
                stem = Path(tx_filepath).stem
                fs_dir = Path(tx_filepath).parent / f"{stem}_FreqSeries"
                vis.interactive_freq_viewer(fs_dir, stem)
            elif inp == 'R':
                stem = Path(rx_filepath).stem
                fs_dir = Path(rx_filepath).parent / f"{stem}_FreqSeries"
                vis.interactive_freq_viewer(fs_dir, stem)
            else:
                print("❌ 输入无效！")

        # ---------------------------------------------------------
        elif choice == '6':
            print("\n>>> 正在执行: [6] 测试 - 不叠加，直接加窗长序列 FFT...")
            stem = Path(rx_filepath).stem
            ts_dir = Path(rx_filepath).parent / f"{stem}_Timeseries"

            try:
                p_idx = int(input("👉 请输入要测试的周期号 (例如 1): ").strip())
                ch_idx = int(input("👉 请输入要处理的通道号 (例如 0 或 1): ").strip())
            except ValueError:
                print("❌ 输入无效，请输入整数！")
                continue

            target_file = ts_dir / f"{stem}_#Period={p_idx:02d}_Timeseries.txt"
            if not target_file.exists():
                print(f"❌ 找不到文件: {target_file}")
                continue

            data = np.loadtxt(target_file, skiprows=2)
            time_series = data[:, ch_idx]

            # 去直流偏置
            time_series = time_series - np.mean(time_series)

            rx_header, _, _, rx_sr, _ = data_io.read_age_binary(rx_filepath)
            cyc_len = int(rx_header['Isw'][30 + (p_idx - 1)])
            theory_f0 = rx_sr / cyc_len

            user_f0 = input(f"👉 请输入发射基频 (直接回车使用 {theory_f0:.4f} Hz): ").strip()
            f0 = theory_f0 if not user_f0 else float(user_f0)

            print(f"  [>] 原始序列总点数: {len(time_series)} 点")
            print("  [>] 正在应用 Hann 窗并执行全长 FFT...")

            freqs, amplitude = signal_processing.long_fft_with_window(time_series, rx_sr, window_type='react')
            print("[√] 计算完成！正在出图...")

            vis.plot_analyzed_spectrum(
                freqs_fft=freqs,
                yf_fft=amplitude,
                fundamental_freq=f0,
                num_harmonics=15,
                title=f"Unstacked Long-Sequence FFT (Window: Hann) - Period {p_idx} Ch {ch_idx}")

        # ---------------------------------------------------------
        elif choice == '0':
            print("\n👋 感谢使用，系统已退出！")
            sys.exit(0)

        else:
            print("\n❌ 错误: 无效的输入，请输入 0 到 6 之间的数字！")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 强制中断，系统已退出！")
        sys.exit(0)