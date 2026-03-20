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
        print("  2. 叠加傅里叶变换-时域 (先叠波形再FFT)")
        print("  3. 叠加傅里叶变换-频域 (先FFT再均值化) [待完善]")
        print("  4. [交互] 查看时域波形")
        print("  5. [交互] 查看全频段频谱")
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

            print(f"  [>] 执行参数: 循环长度={cyc_len}, 循环次数={cyc_num}")
            stacked_wave = signal_processing.time_domain_stacking(time_series, cyc_len, cyc_num)
            freqs, spectrum = signal_processing.fft_short(stacked_wave, rx_sr)
            print("[√] 时域叠加及 FFT 计算完成！")

            raw_preview_len = cyc_len * 3
            vis.plot_waveform(time_series[:raw_preview_len],
                                title=f"Raw Data (First 3 Cycles) - Ch {ch_idx}",
                                vlines=cyc_len)
            vis.plot_waveform(stacked_wave,
                                title=f"Time-Domain Stacked Waveform (Averaged {cyc_num} Cycles) - Ch {ch_idx}")
            if input("  ❓ 是否查看叠加后的 FFT 频谱图？(y/n): ").strip().lower() == 'y':
                # 推算理论基频
                theory_f0 = rx_sr / cyc_len
                user_f0 = input(f"👉 请输入发射基频 (直接回车使用 {theory_f0:.4f} Hz): ").strip()
                f0 = theory_f0 if not user_f0 else float(user_f0)

                # 调用我们写好的工业级精密画图函数
                vis.plot_analyzed_spectrum(
                    freqs_fft=freqs,
                    yf_fft=spectrum,
                    fundamental_freq=f0,
                    num_harmonics=15,
                    title=f"Stacked Spectrum (Time-Domain) - Ch {ch_idx}"
                )


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

            # 提取头文件参数
            rx_header, _, _, rx_sr, _ = data_io.read_age_binary(rx_filepath)
            cyc_len = int(rx_header['Isw'][30 + (p_idx - 1)])
            cyc_num = int(rx_header['Isw'][60 + (p_idx - 1)])

            # 提前交互，防止图表卡死空白
            theory_f0 = rx_sr / cyc_len
            user_f0 = input(f"👉 请输入发射基频 (直接回车使用 {theory_f0:.4f} Hz): ").strip()
            f0 = theory_f0 if not user_f0 else float(user_f0)

            print(f"  [>] 执行参数: 循环长度={cyc_len}, 循环次数={cyc_num}")
            print("  [>] 正在对每个周期独立加窗、FFT，并在频域进行均值化叠加...")

            # ==========================================
            # 1. 计算频域叠加后的结果 (用于最终的精密分析)
            # ==========================================
            freqs, spectrum = signal_processing.fft_freq_stacking(
                timeseries=time_series,
                sample_rate=rx_sr,
                cyc_len=cyc_len,
                cyc_num=cyc_num,
                window_type='hann'
            )

            # ==========================================
            # 【核心修改】：2. 单独计算第一个单周期的 FFT (不叠加)
            # ==========================================
            single_cycle = time_series[:cyc_len]  # 切出第1个周期的波形
            freqs_single, spectrum_single = signal_processing.fft_no_stack(
                single_cycle, rx_sr, window_type='hann'
            )

            print("[√] 频域叠加计算完成！正在出图...")

            # ------------------------------------------
            # 画图 1：只输出单周期变过去的整体频率图 (看叠加前的真实底噪)
            # ------------------------------------------
            vis.plot_overall_spectrum(
                freqs_fft=freqs_single,
                yf_fft=spectrum_single,
                title=f"Single Period Spectrum (Unstacked) - Ch {ch_idx}"
            )

            # ------------------------------------------
            # 画图 2：输出频域叠加后的精密分析图 (看叠加后的提纯结果)
            # ------------------------------------------
            vis.plot_analyzed_spectrum(
                freqs_fft=freqs,
                yf_fft=spectrum,
                fundamental_freq=f0,
                num_harmonics=15,
                title=f"Stacked Spectrum (Frequency-Domain) - Ch {ch_idx}"
            )

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

            #hann表示加窗函数，其他识别为不加窗函数
            freqs, amplitude = signal_processing.long_fft_with_window(time_series, rx_sr, window_type='react')
            print("[√] 计算完成！正在出图...")

            vis.plot_hybrid_spectrum(
                time_series=time_series,
                sample_rate=rx_sr,
                freqs_fft=freqs,
                yf_fft=amplitude,
                fundamental_freq=f0,
                num_harmonics=15,
                extra_freqs=[50.0],
                title=f"Unstacked Long-Sequence FFT (Window: Hann) - Ch {ch_idx}"
            )

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