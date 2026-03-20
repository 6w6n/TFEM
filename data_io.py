import numpy as np
from pathlib import Path
import signal_processing

file_header_dtype = np.dtype([
    ('day', 'i2'), ('month', 'i2'), ('year', 'i2'), ('hour', 'i2'), ('minute', 'i2'),
    ('geo', 'S20'), ('backup0', 'i1'), ('met', 'i1'), ('backup1', '20i1'),
    ('LST', 'i1'), ('pro', 'i1'), ('kan', 'i1'), ('backup2', 'i1'),
    ('ab1', 'i2'), ('ab2', 'i2'), ('ab3', 'i2'), ('key25', 'i2'),
    ('Tok', 'f4'), ('Ndt', 'i2'), ('T0', 'i2'), ('Tgu', 'i1'), ('Ddt', 'i1'),
    ('Tpi', 'i1'), ('Ngu', 'i1'), ('colibr_period', 'i4'), ('colibr_pulse', 'i4'),
    ('backup3', '8i1'), ('Nom', 'i2'), ('kdt', 'i2'), ('Pr2', 'i2'), ('Pima', 'i2'),
    ('Fam', 'S20'), ('Lps', 'i4'), ('backup4', 'S6'), ('backup5', 'S1'),
    ('Npb', 'i1'), ('Izm', 'i2'), ('Ntk', 'i2'), ('backup6', 'S2'), ('Lgr', 'i2'),
    ('Ntik', 'i2'), ('Nst', 'i2'), ('backup7', 'S6'), ('backup8', 'S1'), ('Vup', 'i1'),
    ('Com', 'S48'), ('Isw', '90i2'), ('ProgNum', 'i2'), ('fNid', 'i4'),
    ('Pd_16', 'i2'), ('Pd_24', 'i4'), ('StartMode', 'i2'), ('StartExt', 'i2'),
    ('Pac', 'i2'), ('T_gps', 'i4'), ('T_kp', 'i4'), ('backup9', 'S18'),
    ('iADType', 'i2'), ('backup10', 'S86')
])

chan_header_dtype = np.dtype([
    ('Idk', 'i1'), ('standby0', 'i1'), ('Uko', 'i1'), ('Ufl', 'i1'),
    ('Pkt', 'i2'), ('Prf', 'i2'), ('Damp', 'i2'), ('Ddac', 'i2'),
    ('standby3', 'S2'), ('Razm', 'i2'), ('Nvt', 'i2'), ('Ubal', 'i2'),
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('Ecs', 'f4'), ('standby5', 'S4'),
    ('k1', 'f4'), ('k10', 'f4'), ('k100', 'f4'), ('k1000', 'f4'),
    ('Ugl', 'f4'), ('Er', 'f4')
])


def read_age_binary(filepath):
    """读取 AGE 二进制文件，直接提取原始数据机器码（不进行物理转换）"""
    with open(filepath, 'rb') as f:
        file_header = np.frombuffer(f.read(512), dtype=file_header_dtype)[0]
        n_chan = int(file_header['kan'])
        chan_headers = np.frombuffer(f.read(64 * n_chan), dtype=chan_header_dtype)

        if file_header['Vup'] < 100:
            sample_rate = 1000.0 / (2.0 ** (file_header['pro'] - 1))
        else:
            sample_rate = float(file_header['Ndt'])

        n_max_period = file_header['Pima']

        f.seek(2048)
        raw_data = np.fromfile(f, dtype=np.int32).astype(np.float64)
        n_total_sam = len(raw_data) // n_chan

        timeseries = raw_data[:n_total_sam * n_chan].reshape((n_chan, n_total_sam), order='F')
        timeseries = np.nan_to_num(timeseries, nan=0.0, posinf=0.0, neginf=0.0)

    return file_header, chan_headers, timeseries, sample_rate, n_max_period


def export_info_file(filepath, out_dir, header, chans, sr, periods):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filepath).stem

    with open(out_dir / f"{stem}.info", 'w', encoding='utf-8') as f:
        f.write(f"  SampleRate :    {sr:.3E}\n")
        f.write(f"  #MaxPeriod :      {periods}\n")
        f.write("         #Chan          Gain            MN           ADC           PRF           PKT           IDK\n")
        for i in range(len(chans)):
            uko = chans[i]['Uko']
            gain_map = {1: 'k1', 2: 'k1', 3: 'k10', 4: 'k100', 5: 'k1000'}
            gain = chans[i][gain_map.get(uko, 'k1')]
            if np.abs(gain) < 1e-15 or np.isnan(gain): gain = 1.0
            f.write(f"  {i + 1:10}  {gain:12.5f}  {1.0:12.5f}  {chans[i]['Ecs']:12.5f}  "
                    f"{chans[i]['Prf']:12}  {chans[i]['Pkt']:12}  {chans[i]['Idk']:12}\n")

        f.write("\n   #Period   #CycLen   #CycNum\n")
        for i in range(periods):
            f.write(f"  {i + 1:8}  {int(header['Isw'][30 + i]):8}  {int(header['Isw'][60 + i]):8}\n")


def export_timeseries(filepath, out_dir, period_idx, cyc_len, cyc_num, n_chan, ts_data):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filepath).stem

    with open(out_dir / f"{stem}_#Period={period_idx:02d}_Timeseries.txt", 'w', encoding='utf-8') as f:
        f.write("   #Period   #CycLen   #CycNum     #Chan\n")
        f.write(f"  {period_idx:8d}  {cyc_len:8d}  {cyc_num:8d}  {n_chan:8d}\n")
        np.savetxt(f, ts_data.T, fmt="  %30.12E")


def process_and_export_all_data(tx_filepath, rx_filepath):
    print(">>> [1/3] 正在加载并处理数据...")
    tx_header, tx_chans, tx_ts, tx_sr, tx_periods = read_age_binary(tx_filepath)
    rx_header, rx_chans, rx_ts, rx_sr, rx_periods = read_age_binary(rx_filepath)

    n_max_period = min(tx_periods, rx_periods)
    n_chan_tx, n_chan_rx = int(tx_header['kan']), int(rx_header['kan'])

    tx_path, rx_path = Path(tx_filepath), Path(rx_filepath)
    tx_ts_dir, tx_fs_dir = tx_path.parent / f"{tx_path.stem}_Timeseries", tx_path.parent / f"{tx_path.stem}_FreqSeries"
    rx_ts_dir, rx_fs_dir = rx_path.parent / f"{rx_path.stem}_Timeseries", rx_path.parent / f"{rx_path.stem}_FreqSeries"

    for d in [tx_ts_dir, rx_ts_dir, rx_fs_dir, tx_fs_dir]: d.mkdir(exist_ok=True)

    print(">>> [2/3] 正在导出 .info 描述文件...")
    export_info_file(tx_filepath, tx_ts_dir, tx_header, tx_chans, tx_sr, tx_periods)
    export_info_file(rx_filepath, rx_ts_dir, rx_header, rx_chans, rx_sr, rx_periods)

    print(">>> [3/3] 正在进行 FFT 并导出 txt 文件...")
    curr_tx_idx, curr_rx_idx = 0, 0

    for i in range(n_max_period):
        cyc_len, cyc_num = int(rx_header['Isw'][30 + i]), int(rx_header['Isw'][60 + i])
        n_sam = cyc_len * cyc_num

        if n_sam <= 0 or curr_rx_idx + n_sam > rx_ts.shape[1]: break

        tx_seg_all = tx_ts[:, curr_tx_idx: curr_tx_idx + n_sam]
        rx_seg_all = rx_ts[:, curr_rx_idx: curr_rx_idx + n_sam]

        export_timeseries(tx_filepath, tx_ts_dir, i + 1, cyc_len, cyc_num, n_chan_tx, tx_seg_all)
        export_timeseries(rx_filepath, rx_ts_dir, i + 1, cyc_len, cyc_num, n_chan_rx, rx_seg_all)

        def _export_spectrum(seg_all, n_chan, sr, out_dir, path_stem):
            export_data = []
            for c in range(n_chan):
                # 核心修复点：使用纯粹的 FFT 提取复数，不再除以 n_sam
                xf, yf = signal_processing.fft_no_stack(seg_all[c, :], sr, window_type='hann')
                if c == 0: export_data.append(xf)
                export_data.extend([np.real(yf), np.imag(yf)])

            header_str = "Freq(Hz)" + "".join([f"\tCh{c + 1}_Re\tCh{c + 1}_Im" for c in range(n_chan)])
            out_file = out_dir / f"{path_stem}_#Period={i + 1:02d}_Spectrum.txt"
            np.savetxt(out_file, np.column_stack(export_data), fmt="%.6e", delimiter="\t", header=header_str,
                       comments="")

        _export_spectrum(tx_seg_all, n_chan_tx, tx_sr, tx_fs_dir, tx_path.stem)
        _export_spectrum(rx_seg_all, n_chan_rx, rx_sr, rx_fs_dir, rx_path.stem)

        curr_tx_idx += n_sam
        curr_rx_idx += n_sam

    print("[√] 数据解析并导出完毕！")