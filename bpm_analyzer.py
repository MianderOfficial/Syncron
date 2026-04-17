
"""
╔══════════════════════════════════════════════════════════╗
║             PRECISION BPM ANALYZER  v2.1                 ║
║    Methods: librosa DP · autocorr · tempogram            ║
║    + Plots: waveform · onset · tempogram · mel spec      ║
╚══════════════════════════════════════════════════════════╝

Usage:
    python bpm_analyzer.py audio.mp3
    python bpm_analyzer.py audio.wav --start 10 --duration 60
    python bpm_analyzer.py audio.flac --all --plot
"""

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import librosa.beat
import librosa.onset
from scipy.signal import correlate


class C:
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════╗
║             PRECISION BPM ANALYZER  v2.1                 ║
║    Methods: librosa · autocorr · tempogram               ║
╚══════════════════════════════════════════════════════════╝{C.RESET}
""")


def load_audio(path: str, offset: float = 0.0, duration: float = None) -> tuple:
    if not os.path.exists(path):
        print(f"{C.RED}Error: file '{path}' not found.{C.RESET}")
        sys.exit(1)

    print(f"{C.DIM}  Loading: {os.path.basename(path)}{C.RESET}", end=" ", flush=True)
    try:
        y, sr = librosa.load(path, sr=44100, offset=offset, duration=duration, mono=True)
    except Exception as e:
        print(f"\n{C.RED}Failed to load audio: {e}{C.RESET}")
        sys.exit(1)

    duration_actual = len(y) / sr
    print(f"{C.GREEN}OK{C.RESET}  [{duration_actual:.1f} sec, {sr} Hz]")
    return y, sr


def normalize_to_range(bpm: float, lo: float = 80.0, hi: float = 160.0) -> float:
    """
    Forces the BPM into the [lo, hi] range by doubling or halving it.
    The 80-160 range covers most popular genres and gets rid of 
    those pesky x2 or x0.5 octave jumps.
    """
    if bpm <= 0:
        return bpm
    while bpm < lo:
        bpm *= 2.0
    while bpm > hi:
        bpm /= 2.0
    return bpm


def algo_librosa_dp(y: np.ndarray, sr: int) -> dict:
    """
    Good old librosa default: onset envelope + dynamic programming beat tracking.
    """
    hop_length = 512
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length,
        aggregate=np.median,
        n_mels=128,
        fmax=8000,
    )

    tempo_arr, beats = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        tightness=100,
        trim=True,
    )

    tempo = float(np.atleast_1d(tempo_arr)[0])

    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    if len(beat_times) > 4:
        ibi = np.diff(beat_times)
        ibi = _iqr_filter(ibi)
        if len(ibi) > 2:
            w = np.hamming(len(ibi))
            tempo = 60.0 / np.average(ibi, weights=w)

    return {
        "name": "librosa DP",
        "bpm": round(tempo, 2),
        "beat_times": beat_times,
        "beat_count": len(beat_times),
    }


def algo_autocorrelation(y: np.ndarray, sr: int) -> dict:
    """
    Finds the BPM by auto-correlating the onset envelope.
    """
    hop_length = 256
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length,
        aggregate=np.median,
        n_mels=128,
        fmax=8000,
    )

    onset_env = onset_env - onset_env.mean()
    std = onset_env.std()
    if std < 1e-8:
        return {"name": "autocorr", "bpm": 0.0}
    onset_env /= std

    ac = correlate(onset_env, onset_env, mode="full")
    ac = ac[len(ac) // 2:]

    fps = sr / hop_length
    min_lag = max(int(60.0 / 250.0 * fps), 1)
    max_lag = int(60.0 / 40.0 * fps)

    if min_lag >= max_lag or max_lag > len(ac):
        return {"name": "autocorr", "bpm": 0.0}

    ac_range = ac[min_lag:max_lag]
    kernel = np.hanning(5)
    kernel /= kernel.sum()
    smoothed = np.convolve(ac_range, kernel, mode="same")

    peak_idx = int(np.argmax(smoothed))
    best_lag = peak_idx + min_lag

    if 1 <= peak_idx < len(ac_range) - 1:
        y0, y1, y2 = smoothed[peak_idx - 1], smoothed[peak_idx], smoothed[peak_idx + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom
            best_lag = best_lag + delta

    bpm = 60.0 / (best_lag / fps)

    return {
        "name": "autocorr",
        "bpm": round(bpm, 2),
    }


def algo_tempogram(y: np.ndarray, sr: int) -> dict:
    """
    Uses fourier_tempogram instead of the autocorrelation one to avoid inf/nan freqs.
    Filters out the crap outside 40-250 BPM and interpolates the peak.
    """
    hop_length = 512
    win_length = 384

    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length,
        n_mels=128,
        fmax=8000,
    )

    ftg = librosa.feature.fourier_tempogram(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
    )

    ftg_mean = np.abs(ftg).mean(axis=1)

    tempo_freqs = librosa.fourier_tempo_frequencies(
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
    )

    valid = (tempo_freqs >= 40) & (tempo_freqs <= 250) & np.isfinite(tempo_freqs) & np.isfinite(ftg_mean)
    if not np.any(valid):
        return {"name": "tempogram", "bpm": 0.0, "candidates": []}

    ftg_valid   = ftg_mean[valid]
    freqs_valid = tempo_freqs[valid]

    top3_idx = np.argsort(ftg_valid)[-3:][::-1]
    candidates = freqs_valid[top3_idx]

    best_i = top3_idx[0]

    bpm = _parabolic_interp(freqs_valid, ftg_valid, best_i)

    return {
        "name": "tempogram",
        "bpm": round(bpm, 2),
        "candidates": [round(c, 2) for c in candidates],
    }


def _parabolic_interp(x: np.ndarray, y: np.ndarray, idx: int) -> float:
    """Sub-pixel accuracy for finding the peak in an array."""
    if idx <= 0 or idx >= len(x) - 1:
        return float(x[idx])
    y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
    denom = 2.0 * (2.0 * y1 - y0 - y2)
    if abs(denom) < 1e-12:
        return float(x[idx])
    delta = (y0 - y2) / denom
    x_step = float(x[min(idx + 1, len(x) - 1)] - x[max(idx - 1, 0)]) / 2.0
    return float(x[idx]) + delta * x_step


def _iqr_filter(data: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Standard IQR filter to strip out extreme outliers."""
    if len(data) < 4:
        return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    clean = data[(data >= lo) & (data <= hi)]
    return clean if len(clean) > 1 else data



def weighted_consensus(results: list) -> float:
    """
    Clusters the results to find a consensus.
    Normalizes everything first so we compare apples to apples.
    """
    bpms = [r["bpm"] for r in results if r.get("bpm", 0) > 0 and np.isfinite(r["bpm"])]
    if not bpms:
        return 0.0
    if len(bpms) == 1:
        return bpms[0]

    normalized = [normalize_to_range(b) for b in bpms]

    weights = [1.0] * len(bpms)
    for i in range(len(bpms)):
        for j in range(len(bpms)):
            if i != j:
                diff = abs(normalized[i] - normalized[j])
                if diff < 3.0:
                    weights[i] += 2.0
                elif diff < 8.0:
                    weights[i] += 0.5

    best = int(np.argmax(weights))

    return round(normalize_to_range(bpms[best]), 2)

def analyze_stability(y: np.ndarray, sr: int, segment_sec: float = 30.0) -> dict:
    """
    Chops the track into segments to see if the tempo drifts.
    Uses autocorrelation for speed since DP can be sluggish on long tracks.
    """
    total_sec = len(y) / sr
    if total_sec < segment_sec * 2:
        return {"stable": True, "variation": 0.0, "segments": []}

    n_seg = min(int(total_sec // segment_sec), 6)
    seg_bpms = []

    for i in range(n_seg):
        start = int(i * segment_sec * sr)
        end   = int((i + 1) * segment_sec * sr)
        y_seg = y[start:end]
        try:
            res = algo_autocorrelation(y_seg, sr)
            bpm = normalize_to_range(res["bpm"])
            if bpm > 0:
                seg_bpms.append(bpm)
        except Exception:
            pass

    if not seg_bpms:
        return {"stable": True, "variation": 0.0, "segments": []}

    variation = max(seg_bpms) - min(seg_bpms)
    return {
        "stable": variation < 2.0,
        "variation": round(variation, 2),
        "segments": [round(b, 2) for b in seg_bpms],
        "min_bpm": round(min(seg_bpms), 2),
        "max_bpm": round(max(seg_bpms), 2),
    }


def plot_analysis(y: np.ndarray, sr: int, results: list, consensus: float, filepath: str):
    """
    Spits out a nice UI with 4 plots: waveform, onset env, tempogram, and mel spec.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import librosa.display
    except ImportError:
        print(f"{C.YELLOW}  ⚠ matplotlib missing. Run: pip install matplotlib{C.RESET}")
        return

    hop = 512
    print(f"  {C.DIM}Generating onset envelope...{C.RESET}", end=" ", flush=True)
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_mels=128, fmax=8000)
    times_env = librosa.times_like(oenv, sr=sr, hop_length=hop)
    print(f"{C.GREEN}OK{C.RESET}")

    print(f"  {C.DIM}Generating tempogram...{C.RESET}", end=" ", flush=True)
    win_length = 384
    ftg = np.abs(librosa.feature.fourier_tempogram(
        onset_envelope=oenv, sr=sr, hop_length=hop, win_length=win_length,
    ))
    tempo_freqs = librosa.fourier_tempo_frequencies(sr=sr, hop_length=hop, win_length=win_length)
    print(f"{C.GREEN}OK{C.RESET}")

    beat_times = None
    for r in results:
        if r.get("beat_times") is not None and len(r["beat_times"]) > 0:
            beat_times = r["beat_times"]
            break

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.55)

    fig.suptitle(
        f"BPM Analysis · {os.path.basename(filepath)} · Result: {consensus:.2f} BPM",
        color="white", fontsize=14, fontweight="bold",
    )

    ax1 = fig.add_subplot(gs[0])
    max_samples = 100_000
    if len(y) > max_samples:
        step = len(y) // max_samples
        y_plot = y[::step]
    else:
        y_plot = y
    t_wave = np.linspace(0, len(y) / sr, len(y_plot))
    ax1.fill_between(t_wave, y_plot, alpha=0.75, color="#4fc3f7")
    if beat_times is not None:
        for bt in beat_times:
            ax1.axvline(x=bt, color="#ffca28", linewidth=0.5, alpha=0.55)
    ax1.set_title("Waveform (yellow lines = beats)", color="#aaa", fontsize=10)
    ax1.set_xlabel("Time (s)", color="#888", fontsize=8)
    ax1.set_ylabel("Amplitude", color="#888", fontsize=8)
    ax1.tick_params(colors="#666")
    ax1.set_facecolor("#111")
    for sp in ax1.spines.values():
        sp.set_color("#333")


    ax2 = fig.add_subplot(gs[1])
    ax2.plot(times_env, oenv, color="#ff7043", linewidth=0.8)
    ax2.fill_between(times_env, oenv, alpha=0.3, color="#ff7043")
    if beat_times is not None:
        for bt in beat_times:
            ax2.axvline(x=bt, color="#ffca28", linewidth=0.5, alpha=0.55)
    ax2.set_title("Onset Strength Envelope", color="#aaa", fontsize=10)
    ax2.set_xlabel("Time (s)", color="#888", fontsize=8)
    ax2.set_ylabel("Strength", color="#888", fontsize=8)
    ax2.tick_params(colors="#666")
    ax2.set_facecolor("#111")
    for sp in ax2.spines.values():
        sp.set_color("#333")

    ax3 = fig.add_subplot(gs[2])
    ftg_mean = ftg.mean(axis=1)
    mask = (tempo_freqs >= 40) & (tempo_freqs <= 250) & np.isfinite(tempo_freqs) & np.isfinite(ftg_mean)
    ax3.plot(tempo_freqs[mask], ftg_mean[mask], color="#ab47bc", linewidth=1.2)
    ax3.fill_between(tempo_freqs[mask], ftg_mean[mask], alpha=0.3, color="#ab47bc")
    ax3.axvline(x=consensus, color="#66bb6a", linewidth=2.0,
                linestyle="--", label=f"Result: {consensus:.2f} BPM")
    
    colors_alg = ["#4fc3f7", "#ff7043", "#ffca28"]
    for i, r in enumerate(results):
        bpm_n = normalize_to_range(r["bpm"])
        if 40 <= bpm_n <= 250:
            ax3.axvline(x=bpm_n, color=colors_alg[i % len(colors_alg)],
                        linewidth=1.0, linestyle=":", alpha=0.7,
                        label=f"{r['name']}: {bpm_n:.1f}")
    ax3.set_title("Fourier Tempogram (Time-Averaged)", color="#aaa", fontsize=10)
    ax3.set_xlabel("BPM", color="#888", fontsize=8)
    ax3.set_ylabel("Magnitude", color="#888", fontsize=8)
    ax3.legend(fontsize=8, facecolor="#222", labelcolor="white", loc="upper right")
    ax3.tick_params(colors="#666")
    ax3.set_facecolor("#111")
    for sp in ax3.spines.values():
        sp.set_color("#333")

    ax4 = fig.add_subplot(gs[3])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop, x_axis="time", y_axis="mel",
        fmax=8000, ax=ax4, cmap="magma",
    )
    plt.colorbar(img, ax=ax4, format="%+2.0f dB")
    ax4.set_title("Mel Spectrogram", color="#aaa", fontsize=10)
    ax4.tick_params(colors="#666")
    ax4.set_facecolor("#111")
    for sp in ax4.spines.values():
        sp.set_color("#333")

    os.makedirs("plots", exist_ok=True)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    out_path = os.path.join("plots", f"{base_name}.png")
    
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print(f"  {C.GREEN}✓ Saved plot to: {out_path}{C.RESET}")
    plt.show()


def print_results(results: list, consensus: float, stability: dict, show_all: bool = False):
    print(f"\n{C.CYAN}{C.BOLD}{'─'*56}")
    print("  ANALYSIS RESULTS")
    print(f"{'─'*56}{C.RESET}")

    if show_all:
        print(f"\n{C.YELLOW}  Algorithms:{C.RESET}")
        for r in results:
            if r["bpm"] <= 0 or not np.isfinite(r["bpm"]):
                print(f"    {r['name']:<22} {C.RED}no data{C.RESET}")
                continue
            bpm_str = f"{r['bpm']:.2f} BPM"
            bpm_norm = normalize_to_range(r['bpm'])
            norm_str = f"  {C.DIM}→ norm: {bpm_norm:.2f}{C.RESET}" if abs(bpm_norm - r['bpm']) > 0.5 else ""
            extra = ""
            if "candidates" in r and r["candidates"]:
                extra = f"  {C.DIM}(candidates: {', '.join(str(c) for c in r['candidates'][:3])}){C.RESET}"
            if "beat_count" in r:
                extra += f"  {C.DIM}[{r['beat_count']} beats]{C.RESET}"
            print(f"    {r['name']:<22} {C.BOLD}{bpm_str}{C.RESET}{norm_str}{extra}")

    print(f"\n  {C.GREEN}{C.BOLD}▶  FINAL BPM:  {consensus:.2f}{C.RESET}")

    bpm = consensus
    if   bpm < 60:   genre = "Larghetto / Very Slow"
    elif bpm < 76:   genre = "Adagio / Slow"
    elif bpm < 108:  genre = "Andante-Moderato / Walking Pace"
    elif bpm < 120:  genre = "Allegretto / Moderately Fast"
    elif bpm < 140:  genre = "Allegro / Fast"
    elif bpm < 168:  genre = "Vivace / Very Fast"
    elif bpm < 200:  genre = "Presto / Super Fast"
    else:            genre = "Prestissimo / Extremely Fast"

    print(f"  {C.DIM}   Tempo: {genre}{C.RESET}")

    if stability.get("segments"):
        print(f"\n  {C.YELLOW}Tempo Stability:{C.RESET}")
        if stability["stable"]:
            print(f"    {C.GREEN}Stable  ({stability['variation']:.2f} BPM variation){C.RESET}")
        else:
            print(f"    {C.RED}Unstable  (±{stability['variation']:.2f} BPM variation){C.RESET}")
            print(f"    {C.DIM}Range: {stability['min_bpm']} – {stability['max_bpm']} BPM{C.RESET}")

        segs = stability["segments"]
        mn, mx = min(segs), max(segs)
        rng = max(mx - mn, 0.1)
        bar = ""
        for b in segs:
            h = int((b - mn) / rng * 6)
            bar += "▁▂▃▄▅▆▇"[h]
        print(f"    {C.DIM}Segment drift: {bar}{C.RESET}")

    print(f"\n{C.CYAN}{'─'*56}{C.RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Precise BPM detection for audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("file",              help="Audio file (mp3/wav/flac/ogg/...)")
    parser.add_argument("--start",  "-s",    type=float, default=0.0,  metavar="SEC",
                        help="Start analysis from this second (default: 0)")
    parser.add_argument("--duration", "-d",  type=float, default=None, metavar="SEC",
                        help="Duration of analysis in seconds (default: full file)")
    parser.add_argument("--all",     "-a",   action="store_true",
                        help="Show output from all internal algorithms")
    parser.add_argument("--plot",    "-p",   action="store_true",
                        help="Generate a summary plot (requires matplotlib)")
    parser.add_argument("--no-stability",    action="store_true",
                        help="Skip stability check for faster execution")
    args = parser.parse_args()

    banner()

    print(f"{C.BOLD}  File:{C.RESET} {args.file}")
    if args.start > 0:
        print(f"{C.BOLD}  Start:{C.RESET} {args.start:.1f} sec")
    if args.duration:
        print(f"{C.BOLD}  Duration:{C.RESET} {args.duration:.1f} sec")
    print()

    y, sr = load_audio(args.file, offset=args.start, duration=args.duration)

    print(f"\n{C.BOLD}  Analyzing...{C.RESET}")
    results = []

    print(f"  {C.DIM}[1/3] librosa dynamic programming...{C.RESET}", end=" ", flush=True)
    r1 = algo_librosa_dp(y, sr)
    results.append(r1)
    print(f"{C.GREEN}{r1['bpm']:.2f} BPM{C.RESET}")

    print(f"  {C.DIM}[2/3] autocorrelation...{C.RESET}", end=" ", flush=True)
    r2 = algo_autocorrelation(y, sr)
    results.append(r2)
    print(f"{C.GREEN}{r2['bpm']:.2f} BPM{C.RESET}")

    print(f"  {C.DIM}[3/3] tempogram...{C.RESET}", end=" ", flush=True)
    r3 = algo_tempogram(y, sr)
    results.append(r3)
    bpm3_str = f"{r3['bpm']:.2f} BPM" if r3['bpm'] > 0 else "no data"
    print(f"{C.GREEN}{bpm3_str}{C.RESET}")

    consensus = weighted_consensus(results)

    if not args.no_stability:
        print(f"\n  {C.DIM}Checking stability across segments...{C.RESET}", end=" ", flush=True)
        stability = analyze_stability(y, sr)
        print(f"{C.GREEN}OK{C.RESET}")
    else:
        stability = {"segments": [], "stable": True, "variation": 0.0}

    print_results(results, consensus, stability, show_all=args.all)

    if args.plot:
        print(f"  {C.CYAN}Generating plots...{C.RESET}")
        plot_analysis(y, sr, results, consensus, args.file)

    print(f"{C.DIM}  BPM={consensus:.2f}{C.RESET}")


if __name__ == "__main__":
    main()
