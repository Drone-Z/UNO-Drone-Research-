import os
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import tkinter as tk
from tkinter import filedialog, messagebox
import ffmpeg


# Input audio file
def probe_audio_sr(path: str) -> int:

    try:
        info = ffmpeg.probe(path)
        for s in info.get("streams", []):
            if s.get("codec_type") == "audio":
                return int(s["sample_rate"])
        raise RuntimeError("No audio stream found in file.")
    except Exception as e:
        raise RuntimeError(f"Could not determine sample rate: {e}")

def load_audio_any(path: str):
    
    sr = probe_audio_sr(path)
    try:
        out, _ = (
            ffmpeg
            .input(path)
            .output(
                "pipe:",
                format="f32le",
                acodec="pcm_f32le",
                ac=1,     
                ar=sr
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        err = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg failed to decode this file.\n\n{err[:800]}")
    y = np.frombuffer(out, dtype=np.float32)
    return y, sr

def find_top_peaks(freqs: np.ndarray, mag_db: np.ndarray, k: int = 5):
    
    if len(mag_db) < 3:
        return []
    idx = np.where((mag_db[1:-1] > mag_db[:-2]) & (mag_db[1:-1] > mag_db[2:]))[0] + 1
    if idx.size == 0:
        return []
    top_idx = idx[np.argsort(mag_db[idx])[-k:]][::-1]
    return [(float(freqs[i]), float(mag_db[i])) for i in top_idx]


# Plotting graph
def analyze_and_plot(path: str, show_peaks: int = 5):
    # Decode
    y, sr = load_audio_any(path)
    if y.size == 0:
        raise ValueError("Decoded audio is empty.")
    y = np.clip(y, -1.0, 1.0)

    # Time axis
    t = np.arange(len(y)) / sr

    # FFT (one-sided, Hann window)
    y0 = y - np.mean(y)
    win = np.hanning(len(y0)) if len(y0) > 1 else np.ones_like(y0)
    yw = y0 * win
    Y = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(len(yw), d=1.0/sr)

    N = len(yw) if len(yw) else 1
    window_correction = (np.sum(win) / N) if N > 0 else 1.0
    mag = np.abs(Y) / (N * window_correction)
    if mag.size > 2:
        mag[1:-1] *= 2.0  
    mag_db = 20 * np.log10(np.maximum(mag, 1e-12))  

    # Normalize waveform for visualization
    peak = np.max(np.abs(y)) or 1.0
    y_vis = y / peak

    # Optional downsample of time plot for very long signals
    max_points = 300_000
    if len(y_vis) > max_points:
        step = int(np.ceil(len(y_vis) / max_points))
        y_vis = y_vis[::step]
        t = t[::step]

    # Figure with stacked axes
    fig, (ax_time, ax_fft) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"{os.path.basename(path)}  •  SR={sr} Hz  •  Nyquist={sr/2:.1f} Hz", y=0.98)

    # Time domain
    ax_time.plot(t, y_vis, linewidth=0.9)
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_title("Time Domain (Waveform)")
    ax_time.grid(True, alpha=0.3)

    # FFT in dB — full native frequency range
    ax_fft.plot(freqs, mag_db, linewidth=0.9)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude (dB)")
    ax_fft.set_title("Frequency Domain (FFT, dB)")
    ax_fft.set_xlim(0, sr/2)
    ax_fft.grid(True, alpha=0.3)

    # Crosshair cursor + click-to-read exact freq/dB
    Cursor(ax_fft, horizOn=True, vertOn=True, color='red', linewidth=1)

    ann = ax_fft.annotate(
        "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="0.5", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="0.5"), visible=False
    )
    def onclick(event):
        if event.inaxes != ax_fft or event.xdata is None:
            return
        x = event.xdata
        idx = int(np.clip(np.searchsorted(freqs, x), 1, len(freqs)-1))
        if abs(freqs[idx] - x) > abs(freqs[idx-1] - x):
            idx -= 1
        fx, dy = freqs[idx], mag_db[idx]
        print(f"Clicked: {fx:.2f} Hz, {dy:.2f} dB")
        ann.xy = (fx, dy)
        ann.set_text(f"{fx:.2f} Hz\n{dy:.2f} dB")
        ann.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', onclick)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Print top spectral peaks
    peaks = find_top_peaks(freqs, mag_db, k=show_peaks)
    if peaks:
        print("\nTop spectral peaks (approx):")
        for f, d in peaks:
            print(f"  {f:10.2f} Hz   {d:8.2f} dB")


def main():
    
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        root = tk.Tk()
        root.withdraw()
        filetypes = [
            ("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac *.wma *.aiff *.aif *.opus"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select an audio file", filetypes=filetypes)

    if not path:
        messagebox.showwarning("No file", "No audio file selected.")
        return

    try:
        analyze_and_plot(path)
    except Exception as e:
        messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    main()
