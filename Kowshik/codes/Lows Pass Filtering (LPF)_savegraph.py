import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import soundfile as sf
import librosa
import librosa.display as ldisplay
from scipy.signal import butter, filtfilt

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# FFmpeg loader for all audio file types
def _probe_audio_sr(path):
    """Get source sample rate via ffprobe. Fallback to 16000 if unknown."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=sample_rate", "-of", "default=nw=1:nk=1", path],
            check=True, capture_output=True, text=True
        )
        return int(out.stdout.strip())
    except Exception:
        return 16000

def load_audio_ffmpeg(path, target_sr=None, mono=True):
    if target_sr is None:
        target_sr = _probe_audio_sr(path)

    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", path,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1" if mono else "2",
        "-ar", str(target_sr),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg binary not found. Install ffmpeg and ensure it’s on PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to decode audio:\n{e.stderr.decode('utf-8', errors='ignore')}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    y = audio.astype(np.float32, copy=False)

    peak = float(np.max(np.abs(y))) if y.size else 1.0
    if peak > 1.0:
        y = y / peak

    return y, target_sr

# Butterworth LPF
def butter_lowpass_filter(y, sr, cutoff_hz=2500.0, order=6):
    nyq = 0.5 * sr
    if cutoff_hz >= nyq:
        raise ValueError(f"cutoff_hz ({cutoff_hz}) must be < Nyquist ({nyq:.1f}) for sr={sr}.")
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(N=order, Wn=normal_cutoff, btype="low", analog=False)
    y_filt = filtfilt(b, a, y).astype(np.float32)
    peak = float(np.max(np.abs(y_filt))) if y_filt.size else 1.0
    if peak > 1.0:
        y_filt /= peak
    return y_filt

def time_axis_for_plot(y, sr, max_points=300_000):
    t = np.arange(len(y)) / sr
    if len(y) > max_points:
        step = int(np.ceil(len(y) / max_points))
        return t[::step], y[::step]
    return t, y

def compute_spectrogram(y, sr, n_fft=2048, hop_length=512, power=1):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
    S = np.abs(S) ** power
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    return S_db

# File pick GUI
class LPFGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Low-Pass Filter (FFmpeg loader + Spectrogram)")
        self.geometry("1150x820")

        self.y = None
        self.sr = None
        self.y_filt = None
        self.in_path = None

        # Track colorbars to prevent stacking
        self._cb_before = None
        self._cb_after = None

        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.btn_open = tk.Button(top, text="Open Audio…", command=self.open_file)
        self.btn_open.pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="Cutoff (Hz):").pack(side=tk.LEFT, padx=(10, 2))
        self.cutoff_var = tk.DoubleVar(value=2500.0)
        self.cutoff_scale = tk.Scale(
            top, from_=200.0, to=12000.0, orient=tk.HORIZONTAL,
            resolution=50.0, variable=self.cutoff_var, length=250,
            command=lambda e: self._maybe_clip_cutoff()
        )
        self.cutoff_scale.pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="Order:").pack(side=tk.LEFT, padx=(10, 2))
        self.order_var = tk.IntVar(value=6)
        self.order_scale = tk.Scale(
            top, from_=2, to=10, orient=tk.HORIZONTAL,
            resolution=1, variable=self.order_var, length=150
        )
        self.order_scale.pack(side=tk.LEFT, padx=4)

        self.btn_apply = tk.Button(top, text="Apply Filter + Plot", command=self.apply_and_plot)
        self.btn_apply.pack(side=tk.LEFT, padx=10)

        self.btn_save_wav = tk.Button(top, text="Save Filtered WAV…", command=self.save_filtered)
        self.btn_save_wav.pack(side=tk.LEFT, padx=4)

        # NEW: Save the entire 4-panel figure
        self.btn_save_fig = tk.Button(top, text="Save Graph…", command=self.save_figure)
        self.btn_save_fig.pack(side=tk.LEFT, padx=10)

        self.status = tk.Label(self, text="Load an audio file to begin.", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        self.fig = plt.Figure(figsize=(11.5, 7.5), dpi=100)
        self.ax_time_before = self.fig.add_subplot(2, 2, 1)
        self.ax_time_after  = self.fig.add_subplot(2, 2, 2)
        self.ax_spec_before = self.fig.add_subplot(2, 2, 3)
        self.ax_spec_after  = self.fig.add_subplot(2, 2, 4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    # ---------- File load ----------
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio", "*.mp3 *.m4a *.wav *.flac *.ogg *.aac *.wma"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return
        try:
            y, sr = load_audio_ffmpeg(path, target_sr=None, mono=True)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not open file.\n\n{e}")
            return

        self.y, self.sr, self.in_path = y, sr, path
        self.y_filt = None

        nyq = 0.5 * sr
        max_cut = max(300.0, nyq - 100.0)
        self.cutoff_scale.configure(to=float(max_cut))
        if self.cutoff_var.get() >= max_cut:
            self.cutoff_var.set(float(max_cut))

        base = os.path.basename(path)
        self.status.config(text=f"Loaded: {base}  •  sr={sr} Hz  •  duration={len(y)/sr:.2f}s")
        self.quick_plot_before()

    def _maybe_clip_cutoff(self):
        if self.sr is None:
            return
        nyq = 0.5 * self.sr
        if self.cutoff_var.get() >= nyq:
            self.cutoff_var.set(nyq - 1.0)

    # Plotting graph
    def _remove_colorbars(self):
        if self._cb_before is not None:
            try: self._cb_before.remove()
            except Exception: pass
            self._cb_before = None
        if self._cb_after is not None:
            try: self._cb_after.remove()
            except Exception: pass
            self._cb_after = None

    def clear_axes(self):
        for ax in [self.ax_time_before, self.ax_time_after, self.ax_spec_before, self.ax_spec_after]:
            ax.clear()
        self._remove_colorbars()

    def quick_plot_before(self):
        if self.y is None:
            return
        self.clear_axes()

        # Time (before)
        t_b, y_b = time_axis_for_plot(self.y, self.sr)
        self.ax_time_before.plot(t_b, y_b)
        self.ax_time_before.set_title("Time (Before)")
        self.ax_time_before.set_xlabel("Time (s)")
        self.ax_time_before.set_ylabel("Amplitude")
        self.ax_time_before.grid(True, alpha=0.3)

        # Spec (before)
        S_db_b = compute_spectrogram(self.y, self.sr)
        img1 = ldisplay.specshow(S_db_b, sr=self.sr, x_axis="time", y_axis="hz",
                                 ax=self.ax_spec_before, cmap="magma")
        self.ax_spec_before.set_title("Spectrogram (Before)")
        self._cb_before = self.fig.colorbar(img1, ax=self.ax_spec_before, format="%+2.0f dB")
        self._cb_before.set_label("Magnitude (dB)")

        # Placeholders (after)
        self.ax_time_after.set_title("Time (After) — apply filter")
        self.ax_time_after.set_xlabel("Time (s)")
        self.ax_time_after.set_ylabel("Amplitude")
        self.ax_time_after.grid(True, alpha=0.3)

        self.ax_spec_after.set_title("Spectrogram (After) — apply filter")

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def apply_and_plot(self):
        if self.y is None:
            messagebox.showinfo("No file", "Please open an audio file first.")
            return
        cutoff = float(self.cutoff_var.get())
        order  = int(self.order_var.get())
        try:
            self.y_filt = butter_lowpass_filter(self.y, self.sr, cutoff_hz=cutoff, order=order)
        except Exception as e:
            messagebox.showerror("Filter error", str(e))
            return

        self.clear_axes()

        # Time before
        t_b, y_b = time_axis_for_plot(self.y, self.sr)
        self.ax_time_before.plot(t_b, y_b)
        self.ax_time_before.set_title("Time (Before)")
        self.ax_time_before.set_xlabel("Time (s)")
        self.ax_time_before.set_ylabel("Amplitude")
        self.ax_time_before.grid(True, alpha=0.3)

        # Time after
        t_a, y_a = time_axis_for_plot(self.y_filt, self.sr)
        self.ax_time_after.plot(t_a, y_a)
        self.ax_time_after.set_title(f"Time (After LPF)  cutoff={cutoff:.0f} Hz, order={order}")
        self.ax_time_after.set_xlabel("Time (s)")
        self.ax_time_after.set_ylabel("Amplitude")
        self.ax_time_after.grid(True, alpha=0.3)

        # Spec before
        S_db_b = compute_spectrogram(self.y, self.sr)
        img1 = ldisplay.specshow(S_db_b, sr=self.sr, x_axis="time", y_axis="hz",
                                 ax=self.ax_spec_before, cmap="magma")
        self.ax_spec_before.set_title("Spectrogram (Before)")
        self._cb_before = self.fig.colorbar(img1, ax=self.ax_spec_before, format="%+2.0f dB")
        self._cb_before.set_label("Magnitude (dB)")

        # Spec after
        S_db_a = compute_spectrogram(self.y_filt, self.sr)
        img2 = ldisplay.specshow(S_db_a, sr=self.sr, x_axis="time", y_axis="hz",
                                 ax=self.ax_spec_after, cmap="magma")
        self.ax_spec_after.set_title("Spectrogram (After)")
        self._cb_after = self.fig.colorbar(img2, ax=self.ax_spec_after, format="%+2.0f dB")
        self._cb_after.set_label("Magnitude (dB)")

        self.fig.tight_layout()
        self.canvas.draw_idle()

        base = os.path.basename(self.in_path) if self.in_path else "audio"
        self.status.config(text=f"Filtered {base}  •  cutoff={cutoff:.0f} Hz, order={order}")

    # Save audio file after LPF
    def save_filtered(self):
        if self.y_filt is None:
            messagebox.showinfo("Nothing to save", "Apply the filter first, then save.")
            return
        base, _ = os.path.splitext(os.path.basename(self.in_path) if self.in_path else "audio")
        default_name = f"{base}_lpf.wav"
        out_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[("WAV", "*.wav"), ("All files", "*.*")]
        )
        if not out_path:
            return
        try:
            sf.write(out_path, self.y_filt, self.sr)
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save file.\n\n{e}")
            return
        messagebox.showinfo("Saved", f"Filtered audio saved:\n{out_path}")
        self.status.config(text=f"Saved: {out_path}")

    # NEW: Save the 4-panel figure
    def save_figure(self):
        if self.y is None:
            messagebox.showinfo("No graph", "Load an audio file and plot first.")
            return

        # suggest a name based on input + state
        base = "plot"
        if self.in_path:
            stem = os.path.splitext(os.path.basename(self.in_path))[0]
            base = stem
        suffix = "_lpf" if self.y_filt is not None else "_raw"
        default_name = f"{base}{suffix}.png"

        out_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("PDF Document", "*.pdf"),
                ("SVG Vector", "*.svg"),
                ("All files", "*.*"),
            ]
        )
        if not out_path:
            return

        try:
            self.fig.savefig(out_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Graph saved:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save figure.\n\n{e}")

if __name__ == "__main__":
    app = LPFGui()
    app.mainloop()
