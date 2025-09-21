import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import librosa
import librosa.display as ldisplay

# FFmpeg audio loader
def _probe_audio_sr(path):
    
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate",
             "-of", "default=nw=1:nk=1", path],
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
        raise RuntimeError("FFmpeg not found. Install it and ensure it’s on PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed to decode:\n{e.stderr.decode('utf-8', errors='ignore')}")

    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise RuntimeError("Decoded audio is empty.")
    
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    return y, target_sr

# STFT
def compute_spectrogram_db(y, sr, n_fft=2048, hop_length=512, window="hann"):
    
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    return S_db

# Auido picker GUI
class STFTViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STFT Spectrogram Viewer")
        self.geometry("1000x700")

        self.y = None
        self.sr = None
        self.in_path = None

        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Button(top, text="Open Audio…", command=self.open_file).pack(side=tk.LEFT, padx=4)

        # Controls for STFT parameters
        tk.Label(top, text="n_fft").pack(side=tk.LEFT, padx=(12,2))
        self.nfft_var = tk.IntVar(value=2048)
        tk.Spinbox(top, from_=256, to=16384, increment=256, textvariable=self.nfft_var, width=6).pack(side=tk.LEFT)

        tk.Label(top, text="hop").pack(side=tk.LEFT, padx=(12,2))
        self.hop_var = tk.IntVar(value=512)
        tk.Spinbox(top, from_=64, to=4096, increment=64, textvariable=self.hop_var, width=6).pack(side=tk.LEFT)

        tk.Label(top, text="window").pack(side=tk.LEFT, padx=(12,2))
        self.win_var = tk.StringVar(value="hann")
        tk.OptionMenu(top, self.win_var, "hann", "hamming", "blackman", "bartlett").pack(side=tk.LEFT)

        tk.Button(top, text="Plot Spectrogram", command=self.plot_spec).pack(side=tk.LEFT, padx=10)

        self.status = tk.Label(self, text="Pick an audio file to begin.", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        # Figure
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            y, sr = load_audio_ffmpeg(path, target_sr=None, mono=True)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        self.y, self.sr, self.in_path = y, sr, path
        base = os.path.basename(path)
        self.status.config(text=f"Loaded: {base}  •  sr={sr} Hz  •  duration={len(y)/sr:.2f}s")
        
        self.plot_spec(auto=True)

    def plot_spec(self, auto=False):
        if self.y is None:
            if not auto:
                messagebox.showinfo("No file", "Open an audio file first.")
            return
        n_fft = int(self.nfft_var.get())
        hop = int(self.hop_var.get())
        window = self.win_var.get()

        try:
            S_db = compute_spectrogram_db(self.y, self.sr, n_fft=n_fft, hop_length=hop, window=window)
        except Exception as e:
            messagebox.showerror("STFT error", str(e))
            return

        self.ax.clear()
        img = ldisplay.specshow(S_db, sr=self.sr, hop_length=hop, x_axis="time", y_axis="hz", ax=self.ax, cmap="magma")
        self.ax.set_title(f"STFT Spectrogram  •  n_fft={n_fft}, hop={hop}, window={window}")
        cbar = self.fig.colorbar(img, ax=self.ax, format="%+2.0f dB")
        cbar.set_label("Magnitude (dB)")
        self.fig.tight_layout()
        self.canvas.draw_idle()

if __name__ == "__main__":
    app = STFTViewer()
    app.mainloop()
