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


# ---------- FFmpeg audio loader ----------
def _probe_audio_sr(path):
    """Return source sample rate with ffprobe, fallback to 16000 if unknown."""
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
    """Decode any audio file using FFmpeg -> float32, return (y, sr)."""
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

    # Soft-clip normalization if needed
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    return y, target_sr


# ---------- STFT helper ----------
def compute_spectrogram_db(y, sr, n_fft=2048, hop_length=512, window="hann"):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    S_mag = np.abs(S)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    return S_db


# ---------- GUI ----------
class STFTViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STFT Spectrogram Viewer")
        self.geometry("1120x820")

        self.y = None
        self.sr = None
        self.in_path = None
        self._cbar = None      # track colorbar to prevent stacking
        self._orig_limits = {} # for Reset View

        # --- Top controls ---
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Button(top, text="Open Audio…", command=self.open_file).pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="n_fft").pack(side=tk.LEFT, padx=(12, 2))
        self.nfft_var = tk.IntVar(value=2048)
        tk.Spinbox(top, from_=256, to=16384, increment=256, textvariable=self.nfft_var, width=6).pack(side=tk.LEFT)

        tk.Label(top, text="hop").pack(side=tk.LEFT, padx=(12, 2))
        self.hop_var = tk.IntVar(value=512)
        tk.Spinbox(top, from_=64, to=4096, increment=64, textvariable=self.hop_var, width=6).pack(side=tk.LEFT)

        tk.Label(top, text="window").pack(side=tk.LEFT, padx=(12, 2))
        self.win_var = tk.StringVar(value="hann")
        tk.OptionMenu(top, self.win_var, "hann", "hamming", "blackman", "bartlett").pack(side=tk.LEFT)

        tk.Button(top, text="Plot Spectrogram", command=self.plot_spec).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="Save Graph", command=self.save_graph).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="Reset View", command=self._reset_view).pack(side=tk.LEFT, padx=10)

        # --- Pan buttons (optional GUI controls) ---
        pan = tk.Frame(self)
        pan.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

        tk.Label(pan, text="Pan:").pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(pan, text="⟵ Left",  command=lambda: self._pan_axis('x', -0.2)).pack(side=tk.LEFT, padx=3)
        tk.Button(pan, text="⟶ Right", command=lambda: self._pan_axis('x', +0.2)).pack(side=tk.LEFT, padx=3)
        tk.Button(pan, text="⟰ Up",    command=lambda: self._pan_axis('y', +0.2)).pack(side=tk.LEFT, padx=(12,3))
        tk.Button(pan, text="⟱ Down",  command=lambda: self._pan_axis('y', -0.2)).pack(side=tk.LEFT, padx=3)

        # --- Status bar ---
        self.status = tk.Label(self, text="Pick an audio file to begin.", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        # --- Figure / Canvas ---
        self.fig = plt.Figure(figsize=(10.6, 7.2), dpi=100)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Mouse-wheel zoom binding: wheel -> X, Shift+wheel -> Y
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

        # Arrow keys for pan (make sure the window has focus)
        self.bind("<Left>",  lambda e: self._pan_axis('x', -0.2))
        self.bind("<Right>", lambda e: self._pan_axis('x', +0.2))
        self.bind("<Up>",    lambda e: self._pan_axis('y', +0.2))
        self.bind("<Down>",  lambda e: self._pan_axis('y', -0.2))
        self.canvas_widget.focus_set()

    # ---------- Auto-pick n_fft & hop from sample rate ----------
    def _suggest_stft_params(self, sr):
        """
        Heuristic:
        - Target ~20 Hz frequency resolution: n_fft ≈ next_pow2(sr / 20), clamped [512, 16384].
        - Hop ~10 ms (or 20 ms if sr < 22050), snapped to multiples of 64,
          then limited to n_fft//2 (>=50% overlap) and typically n_fft//4 (~75% overlap).
        """
        target_df_hz = 20.0
        raw_nfft = max(256, int(sr / target_df_hz))
        n_fft = 1 << (raw_nfft - 1).bit_length()
        n_fft = int(np.clip(n_fft, 512, 16384))

        hop_ms = 10 if sr >= 22050 else 20
        hop = int(sr * (hop_ms / 1000.0))
        hop = max(64, int(round(hop / 64)) * 64)
        hop = min(hop, max(64, n_fft // 2))
        hop = min(hop, max(64, n_fft // 4))
        return n_fft, hop

    # ----- Actions -----
    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            y, sr = load_audio_ffmpeg(path, target_sr=None, mono=True)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        self.y, self.sr, self.in_path = y, sr, path

        # Auto-fill n_fft and hop based on this file
        n_fft, hop = self._suggest_stft_params(self.sr)
        self.nfft_var.set(int(n_fft))
        self.hop_var.set(int(hop))

        base = os.path.basename(path)
        self.status.config(
            text=f"Loaded: {base}  •  sr={sr} Hz  •  duration={len(y)/sr:.2f}s  •  n_fft={n_fft}  hop={hop}"
        )

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

        # Clear axes and old colorbar
        self.ax.clear()
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None

        img = ldisplay.specshow(
            S_db, sr=self.sr, hop_length=hop, x_axis="time", y_axis="hz", ax=self.ax, cmap="magma"
        )
        self.ax.set_title(f"STFT Spectrogram  •  n_fft={n_fft}, hop={hop}, window={window}")
        self._cbar = self.fig.colorbar(img, ax=self.ax, format="%+2.0f dB")
        self._cbar.set_label("Magnitude (dB)")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        # Store original limits for reset
        self._orig_limits = {
            "xlim": self.ax.get_xlim(),
            "ylim": self.ax.get_ylim(),
        }

    # ---------- Mouse-wheel zoom ----------
    def _on_scroll_zoom(self, event):
        """Mouse wheel zoom: default X (time), hold Shift to zoom Y (frequency)."""
        if event.inaxes != self.ax:
            return

        # Determine wheel step: +1 up, -1 down
        step = getattr(event, "step", None)
        if step is None:
            step = 1 if getattr(event, "button", "up") == "up" else -1

        base = 1.2
        zoom_in = step > 0
        shift_pressed = (event.key is not None) and ("shift" in event.key)

        if shift_pressed:
            self._zoom_axis_at_point(axis='y', center=event.ydata, zoom_in=zoom_in, base=base)
        else:
            self._zoom_axis_at_point(axis='x', center=event.xdata, zoom_in=zoom_in, base=base)

        self.canvas.draw_idle()

    def _zoom_axis_at_point(self, axis='x', center=None, zoom_in=True, base=1.2):
        """Zooms one axis keeping the cursor's data point fixed."""
        if center is None:
            return

        if axis == 'x':
            lo, hi = self.ax.get_xlim()
        else:
            lo, hi = self.ax.get_ylim()

        if zoom_in:
            new_lo = center - (center - lo) / base
            new_hi = center + (hi - center) / base
        else:
            new_lo = center - (center - lo) * base
            new_hi = center + (hi - center) * base

        if axis == 'x':
            self.ax.set_xlim(new_lo, new_hi)
        else:
            self.ax.set_ylim(new_lo, new_hi)

    # ---------- Pan (arrow keys or buttons) ----------
    def _pan_axis(self, axis='x', frac=0.2):
        """Pan view by a fraction of current span (positive shifts right/up)."""
        if self.y is None:
            return

        if axis == 'x':
            lo, hi = self.ax.get_xlim()
        else:
            lo, hi = self.ax.get_ylim()

        span = hi - lo
        shift = span * frac
        new_lo, new_hi = lo + shift, hi + shift

        if axis == 'x':
            self.ax.set_xlim(new_lo, new_hi)
        else:
            self.ax.set_ylim(new_lo, new_hi)

        self.canvas.draw_idle()

    # ---------- Reset ----------
    def _reset_view(self):
        if not self._orig_limits:
            return
        self.ax.set_xlim(*self._orig_limits["xlim"])
        self.ax.set_ylim(*self._orig_limits["ylim"])
        self.canvas.draw_idle()

    # ---------- Save ----------
    def save_graph(self):
        """Save the currently displayed spectrogram to an image/PDF file."""
        if self.y is None:
            messagebox.showinfo("No graph", "Please load an audio file and plot the spectrogram first.")
            return

        default_name = "spectrogram"
        if self.in_path:
            stem = os.path.splitext(os.path.basename(self.in_path))[0]
            default_name = f"{stem}_spec_nfft{self.nfft_var.get()}_hop{self.hop_var.get()}"

        file_path = filedialog.asksaveasfilename(
            initialfile=f"{default_name}.png",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg"),
                ("PDF Document", "*.pdf"),
                ("SVG Vector", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            self.fig.savefig(file_path, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Graph saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


if __name__ == "__main__":
    app = STFTViewer()
    app.mainloop()
