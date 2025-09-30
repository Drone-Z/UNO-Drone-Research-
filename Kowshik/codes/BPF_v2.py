import os
import shutil
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from scipy.signal import butter, sosfiltfilt, stft

# Graphs embedded in TK bundle
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    import soundfile as sf
except Exception:
    sf = None


#Function to check is ffmpeg and ffprobe is installed in my environment
def _have_ffmpeg():   
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

#Function to find actual sample rate of audio file
def _probe_audio_sr(path):  
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate", "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, check=True
        )
        sr = int(out.stdout.strip())
        return sr if sr > 0 else 16000
    except Exception:
        return 16000

#Decode any audio to float32 using ffmpeg.
def load_audio_any(path, target_sr=None, mono=True):
    
    if not _have_ffmpeg():
        raise RuntimeError("FFmpeg/FFprobe not found on PATH. Please install and add to PATH.")
    if target_sr is None:
        target_sr = _probe_audio_sr(path)

    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1" if mono else "2",
        "-ar", str(target_sr),
        "pipe:1"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg failed to decode file:\n{msg}") from e

    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if not mono:
        y = y.reshape(-1, 2)
    return y, target_sr


#Butterworth filter for low, high and band filtering
def design_filter(low_hz, high_hz, fs, order=6):
    
    nyq = fs / 2.0
    low = 0.0 if low_hz is None else float(low_hz)
    high = nyq if high_hz is None else float(high_hz)

    if low < 0: low = 0.0
    if high > nyq: high = nyq
    if low == 0.0 and high <= 0.0:
        return None, "Cutoff(s) invalid. Enter a positive high frequency."
    if low >= high:
        return None, "Low cutoff must be less than high cutoff."

    if low <= 0.0 and high < nyq:
        wn = high / nyq
        sos = butter(order, wn, btype="low", output="sos")
        mode = f"Low-pass (≤ {high:.1f} Hz)"
    elif low > 0.0 and high >= nyq:
        wn = low / nyq
        sos = butter(order, wn, btype="high", output="sos")
        mode = f"High-pass (≥ {low:.1f} Hz)"
    elif 0.0 < low < high < nyq:
        wn = [low / nyq, high / nyq]
        sos = butter(order, wn, btype="band", output="sos")
        mode = f"Band-pass ({low:.1f}–{high:.1f} Hz)"
    else:
        return None, "Provided cutoffs are out of valid range."
    return (sos, mode), None

def apply_filter(y, sos):
    """Zero-phase SOS filtering (filtfilt)."""
    if y.ndim == 1:
        return sosfiltfilt(sos, y).astype(np.float32)
    left = sosfiltfilt(sos, y[:, 0])
    right = sosfiltfilt(sos, y[:, 1])
    return np.column_stack([left, right]).astype(np.float32)

#STFT graph
def _mag2db(mag, eps=1e-10):
    return 20.0 * np.log10(np.maximum(np.abs(mag), eps))

def _downsample_for_plot(y, sr, max_points=250_000):
    n = len(y)
    if n <= max_points:
        t = np.arange(n) / sr
        return t, y
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step, dtype=int)
    return idx / sr, y[idx]

def compute_stft_db(y, sr, n_fft, hop, window="hann", fmax_show=None):
    nperseg = int(max(16, min(n_fft, len(y))))
    noverlap = max(0, nperseg - int(hop))
    f, t, Z = stft(y, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap, boundary=None)
    S_db = _mag2db(Z)
    if fmax_show is not None and fmax_show > 0:
        mask = f <= min(fmax_show, f.max() if f.size else fmax_show)
        f = f[mask]
        S_db = S_db[mask, :]
    return f, t, S_db

def robust_vmin_vmax(*arrays, lo=1.0, hi=99.0):
    vals = np.concatenate([a.ravel() for a in arrays if a.size > 0])
    vmin = float(np.percentile(vals, lo))
    vmax = float(np.percentile(vals, hi))
    if vmin >= vmax:
        vmin = float(vals.min())
        vmax = float(vals.max())
    return vmin, vmax


#Auto SR,n_fft and hop counting
def _parse_float_var(var):
    s = var.get().strip()
    return float(s) if s else None

def _parse_int_var(var):
    s = var.get().strip()
    return int(float(s)) if s else None

def _suggest_params(sr, low_hz, high_hz):
    
    import math
    nyq = sr / 2.0

    # n_fft
    target_win_sec = 0.0464  # ~46 ms
    nf_raw = max(256, min(8192, int(sr * target_win_sec)))
    n_fft = 1 << max(8, min(13, int(round(math.log2(nf_raw)))))

    # hop
    hop = max(1, n_fft // 4)

    lo = 0.0 if low_hz is None else float(low_hz)
    hi = nyq if high_hz is None else float(high_hz)
    bw = max(1.0, hi - lo)
    rel = bw / max(1.0, nyq)
    if rel < 0.05:
        order = 10
    elif rel < 0.15:
        order = 8
    else:
        order = 6
    if order % 2 == 1:
        order += 1
    order = min(10, max(4, order))

    fmax_show = min(8000.0, nyq)
    return n_fft, hop, order, fmax_show


#Saving audio files
def save_audio_wav(path, y, sr):
    y = np.asarray(y)
    y = np.clip(y, -1.0, 1.0)
    if sf is not None:
        sf.write(path, y, sr, subtype="PCM_16")
    else:
        from scipy.io.wavfile import write as wavwrite
        y16 = np.int16(y * 32767.0)
        wavwrite(path, sr, y16)


#GUI
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Filter — 4-in-1 Embedded View")
        self.geometry("1200x820")

        self._last_filtered = None
        self._last_sr = None
        self._last_filename = ""
        self._orig_limits = {}   
        self._scroll_cid = None  

        ctrl = tk.Frame(self, padx=6, pady=6)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        self.audio_path = tk.StringVar(value="")
        self.low_hz   = tk.StringVar(value="300")
        self.high_hz  = tk.StringVar(value="3000")
        self.sr       = tk.StringVar(value="")    
        self.nfft     = tk.StringVar(value="")    
        self.hop      = tk.StringVar(value="")    
        self.order    = tk.StringVar(value="")    
        self.fmaxshow = tk.StringVar(value="")    

        tk.Label(ctrl, text="Audio file:").grid(row=0, column=0, sticky="e")
        tk.Entry(ctrl, textvariable=self.audio_path, width=70).grid(row=0, column=1, sticky="we", columnspan=4, padx=(4,8))
        tk.Button(ctrl, text="Browse…", command=self.browse).grid(row=0, column=5, padx=2)
        tk.Button(ctrl, text="Process & Plot", command=self.run).grid(row=0, column=6, padx=2)
        self.btn_save = tk.Button(ctrl, text="Save filtered WAV…", command=self.save, state=tk.DISABLED)
        self.btn_save.grid(row=0, column=7, padx=2)

        # Row 1: numeric params
        tk.Label(ctrl, text="Low (Hz):").grid(row=1, column=0, sticky="e", pady=4)
        tk.Entry(ctrl, textvariable=self.low_hz, width=8).grid(row=1, column=1, sticky="w")

        tk.Label(ctrl, text="High (Hz):").grid(row=1, column=2, sticky="e")
        tk.Entry(ctrl, textvariable=self.high_hz, width=8).grid(row=1, column=3, sticky="w")

        tk.Label(ctrl, text="Target SR:").grid(row=1, column=4, sticky="e")
        tk.Entry(ctrl, textvariable=self.sr, width=8).grid(row=1, column=5, sticky="w")

        tk.Label(ctrl, text="n_fft:").grid(row=1, column=6, sticky="e")
        tk.Entry(ctrl, textvariable=self.nfft, width=7).grid(row=1, column=7, sticky="w")

        tk.Label(ctrl, text="hop:").grid(row=1, column=8, sticky="e")
        tk.Entry(ctrl, textvariable=self.hop, width=7).grid(row=1, column=9, sticky="w")

        tk.Label(ctrl, text="order:").grid(row=1, column=10, sticky="e")
        tk.Entry(ctrl, textvariable=self.order, width=7).grid(row=1, column=11, sticky="w")

        tk.Label(ctrl, text="Max freq show (Hz):").grid(row=1, column=12, sticky="e")
        tk.Entry(ctrl, textvariable=self.fmaxshow, width=10).grid(row=1, column=13, sticky="w")

        tk.Button(ctrl, text="Save Figure…", command=self.save_figure).grid(row=2, column=6, padx=2, pady=4, sticky="we")
        tk.Button(ctrl, text="Save Panels Separately…", command=self.save_panels).grid(row=2, column=7, padx=2, pady=4, sticky="we")
        tk.Button(ctrl, text="Reset View", command=self.reset_view).grid(row=2, column=8, padx=2, pady=4, sticky="we")

        for c in range(14):
            ctrl.grid_columnconfigure(c, weight=0)
        ctrl.grid_columnconfigure(1, weight=1)

        plot_frame = tk.Frame(self)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(10, 6), dpi=110, layout="constrained")
        self.ax_time_before = self.fig.add_subplot(2, 2, 1)
        self.ax_time_after  = self.fig.add_subplot(2, 2, 2)
        self.ax_stft_before = self.fig.add_subplot(2, 2, 3)
        self.ax_stft_after  = self.fig.add_subplot(2, 2, 4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self._cbar = None

        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self._connect_scroll_zoom()

    def _clear_axes(self):
        for ax in (self.ax_time_before, self.ax_time_after, self.ax_stft_before, self.ax_stft_after):
            ax.clear()
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None
        self._orig_limits.clear()

    def _store_orig_limits(self):
        for ax in (self.ax_time_before, self.ax_time_after, self.ax_stft_before, self.ax_stft_after):
            self._orig_limits[ax] = (ax.get_xlim(), ax.get_ylim())

    def _connect_scroll_zoom(self):
        if self._scroll_cid is None:
            self._scroll_cid = self.canvas.mpl_connect('scroll_event', self._on_scroll)

    def _on_scroll(self, event):
        if event.inaxes is None:
            return
        ax = event.inaxes
        scale = 1.2 if event.button == 'up' else (1/1.2)
        xdata = event.xdata
        ydata = event.ydata

        def _zoom_lim(lim_min, lim_max, center, scale):
            rng = (lim_max - lim_min)
            new_rng = rng / scale
            new_min = center - (xdata - lim_min) * (new_rng / rng)
            new_max = new_min + new_rng
            return new_min, new_max

        xmin, xmax = ax.get_xlim()
        if xmax > xmin:
            nxmin, nxmax = _zoom_lim(xmin, xmax, xdata, scale)
            ax.set_xlim(nxmin, nxmax)

        ymin, ymax = ax.get_ylim()
        if ymax > ymin:
            nymin, nymax = _zoom_lim(ymin, ymax, ydata, scale)
            ax.set_ylim(nymin, nymax)

        self.canvas.draw_idle()

    def reset_view(self):
        if not self._orig_limits:
            return
        for ax, (xlim, ylim) in self._orig_limits.items():
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self.canvas.draw_idle()

    def browse(self):
        path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[
                ("Audio", "*.wav *.mp3 *.m4a *.flac *.aac *.ogg *.wma *.aiff *.aif *.au"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        self.audio_path.set(path)

        try:
            sr = _probe_audio_sr(path)
            self.sr.set(str(sr))

            low  = _parse_float_var(self.low_hz)
            high = _parse_float_var(self.high_hz)

            n_fft_s, hop_s, order_s, fmax_s = _suggest_params(sr, low, high)

            if not self.nfft.get().strip():
                self.nfft.set(str(n_fft_s))
            if not self.hop.get().strip():
                self.hop.set(str(hop_s))
            if not self.order.get().strip():
                self.order.set(str(order_s))
            if not self.fmaxshow.get().strip():
                self.fmaxshow.set(str(int(fmax_s)))
        except Exception:
            pass

    def run(self):
        try:
            if not self.audio_path.get():
                messagebox.showwarning("No file", "Please choose an audio file.")
                return

            low  = _parse_float_var(self.low_hz)
            high = _parse_float_var(self.high_hz)
            tgt_sr = _parse_int_var(self.sr)

            y, sr = load_audio_any(self.audio_path.get(), target_sr=tgt_sr, mono=True)
            if y.size == 0:
                raise RuntimeError("Decoded audio is empty.")

            n_fft = _parse_int_var(self.nfft)
            hop   = _parse_int_var(self.hop)
            order = _parse_int_var(self.order)
            fmax_show = _parse_float_var(self.fmaxshow)

            if (n_fft is None) or (hop is None) or (order is None) or (fmax_show is None):
                n_fft_s, hop_s, order_s, fmax_s = _suggest_params(sr, low, high)
                if n_fft is None:
                    n_fft = n_fft_s
                    self.nfft.set(str(n_fft))
                if hop is None:
                    hop = hop_s
                    self.hop.set(str(hop))
                if order is None:
                    order = order_s
                    self.order.set(str(order))
                if fmax_show is None:
                    fmax_show = fmax_s
                    self.fmaxshow.set(str(int(fmax_show)))

            nyq = sr / 2.0
            if low is None:
                low = 0.0
            if high is None:
                high = nyq

            filt, err = design_filter(low, high if high is not None else sr / 2.0, fs=sr, order=order)
            if err:
                messagebox.showerror("Filter error", err)
                return
            (sos, mode) = filt

            y_filt = apply_filter(y, sos)

            self._last_filtered = y_filt
            self._last_sr = sr
            self._last_filename = os.path.basename(self.audio_path.get())
            self.btn_save.config(state=tk.NORMAL)

            self._clear_axes()

            t0, y0 = _downsample_for_plot(y, sr)
            t1, y1 = _downsample_for_plot(y_filt, sr)

            self.ax_time_before.plot(t0, y0, linewidth=0.8)
            self.ax_time_before.set_title("Time — Before")
            self.ax_time_before.set_xlabel("Time (s)")
            self.ax_time_before.set_ylabel("Amplitude")
            self.ax_time_before.grid(True, alpha=0.3)

            self.ax_time_after.plot(t1, y1, linewidth=0.8)
            self.ax_time_after.set_title("Time — After")
            self.ax_time_after.set_xlabel("Time (s)")
            self.ax_time_after.set_ylabel("Amplitude")
            self.ax_time_after.grid(True, alpha=0.3)

            f0, tt0, S0 = compute_stft_db(y,      sr, n_fft=n_fft, hop=hop, window="hann", fmax_show=fmax_show)
            f1, tt1, S1 = compute_stft_db(y_filt, sr, n_fft=n_fft, hop=hop, window="hann", fmax_show=fmax_show)

            vmin, vmax = robust_vmin_vmax(S0, S1, lo=2.0, hi=98.0)

            im0 = self.ax_stft_before.imshow(
                S0, origin="lower", aspect="auto",
                extent=[tt0.min() if tt0.size else 0, tt0.max() if tt0.size else 0,
                        f0.min() if f0.size else 0, f0.max() if f0.size else 0],
                vmin=vmin, vmax=vmax, cmap="magma", interpolation="nearest"
            )
            self.ax_stft_before.set_title("STFT — Before")
            self.ax_stft_before.set_xlabel("Time (s)")
            self.ax_stft_before.set_ylabel("Frequency (Hz)")

            im1 = self.ax_stft_after.imshow(
                S1, origin="lower", aspect="auto",
                extent=[tt1.min() if tt1.size else 0, tt1.max() if tt1.size else 0,
                        f1.min() if f1.size else 0, f1.max() if f1.size else 0],
                vmin=vmin, vmax=vmax, cmap="magma", interpolation="nearest"
            )
            self.ax_stft_after.set_title("STFT — After")
            self.ax_stft_after.set_xlabel("Time (s)")
            self.ax_stft_after.set_ylabel("Frequency (Hz)")

            self._cbar = self.fig.colorbar(im1, ax=[self.ax_stft_before, self.ax_stft_after], location="right", pad=0.02)
            self._cbar.set_label("Magnitude (dB)")

            base = self._last_filename
            self.title(f"Audio Filter — {base} — {mode} • SR={sr} Hz • n_fft={n_fft} • hop={hop} • order={order}")

            self._store_orig_limits()

            self.canvas.draw()

        except ValueError as ve:
            messagebox.showerror("Value error", f"Check your numeric inputs:\n{ve}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    #Save filtered WAV
    def save(self):
        if self._last_filtered is None or self._last_sr is None:
            messagebox.showwarning("Nothing to save", "Please process audio first.")
            return
        default_name = (os.path.splitext(self._last_filename or "output")[0] + "_filtered.wav")
        out_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[("WAV file", "*.wav")]
        )
        if out_path:
            try:
                save_audio_wav(out_path, self._last_filtered, self._last_sr)
                messagebox.showinfo("Saved", f"Filtered audio saved to:\n{out_path}")
            except Exception as e:
                messagebox.showerror("Save error", str(e))

    #Save figure
    def save_figure(self):
        if self.fig is None:
            return
        default_name = (os.path.splitext(self._last_filename or "figure")[0] + "_figure.png")
        out_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG Image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if out_path:
            try:
                self.fig.savefig(out_path, dpi=250, bbox_inches="tight")
                messagebox.showinfo("Saved", f"Figure saved to:\n{out_path}")
            except Exception as e:
                messagebox.showerror("Save error", str(e))

    def save_panels(self):
        if self.fig is None:
            return
        base = os.path.splitext(self._last_filename or "figure")[0]
        out_dir = filedialog.askdirectory(title="Choose folder to save individual panels")
        if not out_dir:
            return
        try:
            panels = [
                (self.ax_time_before, f"{base}_time_before.png"),
                (self.ax_time_after,  f"{base}_time_after.png"),
                (self.ax_stft_before, f"{base}_stft_before.png"),
                (self.ax_stft_after,  f"{base}_stft_after.png"),
            ]

            for ax, name in panels:
                tmp_fig = plt.Figure(figsize=(8, 4.5), dpi=200, layout="constrained")
                tmp_ax = tmp_fig.add_subplot(1, 1, 1)

                for line in ax.get_lines():
                    tmp_ax.plot(line.get_xdata(), line.get_ydata(), linewidth=line.get_linewidth())

                for im in ax.get_images():
                    arr = im.get_array()
                    extent = im.get_extent()
                    tmp_ax.imshow(arr, origin=im.origin, aspect=im.get_aspect(),
                                  extent=extent, vmin=im.get_clim()[0], vmax=im.get_clim()[1],
                                  cmap=im.get_cmap(), interpolation=im.get_interpolation())

                tmp_ax.set_title(ax.get_title())
                tmp_ax.set_xlabel(ax.get_xlabel())
                tmp_ax.set_ylabel(ax.get_ylabel())
                tmp_ax.grid(ax.xaxis._gridOnMajor or False, alpha=0.3)

                tmp_ax.set_xlim(ax.get_xlim())
                tmp_ax.set_ylim(ax.get_ylim())

                tmp_fig.savefig(os.path.join(out_dir, name), dpi=250, bbox_inches="tight")
                plt.close(tmp_fig)

            messagebox.showinfo("Saved", f"Panels saved in:\n{out_dir}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))


if __name__ == "__main__":
    App().mainloop()
