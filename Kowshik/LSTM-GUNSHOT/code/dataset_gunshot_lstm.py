import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa

class GunshotLSTMDataset(Dataset):
    def __init__(self, hdf5_path, indexes, sr=32000, n_mels=64,
                 win_length=1024, hop_length=512, fmin=50, fmax=16000, augment=False, max_frames=None):
        self.hdf5_path = hdf5_path
        self.indexes = indexes
        self.sr = sr
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.augment = augment
        self.max_frames = max_frames

        with h5py.File(self.hdf5_path, "r") as f:
            self.targets = f["target"][:]
            self.sample_rates = f.attrs["sample_rates"]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        idx_in_h5 = self.indexes[idx]
        with h5py.File(self.hdf5_path, "r") as f:
            wav = f["waveform"][idx_in_h5]  # vlen int16

        # Convert to float32
        wav = wav.astype(np.float32) / 32768.0

        # Resample if needed
        sr_orig = int(self.sample_rates[idx_in_h5])
        if sr_orig != self.sr:
            wav = librosa.resample(wav, orig_sr=sr_orig, target_sr=self.sr)

        # Compute log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sr,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )
        logmel = librosa.power_to_db(mel_spec, ref=np.max)  # [n_mels, T]
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
        logmel = logmel.T  # [T, n_mels]

        # ---- Crop / pad to fixed max_frames (for speed & stability) ----
        T, F = logmel.shape
        if self.max_frames is not None:
            if T > self.max_frames:
                if self.augment:
                    # random crop for training
                    start = np.random.randint(0, T - self.max_frames + 1)
                else:
                    # center crop for validation
                    start = (T - self.max_frames) // 2
                logmel = logmel[start : start + self.max_frames, :]
                T = self.max_frames
            elif T < self.max_frames:
                pad = self.max_frames - T
                pad_left = pad // 2
                pad_right = pad - pad_left
                logmel = np.pad(
                    logmel, ((pad_left, pad_right), (0, 0)),
                    mode="constant"
                )
                T = self.max_frames

        # ---- Simple SpecAugment-style augmentation on training only ----
        if self.augment:
            # random time shift
            shift = np.random.randint(-max(1, T // 10), max(1, T // 10))
            logmel = np.roll(logmel, shift, axis=0)

            # random time mask
            if np.random.rand() < 0.5:
                t0 = np.random.randint(0, T)
                width = np.random.randint(1, max(2, T // 10))
                t1 = min(T, t0 + width)
                logmel[t0:t1, :] = 0.0

            # random frequency mask
            if np.random.rand() < 0.5:
                f0 = np.random.randint(0, F)
                width = np.random.randint(1, max(2, F // 8))
                f1 = min(F, f0 + width)
                logmel[:, f0:f1] = 0.0

        x = torch.from_numpy(logmel).float()
        y = int(self.targets[idx_in_h5])

        return x, y
