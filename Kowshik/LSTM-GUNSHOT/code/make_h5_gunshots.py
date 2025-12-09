import os
import h5py
import numpy as np
import soundfile as sf

ROOT_DIR = "gunshots_training"

# Explicit class list
LABEL_NAMES = [
    "BoltAction22",
    "Colt1911",
    "Glock9",
    "Glock45",
    "HKUSP",
    "Kimber45",
    "Lorcin380",
    "M16",
    "MP40",
    "Remington700",
    "Ruger22",
    "Ruger357",
    "Sig9",
    "Smith&Wesson22",
    "Smith&Wesson38special",
    "SportKing22",
    "WASR-10",
    "WinchesterM14",
]

OUT_H5 = os.path.join("workspace", "feature", "gunshots_18class.h5")


def discover_files(root_dir, label_names):
    """
    Returns:
      files: list of file paths
      targets: integer labels (0..17)
      label_names: same list passed in
    """
    files = []
    labels = []

    for label_idx, label_name in enumerate(label_names):
        class_dir = os.path.join(root_dir, label_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Expected folder not found: {class_dir}")

        for fname in os.listdir(class_dir):
            # Only keep .wav (case-insensitive)
            if fname.lower().endswith(".wav"):
                fpath = os.path.join(class_dir, fname)
                files.append(fpath)
                labels.append(label_idx)

    return files, np.array(labels, dtype=np.int64), label_names


def main():
    files, targets, label_names = discover_files(ROOT_DIR, LABEL_NAMES)
    print(f"Found {len(files)} audio files in {len(label_names)} classes.")
    for idx, name in enumerate(label_names):
        print(f"Label {idx:02d}: {name}")

    # vlen int16 to store variable length waveforms
    dt_wave = h5py.vlen_dtype(np.int16)
    
    out_dir = os.path.dirname(OUT_H5)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(OUT_H5, "w") as f:
        dset_wave = f.create_dataset("waveform", shape=(len(files),), dtype=dt_wave)
        dset_tgt = f.create_dataset("target", data=targets)

        # store filepaths and label names as UTF-8 strings
        dt_str = h5py.string_dtype(encoding="utf-8")
        dset_path = f.create_dataset("filepath", shape=(len(files),), dtype=dt_str)
        dset_label_names = f.create_dataset(
            "label_names", shape=(len(label_names),), dtype=dt_str
        )
        dset_label_names[...] = label_names

        sample_rates = []

        for i, fpath in enumerate(files):
            audio, sr = sf.read(fpath)

            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Convert to int16
            if audio.dtype != np.int16:
                if np.issubdtype(audio.dtype, np.floating):
                    # normalize and scale to int16 range
                    maxval = np.max(np.abs(audio))
                    if maxval < 1e-9:
                        audio = np.zeros_like(audio, dtype=np.int16)
                    else:
                        audio = (audio / maxval * 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)

            dset_wave[i] = audio
            dset_path[i] = fpath
            sample_rates.append(sr)

            if (i + 1) % 50 == 0 or i == len(files) - 1:
                print(f"Processed {i + 1}/{len(files)} files")

        f.attrs["sample_rates"] = np.array(sample_rates, dtype=np.int32)

        print("Done! Saved HDF5 to", OUT_H5)


if __name__ == "__main__":
    main()