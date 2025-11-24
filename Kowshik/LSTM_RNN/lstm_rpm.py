import os, argparse, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# index to store labels and RPM
RPM2IDX = {}   # rpm (int)  -> class index [0,1,2]
IDX2RPM = []   # class index -> rpm (int)


def _read_one_excel(path):
    """Return (freq, data) where data has shape (N, 4) for M1~M4."""
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    freq_col = df.columns[0]
    spl_cols = [c for c in df.columns[1:] if c.lower().startswith("m") or c.lower().startswith("spl")]
    if len(spl_cols) < 4:
        spl_cols = list(df.columns[1:1+4])  # fallback
    df = df[[freq_col] + spl_cols].dropna().sort_values(freq_col).reset_index(drop=True)
    freq = df[freq_col].values.astype(np.float32)
    data = df[spl_cols[:4]].values.astype(np.float32)  # (N,4)
    return freq, data


def build_dataset(data_dir, window_len=256, stride=64):
    """
    Returns: X_train, y_train, X_val, y_val
    X_* shape: (B, T, C) with C=5 (M1..M4 raw, log10(freq) raw)
    - Each .xlsx in data_dir that has an integer in its filename is treated as one RPM run.
    - That integer (e.g., 3720) becomes the RPM label. We map RPM -> class index internally.
    """
    global RPM2IDX, IDX2RPM
    RPM2IDX = {}
    IDX2RPM = []

    # Discover all Excel files and parse RPM from filename
    all_entries = []   # list of dicts: {'rpm': int, 'freq': (N,), 'data': (N,4)}
    rpm_values = []

    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith(".xlsx"):
            continue
        m = re.search(r"(\d+)", fn)
        if not m:
            # skip files that do not contain an integer (no RPM)
            continue
        rpm = int(m.group(1))
        fpath = os.path.join(data_dir, fn)
        freq, data = _read_one_excel(fpath)
        all_entries.append({"rpm": rpm, "freq": freq.astype(np.float32), "data": data.astype(np.float32)})
        rpm_values.append(rpm)

    if not all_entries:
        raise RuntimeError(f"No usable .xlsx files found in {data_dir}")

    # Define RPM -> class index mapping
    for rpm in sorted(set(rpm_values)):
        if rpm not in RPM2IDX:
            RPM2IDX[rpm] = len(IDX2RPM)
            IDX2RPM.append(rpm)
    
    # Each (frequency, channel) pair becomes its own sequence.
    X_list, y_list = [], []
    for e in all_entries:
        rpm, freq, data = e["rpm"], e["freq"], e["data"]  # data: (N, 4)

        # log-frequency as feature
        flog = np.log10(np.clip(freq, 1e-3, None)).astype(np.float32)  # shape (N,)

        N = len(freq)

        # handle short sequences by padding BOTH SPL and freq
        spl = data.astype(np.float32)        # (N, 4)
        if N < window_len:
            pad = window_len - N
            spl = np.vstack([spl, np.repeat(spl[-1:], pad, axis=0)])
            flog = np.concatenate([flog, np.repeat(flog[-1:], pad)])
            N = window_len

        # compute sliding-window start indices
        if N == window_len:
            starts = [0]
        else:
            starts = list(range(0, N - window_len + 1, stride))

        class_idx = RPM2IDX[rpm]

        # create windows for each channel separately
        num_channels = spl.shape[1]  # should be 4 (M1..M4)
        for ch in range(num_channels):
            feat_ch = np.stack([spl[:, ch], flog], axis=1).astype(np.float32)

            for s in starts:
                X_list.append(feat_ch[s:s + window_len])  # (T, 2)
                y_list.append(class_idx)

    X = np.stack(X_list, axis=0).astype(np.float32)  # (B,T,5)
    y = np.array(y_list, dtype=np.int64)

    # 80/20 split per class (per RPM)
    train_idx, val_idx = [], []
    num_classes = len(IDX2RPM)
    for c in range(num_classes):
        idxs = np.where(y == c)[0]
        if len(idxs) == 0:
            continue
        split = int(0.8 * len(idxs))
        train_idx += idxs[:split].tolist()
        val_idx   += idxs[split:].tolist()

    if not train_idx or not val_idx:
        raise RuntimeError("Train/val split resulted in empty set; try smaller window_len/stride or more data.")

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class BiLSTMClassifier(nn.Module):
    def __init__(self, in_dim=5, hidden=128, layers=2, classes=3, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=(dropout if layers > 1 else 0.0)
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * out_dim, classes)

    def forward(self, x):
        y, _ = self.lstm(x)        # (B,T,2H)
        mean_pool = y.mean(dim=1)  # (B,2H)
        max_pool  = y.max(dim=1).values
        h = torch.cat([mean_pool, max_pool], dim=1)  # (B,4H if bi)
        h = self.drop(h)
        return self.fc(h)


def _plot_sequence(seq_2d, out_png, rpm_label=None):
    """
    seq_2d: numpy array (T, C)
    rpm_label: e.g. 3720, 5190, 7150
    """
    plt.figure(figsize=(7,4))
    T, C = seq_2d.shape
    for c in range(C):
        plt.plot(range(T), seq_2d[:, c], label=f"ch{c+1}")
    plt.xlabel("Time step (frequency bin index)")
    plt.ylabel("Raw feature value")
    title = "Input window"
    if rpm_label is not None:
        title += f" (RPM = {rpm_label})"
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()



def train_one(args):
    # Build dataset with RAW features
    Xtr, Ytr, Xva, Yva = build_dataset(args.data_dir, args.window_len, args.stride)

    # make sure the model can overfit a tiny subset
    DEBUG_OVERFIT = False
    if DEBUG_OVERFIT:
        Xtr, Ytr = Xtr[:64], Ytr[:64]
        Xva, Yva = Xtr, Ytr

    save_dir = os.path.dirname(args.save)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    # make sure confusion-matrix directory exists
    if getattr(args, "cm_dir", None):
        os.makedirs(args.cm_dir, exist_ok=True)

    # device selection using args.device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pin_mem = (device.type == "cuda")
    print("Using device:", device)

    tr_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                          torch.tensor(Ytr, dtype=torch.long))


    tr_ds = TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                          torch.tensor(Ytr, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(Xva, dtype=torch.float32),
                          torch.tensor(Yva, dtype=torch.long))
    tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  pin_memory=pin_mem)
    va = DataLoader(va_ds, batch_size=args.batch, shuffle=False, pin_memory=pin_mem)

    num_classes = len(IDX2RPM)

    model = BiLSTMClassifier(in_dim=Xtr.shape[2], hidden=args.hidden,
                             layers=args.layers, classes=num_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss()
    steps_per_epoch = max(1, len(tr))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch
    )
    
    # make sure curve directory exists & prepare history buffers
    if getattr(args, "curve_dir", None):
        os.makedirs(args.curve_dir, exist_ok=True)
    hist_epoch, hist_tr_loss, hist_val_acc = [], [], []

    best_acc, best_state = 0.0, None
    for ep in range(1, args.epochs+1):
        model.train()

        # per-epoch dump setup
        dump_all_batches = bool(args.dump_inputs_dir and args.dump_plots and (ep == 1 or ep == args.epochs))
        dump_count = 0

        total_loss, total = 0.0, 0

        for batch_idx, (xb_cpu, yb_cpu) in enumerate(tr):
            xb = xb_cpu.to(device)
            yb = yb_cpu.to(device)

            # ---- DUMP INPUTS (PNG only) ----
            dump_this_epoch = (ep == 1)
            if args.dump_inputs_dir and args.dump_plots and dump_this_epoch:
                epoch_dir = os.path.join(args.dump_inputs_dir, f"epoch_{ep:03d}")
                os.makedirs(epoch_dir, exist_ok=True)

                X_np = xb.detach().cpu().numpy()           # (B, T, C)
                labels_np = yb_cpu.numpy()                 # (B,)
                for i in range(X_np.shape[0]):
                    rpm = IDX2RPM[labels_np[i]]            # convert class index -> actual RPM
                    png_path = os.path.join(
                        epoch_dir,
                        f"batch_{batch_idx:04d}_item_{i:03d}_rpm_{rpm}.png"
                    )
                    _plot_sequence(X_np[i], png_path, rpm_label=rpm)

            # ---- END DUMP ----

            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            scheduler.step()

            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)

        tr_loss = total_loss / max(1, total)

        # validation
        model.eval()
        correct, count = 0, 0
        with torch.no_grad():
            for xb, yb in va:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                count   += yb.size(0)
        val_acc = correct / max(1, count)

        # compute & save CM only on last epoch
        dump_cm_epoch = (ep == args.epochs)
        if dump_cm_epoch:
            all_preds, all_true = [], []
            with torch.no_grad():
                for xb, yb in va:
                    xb = xb.to(device)
                    p = model(xb).argmax(1).cpu().numpy()
                    all_preds.extend(p.tolist())
                    all_true.extend(yb.numpy().tolist())

            labels = [str(rpm) for rpm in IDX2RPM]  # true RPM values for each class index
            K = len(labels)
            cm = np.zeros((K, K), dtype=int)
            for t, p in zip(all_true, all_preds):
                cm[t, p] += 1

            if getattr(args, "cm_dir", None):
                os.makedirs(args.cm_dir, exist_ok=True)
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_xlabel("Predicted RPM")
                ax.set_ylabel("True RPM")
                ax.set_xticks(range(K)); ax.set_yticks(range(K))
                ax.set_xticklabels(labels); ax.set_yticklabels(labels)
                for i in range(K):
                    for j in range(K):
                        ax.text(
                            j, i, str(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=10
                        )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                png_path = os.path.join(args.cm_dir, f"cm_epoch_{ep:03d}_last.png")
                fig.savefig(png_path, dpi=150)
                plt.close(fig)
        # --- end CM section ---

        hist_epoch.append(ep)
        hist_tr_loss.append(tr_loss)
        hist_val_acc.append(val_acc)

        print(f"Epoch {ep:02d}/{args.epochs} - loss={tr_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        torch.save(best_state, args.save)
        print(f"Saved best model to: {args.save} (val_acc={best_acc:.3f})")
        
    # save validation-accuracy curve (PNG only)
    if getattr(args, "curve_dir", None):
        os.makedirs(args.curve_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(hist_epoch, hist_val_acc)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title("Validation Accuracy vs Epoch")
        ax.grid(True, linestyle="--", linewidth=0.5)
        fig.tight_layout()
        png_path = os.path.join(args.curve_dir, "val_acc_vs_epoch.png")
        fig.savefig(png_path, dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",type=str,default="auto",choices=["auto", "cpu", "cuda"],help="Where to run the model")
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--window_len", type=int, default=128,
                    help="#frequency bins per sequence window")
    ap.add_argument("--stride", type=int, default=64,
                    help="hop between adjacent windows (controls overlap)")
    ap.add_argument("--hidden", type=int, default=64,
                    help="LSTM hidden size")
    ap.add_argument("--layers", type=int, default=1,
                    help="number of stacked LSTM layers")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", type=str, default=os.path.join("models", "bilstm_rpm.pth"))
    ap.add_argument("--dump_inputs_dir", type=str, default=None,
                    help="If set, save a few input batches per epoch here (PNG).")
    ap.add_argument("--dump_max_batches", type=int, default=3,
                    help="(unused now, kept for compatibility)")
    ap.add_argument("--dump_plots", action="store_true",
                    help="Also save a PNG plot for items in each dumped batch.")
    ap.add_argument("--cm_dir", type=str,
                    default=os.path.join("metrics", "confusion_matrices"),
                    help="Folder to save last-epoch confusion matrix PNG.")
    ap.add_argument("--curve_dir", type=str,
                    default=os.path.join("metrics", "curves"),
                    help="Folder to save val-acc vs epoch plot.")
    args = ap.parse_args()
    train_one(args)


if __name__ == "__main__":
    main()
