import os
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_gunshot_lstm import GunshotLSTMDataset
from model_bilstm_gunshot import BiLSTMGunshot


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5_path", type=str, default="gunshots_18class.h5")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--sr", type=int, default=32000)
    ap.add_argument("--device", type=str, default="cuda")  # use RTX

    # STFT / mel settings
    ap.add_argument("--win_length", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)  # use this instead of --stride

    # Model size + regularization + early stopping
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    # Debug / plots
    ap.add_argument("--dump_inputs_dir", type=str, default=None)
    ap.add_argument("--plot_dir", type=str, default="workspace/plots_lstm")

    return ap.parse_args()




def get_device(device_str):
    if device_str == "cpu":
        device = torch.device("cpu")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def collate_pad(batch):
    """
    batch: list of (x, y)
    x: [T, F], y: scalar
    returns: padded_x [B, T_max, F], y [B], lengths [B]
    """
    xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    max_len = max(lengths)
    feat_dim = xs[0].shape[1]

    padded = torch.zeros(len(xs), max_len, feat_dim, dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[0]
        padded[i, :T, :] = x

    ys = torch.tensor(ys, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded, ys, lengths


def dump_debug_batch(xb, yb, label_names, out_dir, max_plots=None):
    """
    Dump first batch's inputs as time-frequency spectrogram images (log-mel).
    xb: [B, T, F] on CPU
    yb: [B] on CPU
    label_names: list of class names (index -> name)
    """
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    xb = xb.numpy()
    yb = yb.numpy()

    B, T, F = xb.shape
    num_to_plot = B if max_plots is None else min(B, max_plots)

    for i in range(num_to_plot):
        spec = xb[i]  # [T, F]
        label_idx = int(yb[i])
        class_name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)

        plt.figure(figsize=(4, 3))
        # Transpose to [F, T] so frequency is vertical, time horizontal
        plt.imshow(spec.T, aspect="auto", origin="lower")
        plt.colorbar(label="normalized log-mel")
        plt.title(f"Class: {class_name} (idx={label_idx})")
        plt.xlabel("Time frames")
        plt.ylabel("Mel bins")
        fname = f"epoch1_batch0_item_{i:03d}_{class_name}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

    print(f"Dumped first batch spectrograms to: {out_dir}")


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, out_dir):
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_vs_epoch.png"))
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_accs, label="Train acc")
    plt.plot(epochs, val_accs, label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_vs_epoch.png"))
    plt.close()

    print(f"Saved training curves to {out_dir}")


def eval_confusion_tsne(model, val_loader, device, label_names, out_dir):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from sklearn.manifold import TSNE

    os.makedirs(out_dir, exist_ok=True)

    all_true = []
    all_pred = []
    all_feats = []

    model.eval()
    with torch.no_grad():
        for xb, yb, lengths in tqdm(val_loader, desc="Eval for CM & t-SNE"):
            xb = xb.to(device)
            yb = yb.to(device)
            lengths = lengths.to(device)

            feats, logits = model(xb, lengths, return_features=True)
            preds = logits.argmax(dim=1)

            all_true.append(yb.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_feats.append(feats.cpu().numpy())

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    feats = np.concatenate(all_feats, axis=0)  # [N, feat_dim]

    num_classes = len(label_names)
    
    # ---- Confusion matrix with counts and percentages ----
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # ---- row-normalize to get percentages ----
    cm_sum = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.zeros_like(cm, dtype=float)
    nonzero_rows = cm_sum.squeeze() != 0
    cm_norm[nonzero_rows] = cm[nonzero_rows] / cm_sum[nonzero_rows]

    plt.figure(figsize=(10, 8))

    # use normalized matrix for colors (0–1)
    im = plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, label="Proportion (row-normalized)")

    plt.title("Confusion Matrix (Validation)")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, label_names, rotation=90)
    plt.yticks(tick_marks, label_names)

    # ---- add BOTH count and % text on each non-zero cell ----
    thresh = cm_norm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            count = cm[i, j]
            if count == 0:
                continue  # skip zeros to keep it readable
            pct = cm_norm[i, j] * 100.0
            text = f"{count}\n{pct:.1f}%"
            plt.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=7,
            )

    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # ---- t-SNE visualization ----
    print("Running t-SNE on validation embeddings (this may take a while)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
    emb_2d = tsne.fit_transform(feats)  # [N, 2]

    plt.figure(figsize=(8, 6))
    for class_idx in range(num_classes):
        mask = (y_true == class_idx)
        if not np.any(mask):
            continue
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=5, label=label_names[class_idx])
    plt.title("t-SNE of BiLSTM embeddings (validation)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(markerscale=3, fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    tsne_path = os.path.join(out_dir, "tsne_embeddings.png")
    plt.savefig(tsne_path)
    plt.close()
    print(f"Saved t-SNE plot to {tsne_path}")


def main():
    args = parse_args()
    device = get_device(args.device)

    # ---- load HDF5 size and label names ----
    with h5py.File(args.hdf5_path, "r") as f:
        num_samples = f["target"].shape[0]
        if "label_names" in f:
            label_names_raw = f["label_names"][...]
            label_names = [
                (ln.decode("utf-8") if isinstance(ln, bytes) else str(ln))
                for ln in label_names_raw
            ]
        else:
            # fallback
            label_names = [str(i) for i in range(18)]

    all_idx = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(all_idx)
    split = int(0.8 * len(all_idx))
    train_idx = all_idx[:split]
    val_idx = all_idx[split:]

    # at sr=16000, hop=256, 300 frames ≈ 4.8 seconds
    max_frames = 300

    train_ds = GunshotLSTMDataset(
        args.hdf5_path, train_idx,
        sr=args.sr,
        n_mels=args.n_mels,
        win_length=args.win_length,
        hop_length=args.hop_length,
        fmax=args.sr // 2,
        augment=True,
        max_frames=max_frames,
    )
    val_ds = GunshotLSTMDataset(
        args.hdf5_path, val_idx,
        sr=args.sr,
        n_mels=args.n_mels,
        win_length=args.win_length,
        hop_length=args.hop_length,
        fmax=args.sr // 2,
        augment=False,
        max_frames=max_frames,
    )

    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=pin_mem, collate_fn=collate_pad
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=pin_mem, collate_fn=collate_pad
    )

    input_dim = args.n_mels
    num_classes = len(label_names)

    model = BiLSTMGunshot(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
    )


    # For plotting
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    # ---- Early stopping setup ----
    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt_path = os.path.join("checkpoints", "bilstm_gunshot_18class_best.pt")
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    dumped_inputs = False  # to only dump first epoch's first batch once

    for ep in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_idx, (xb, yb, lengths) in enumerate(
            tqdm(train_loader, desc=f"Train Epoch {ep}", disable=True)
        ):
            if (ep == 1) and (batch_idx == 0) and args.dump_inputs_dir:
                # Dump this first batch in spectrogram domain with class names
                dump_debug_batch(xb.cpu(), yb.cpu(), label_names, args.dump_inputs_dir)
                dumped_inputs = True

            xb = xb.to(device)
            yb = yb.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits = model(xb, lengths)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- Validation ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb, lengths in tqdm(val_loader, desc=f"Val Epoch {ep}", disable=True):
                xb = xb.to(device)
                yb = yb.to(device)
                lengths = lengths.to(device)

                logits = model(xb, lengths)
                loss = criterion(logits, yb)

                val_loss += loss.item() * yb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {ep:03d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )
        # ---- Early stopping check ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> New best model saved with Val Acc = {best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(
                    f"Early stopping triggered at epoch {ep}. "
                    f"Best Val Acc = {best_val_acc:.4f}"
                )
                break

    # Save last-epoch model too
    last_path = os.path.join("checkpoints", "bilstm_gunshot_18class_last.pt")
    torch.save(model.state_dict(), last_path)
    print("Saved last-epoch model to", last_path)

    # Load best model (based on val accuracy) before evaluation
    best_ckpt_path = os.path.join("checkpoints", "bilstm_gunshot_18class_best.pt")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        model.to(device)
        print(f"Loaded best model from {best_ckpt_path} for evaluation.")
    else:
        print("WARNING: best checkpoint not found, using last-epoch model for evaluation.")

    # Plots: loss + accuracy
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, args.plot_dir)

    # Confusion matrix + t-SNE
    eval_confusion_tsne(model, val_loader, device, label_names, args.plot_dir)


if __name__ == "__main__":
    main()
