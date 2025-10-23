# Adversarial.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

#Reproducibility & Device

os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Data: standard CIFAR-10 normalization
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2470, 0.2435, 0.2616]

train_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_ds = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_ds  = CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False, num_workers=2, pin_memory=True)


#Simple CNN Model (kept from your original for continuity)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 8 * 8, 512)
        self.fc2   = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B,32,16,16]
        x = self.pool(F.relu(self.conv2(x)))   # [B,64, 8, 8]
        x = x.view(x.size(0), -1)              # [B,4096]
        x = F.relu(self.fc1(x))                # [B,512]
        return self.fc2(x)                     # [B,10] (logits)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


#Train the model (simple training loop)

def train(model, loader, optimizer, criterion, epochs=20):
    model.train()
    for ep in range(epochs):
        running = 0.0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch [{ep+1}/{epochs}] Step [{i+1}/{len(loader)}] "
                      f"Loss: {running/100:.4f}")
                running = 0.0

print("Training...")
train(model, train_loader, optimizer, criterion, epochs=20)
torch.save(model.state_dict(), "cnn_cifar10.pth")
print("Training complete. Weights saved to cnn_cifar10.pth")


#Helpers for normalization-aware attacks
#Convert eps/alpha from pixel space → normalized model space.
#Also compute per-channel bounds in normalized space.

def _to_tensors_on(device, mean, std):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, 3, 1, 1)
    return mean_t, std_t

def _normalized_bounds(mean, std, device):
    mean_t, std_t = _to_tensors_on(device, mean, std)
    lower = (0 - mean_t) / std_t
    upper = (1 - mean_t) / std_t
    return lower, upper, std_t

#FGSM (L_inf) in normalized space


def fgsm_attack_normalized(x, y, model, eps_pix, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    model.eval()
    lower, upper, std_t = _normalized_bounds(mean, std, x.device)

    # Convert pixel-space epsilon to normalized epsilon per-channel
    eps = eps_pix / std_t  # broadcast [1,3,1,1]

    # Require grad on input for ∂loss/∂x
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    grad = x.grad.detach().sign()

    x_adv = x + eps * grad
    x_adv = torch.max(torch.min(x_adv, upper), lower)  # clip to data range
    return x_adv.detach()


#PGD (L_inf) in normalized space

def pgd_linf_normalized(model, x, y, eps_pix=8/255, alpha_pix=2/255, steps=10,
                        mean=CIFAR10_MEAN, std=CIFAR10_STD):
    model.eval()  # important to keep BN/Dropout stable during attack
    lower, upper, std_t = _normalized_bounds(mean, std, x.device)

    # Convert pixel-space eps/alpha to normalized space
    eps   = eps_pix   / std_t
    alpha = alpha_pix / std_t

    x0    = x.detach()
    x_adv = x0.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad = x_adv.grad.detach().sign()

        # Gradient step + projection to L_inf ball
        x_adv = x_adv + alpha * grad
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)

        # Clip to valid normalized range and detach to make a new leaf
        x_adv = torch.max(torch.min(x_adv, upper), lower).detach()

    return x_adv


#Metrics: Clean-correct mask, and per-attack (Acc, Loss, ASR)
#ASR = 1 - robust_accuracy (using ONLY clean-correct samples).

@torch.no_grad()
def _clean_correct_mask(model, x, y):
    model.eval()
    logits = model(x)
    return (logits.argmax(1) == y)  # [B]

def fgsm_metrics(model, loader, device, eps_pix):
    model.eval()
    correct_adv, considered, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mask = _clean_correct_mask(model, x, y)
        if mask.sum().item() == 0:
            continue

        x_sel, y_sel = x[mask], y[mask]
        x_adv = fgsm_attack_normalized(x_sel, y_sel, model, eps_pix)

        with torch.no_grad():
            logits_adv = model(x_adv)
            loss       = F.cross_entropy(logits_adv, y_sel, reduction="sum")
            pred_adv   = logits_adv.argmax(1)
            correct_adv += (pred_adv == y_sel).sum().item()
            considered += y_sel.size(0)
            loss_sum   += loss.item()

    acc = (correct_adv / considered) if considered else 0.0
    avg_loss = (loss_sum / considered) if considered else 0.0
    asr = 1.0 - acc
    return acc, avg_loss, asr, considered

def pgd_metrics(model, loader, device, eps_pix, alpha_pix=2/255, steps=10):
    model.eval()
    correct_adv, considered, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mask = _clean_correct_mask(model, x, y)
        if mask.sum().item() == 0:
            continue

        x_sel, y_sel = x[mask], y[mask]
        x_adv = pgd_linf_normalized(model, x_sel, y_sel, eps_pix, alpha_pix, steps)

        with torch.no_grad():
            logits_adv = model(x_adv)
            loss       = F.cross_entropy(logits_adv, y_sel, reduction="sum")
            pred_adv   = logits_adv.argmax(1)
            correct_adv += (pred_adv == y_sel).sum().item()
            considered += y_sel.size(0)
            loss_sum   += loss.item()

    acc = (correct_adv / considered) if considered else 0.0
    avg_loss = (loss_sum / considered) if considered else 0.0
    asr = 1.0 - acc
    return acc, avg_loss, asr, considered


#Evaluate across epsilons (pixel-space) and collect curves

epsilons = [0/255, 2/255, 4/255, 8/255, 12/255, 16/255]
fgsm_accs, fgsm_losses, fgsm_asrs = [], [], []
pgd_accs,  pgd_losses,  pgd_asrs  = [], [], []

print("\nEvaluating FGSM & PGD robustness (conditional on clean correctness):")
for eps in epsilons:
    fg_acc, fg_loss, fg_asr, fg_n = fgsm_metrics(model, test_loader, device, eps)
    pg_acc, pg_loss, pg_asr, pg_n = pgd_metrics(model,  test_loader, device, eps, alpha_pix=2/255, steps=10)

    fgsm_accs.append(fg_acc); fgsm_losses.append(fg_loss); fgsm_asrs.append(fg_asr)
    pgd_accs.append(pg_acc);  pgd_losses.append(pg_loss);  pgd_asrs.append(pg_asr)

    print(f"  ε={eps:.4f} | FGSM: acc={fg_acc:.3f}, loss={fg_loss:.3f}, ASR={fg_asr:.3f} | "
          f"PGD: acc={pg_acc:.3f}, loss={pg_loss:.3f}, ASR={pg_asr:.3f}")


#Plotting (no seaborn; separate + comparison charts)

def plot_line(xs, ys, title, xlab, ylab, fname, label=None, marker='o'):
    plt.figure(figsize=(8,6))
    plt.plot(xs, ys, marker=marker, label=label if label else title)
    plt.title(title)
    plt.xlabel(xlab); plt.ylabel(ylab)
    plt.grid(True); 
    if label: plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()

# FGSM-only: Accuracy / Loss / ASR
plot_line(epsilons, fgsm_accs,  "FGSM: Accuracy vs Epsilon", "Epsilon (ε)", "Accuracy",             "fgsm_accuracy_vs_epsilon.png")
plot_line(epsilons, fgsm_losses,"FGSM: Loss vs Epsilon",     "Epsilon (ε)", "Cross-Entropy Loss",   "fgsm_loss_vs_epsilon.png")
plot_line(epsilons, fgsm_asrs,  "FGSM: ASR vs Epsilon",      "Epsilon (ε)", "Attack Success Rate",  "fgsm_asr_vs_epsilon.png")

# PGD-only: Accuracy / Loss / ASR
plot_line(epsilons, pgd_accs,   "PGD: Accuracy vs Epsilon",  "Epsilon (ε)", "Accuracy",             "pgd_accuracy_vs_epsilon.png", marker='s')
plot_line(epsilons, pgd_losses, "PGD: Loss vs Epsilon",      "Epsilon (ε)", "Cross-Entropy Loss",   "pgd_loss_vs_epsilon.png", marker='s')
plot_line(epsilons, pgd_asrs,   "PGD: ASR vs Epsilon",       "Epsilon (ε)", "Attack Success Rate",  "pgd_asr_vs_epsilon.png", marker='s')

# Comparisons (FGSM vs PGD)
def plot_compare(xs, y1, y2, title, ylab, fname, label1, label2):
    plt.figure(figsize=(8,6))
    plt.plot(xs, y1, marker='o', label=label1)
    plt.plot(xs, y2, marker='s', label=label2)
    plt.title(title)
    plt.xlabel("Epsilon (ε)"); plt.ylabel(ylab)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()

plot_compare(epsilons, fgsm_accs,  pgd_accs,  "Accuracy vs Epsilon (FGSM vs PGD)", "Accuracy",             "compare_accuracy.png", "FGSM", "PGD")
plot_compare(epsilons, fgsm_losses,pgd_losses,"Loss vs Epsilon (FGSM vs PGD)",     "Cross-Entropy Loss",   "compare_loss.png",     "FGSM", "PGD")
plot_compare(epsilons, fgsm_asrs,  pgd_asrs,  "ASR vs Epsilon (FGSM vs PGD)",      "Attack Success Rate",  "compare_asr.png",      "FGSM", "PGD")

print("\nSaved figures:")
print("  - fgsm_accuracy_vs_epsilon.png")
print("  - fgsm_loss_vs_epsilon.png")
print("  - fgsm_asr_vs_epsilon.png")
print("  - pgd_accuracy_vs_epsilon.png")
print("  - pgd_loss_vs_epsilon.png")
print("  - pgd_asr_vs_epsilon.png")
print("  - compare_accuracy.png")
print("  - compare_loss.png")
print("  - compare_asr.png")
