import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

# Force CUDA error visibility (optional for debugging)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CIFAR-10 transforms
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load datasets
train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the model (1 epoch for demo)
print("Training model...")
model.train()
for epoch in range(1):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # # visualization of some training images
        # images[0].cpu().numpy()
        # plt.imshow(np.transpose(images[0].cpu().numpy(), (1, 2, 0)))
        # plt.title(f"Label: {CIFAR10_CLASSES[labels[0]]}")
        # plt.show()
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print("Training complete.")

# FGSM attack function
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed = image + epsilon * sign_data_grad
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed

# Test FGSM
def test(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []

    model.eval()

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
    

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]
        if init_pred.item() != target.item():
            continue

        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1

        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    print(f"Original: {CIFAR10_CLASSES[init_pred.item()]}, Adversarial: {CIFAR10_CLASSES[final_pred.item()]}")
    final_acc = correct / float(len(test_loader))
    print(f"Epsilon: {epsilon:.3f} \tTest Accuracy = {final_acc * 100:.2f}%")

    return final_acc, adv_examples

# Run test for different epsilons
epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.3]
accuracies = []
examples = []

for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# Plot accuracy
plt.figure(figsize=(6,6))
plt.plot(epsilons, accuracies, "*-")
plt.title("FGSM Attack - Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.grid()
plt.show()

# Show examples
plt.figure(figsize=(10,10))
cnt = 0
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]), cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel(f"Eps: {epsilons[i]}")
        orig, adv, ex = examples[i][j]
        plt.title(f"{CIFAR10_CLASSES[orig]}→{CIFAR10_CLASSES[adv]}")
        plt.imshow(np.transpose(ex, (1, 2, 0)))
plt.tight_layout()
plt.show()


