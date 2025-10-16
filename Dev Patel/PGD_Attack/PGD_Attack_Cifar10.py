import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
using_device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"Using device: {using_device}")

# Class labels for CIFAR-10
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Define a basic CNN
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

# PGD Attack
def pgd_attack(model, images, labels, epsilon, alpha, iters):
    ori_images = images.clone().detach()

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        grad = images.grad.data
        adv_images = images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    return images

# Load a small batch of test data
test_set = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

# Load model (random weights for now)
# Training the CNN on CIFAR-10 (simplified training for demonstration)

# Load CIFAR-10 training data
train_set = CIFAR10(root='./data', train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# Define loss and optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train for a few epochs
model.train()
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save trained model weights
torch.save(model.state_dict(), "cnn_cifar10_trained.pth")

# Clear model
model = CNN().to(device)
model.load_state_dict(torch.load("cnn_cifar10_trained.pth"))
model.eval()

# Evaluate PGD on a few samples
epsilon = 0.07
alpha = 0.01
iters = 10

pgd_results = []
correct = 0

for i, (data, target) in enumerate(test_loader):
    if i >= 5:
        break
    data, target = data.to(device), target.to(device)
    adv_data = pgd_attack(model, data.clone(), target, epsilon, alpha, iters)
    output = model(adv_data)
    pred = output.max(1, keepdim=True)[1]

    pgd_results.append((data.squeeze().detach().cpu(), adv_data.squeeze().detach().cpu(), target.item(), pred.item()))

pgd_results

# Plot results
fig, axes = plt.subplots(len(pgd_results), 2, figsize=(8, 4 * len(pgd_results)))
for i, (orig, adv, true_label, adv_label) in enumerate(pgd_results):
    axes[i, 0].imshow(np.transpose(orig.numpy(), (1, 2, 0)))
    axes[i, 0].set_title(f"Original: {CIFAR10_CLASSES[true_label]}")
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(np.transpose(adv.numpy(), (1, 2, 0)))
    axes[i, 1].set_title(f"Adversarial: {CIFAR10_CLASSES[adv_label]}")
    axes[i, 1].axis('off')  
plt.tight_layout()
plt.savefig('pgd_cifar10_results.png')
plt.show()  


# Evaluate accuracy under PGD attack for different epsilons
epsilon = [0, 0.01, 0.03, 0.05, 0.07, 0.1]
pgd_results = []
for eps in epsilon:
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        adv_data = pgd_attack(model, data.clone(), target, eps, alpha=0.01, iters=10)
        output = model(adv_data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    accuracy = correct / total
    pgd_results.append(accuracy)
    print(f"Epsilon: {eps}\tTest Accuracy = {accuracy * 100:.2f}%") 

# Plot accuracy vs epsilon

plt.plot(epsilon, pgd_results, marker='o')
plt.title("PGD Attack - Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.grid()
plt.savefig('pgd_cifar10_accuracy.png')
plt.show()

print("PGD attack results saved as 'pgd_cifar10_results.png' and accuracy plot as 'pgd_cifar10_accuracy.png'")