"""
Author: Riyaz
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time
import numpy as np

# -------------------------------
# Step 1: Prepare dataset
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

input_size = 28*28
output_size = 10
n_epochs = 10
batch_size = 64

# Split train/validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# Step 2: Define logistic regression from scratch (no class)
# -------------------------------

# Initialize weights and bias
w = torch.randn(input_size, output_size, requires_grad=True)
b = torch.zeros(output_size, requires_grad=True)

# Softmax function for predictions
def softmax(x):
    exp_x = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return exp_x / exp_x.sum(dim=1, keepdim=True)

# Prediction function
def predict(X):
    return X @ w + b

# -------------------------------
# Step 3: Loss and optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([w, b], lr=0.001)

# -------------------------------
# Step 4: Training loop
# -------------------------------
train_losses, val_losses = [], []
train_acc_list, val_acc_list = [], []

for epoch in range(n_epochs):
    start_time = time.time()
    
    # Training
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images = images.view(-1, input_size)
        optimizer.zero_grad()
        outputs = predict(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_losses.append(total_loss / len(train_loader))
    train_acc_list.append(correct / total)
    
    # Validation
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.view(-1, input_size)
            outputs = predict(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_losses.append(val_loss / len(val_loader))
    val_acc_list.append(correct / total)
    
    print(f"Epoch {epoch+1}/{n_epochs}, "
          f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, "
          f"Train Acc: {train_acc_list[-1]:.4f}, Val Acc: {val_acc_list[-1]:.4f}, "
          f"Time: {time.time()-start_time:.2f}s")
    
    # Plotting 9 test images at last epoch
    if epoch == n_epochs - 1:
        test_iter = iter(test_loader)
        images, labels = next(test_iter)
        images_flat = images.view(-1, input_size)
        with torch.no_grad():
            outputs = predict(images_flat)
            preds = torch.argmax(outputs, dim=1)
        plot_images(images=images[:9], labels=labels[:9], preds=preds[:9])

# -------------------------------
# Step 5: Test accuracy
# -------------------------------
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, input_size)
        outputs = predict(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")

# -------------------------------
# Step 6: Plot training and validation loss
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, n_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, n_epochs+1), val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Step 7: Plot weights
# -------------------------------
def plot_weights(w, shape=(28,28)):
    w_min, w_max = w.min().item(), w.max().item()
    fig, axes = plt.subplots(2,5, figsize=(12,5))
    axes = axes.flatten()
    for i in range(10):
        img = w[:, i].view(shape).detach().numpy()
        axes[i].imshow(img, cmap='seismic', vmin=w_min, vmax=w_max)
        axes[i].set_title(f"Class {i}")
        axes[i].axis('off')
    plt.show()

plot_weights(w)

# -------------------------------
# Step 8: Plot sample images with true/predicted labels
# -------------------------------
def plot_images(images, labels, preds=None, img_shape=(28,28)):
    assert len(images) == len(labels) == 9  # 3x3 grid
    
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, ax in enumerate(axes.flat):
        img = images[i].view(img_shape).numpy() if torch.is_tensor(images[i]) else images[i].reshape(img_shape)
        ax.imshow(img, cmap='binary')
        
        if preds is None:
            xlabel = f"True: {labels[i].item()}" if torch.is_tensor(labels[i]) else f"True: {labels[i]}"
        else:
            xlabel = f"True: {labels[i].item() if torch.is_tensor(labels[i]) else labels[i]}, Pred: {preds[i].item() if torch.is_tensor(preds[i]) else preds[i]}"
        
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
