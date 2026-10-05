import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split

# ======================
# Config
# ======================
data_dir = "C:\\Users\\K. Kishan Babu\\Documents\\classifier_dataset\\MRI"
num_epochs = 120
batch_size = 16
lr = 1e-4
best_model_path = "best_convnext_custom_mri.pth"
val_split = 0.2
patience = 20
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Transforms (MATCH YOUR INFERENCE)
# ======================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# Dataset Split
# ======================
base_dataset = datasets.ImageFolder(data_dir)
targets = base_dataset.targets
class_names = base_dataset.classes

train_idx, val_idx = train_test_split(
    np.arange(len(targets)),
    test_size=val_split,
    stratify=targets,
    random_state=seed
)

train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(data_dir, transform=val_transform)

train_ds = Subset(train_dataset, train_idx)
val_ds   = Subset(val_dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

print("Classes:", class_names)
print("Train size:", len(train_ds))
print("Val size:", len(val_ds))

# ======================
# Model (YOUR CUSTOM VERSION)
# ======================
model = models.convnext_base(pretrained=True)

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

in_features = model.classifier[2].in_features

# 🔥 YOUR CLASSIFIER
model.classifier = nn.Sequential(
    model.classifier[0],   # keep original LayerNorm2d
    nn.Flatten(1),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 2)
)

model = model.to(device)

# ======================
# Loss & Optimizer
# ======================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {"params": model.classifier.parameters(), "lr": 1e-4},
    {"params": model.features.parameters(), "lr": 1e-5}
])

# ======================
# Training Setup
# ======================
best_val_loss = float("inf")
best_val_acc = 0.0
best_weights = copy.deepcopy(model.state_dict())
counter = 0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

# ======================
# Training Loop
# ======================
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    for phase in ["train", "val"]:
        model.train() if phase == "train" else model.eval()
        loader = train_loader if phase == "train" else val_loader

        running_loss = 0.0
        running_corrects = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.double() / len(loader.dataset)

        if phase == "train":
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.item())
        else:
            val_losses.append(epoch_loss)
            val_accs.append(epoch_acc.item())

            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_val_acc = epoch_acc.item()
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, best_model_path)
                counter = 0
            else:
                counter += 1
                print(f"EarlyStopping counter: {counter}/{patience}")
                if counter >= patience:
                    print("Early stopping triggered!")
                    break

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # 🔥 Fine-tuning after 10 epochs
    if epoch == 10:
        print("Unfreezing last ConvNeXt block...")
        for param in model.features[-1].parameters():
            param.requires_grad = True

    if counter >= patience:
        break

print("\nTraining complete.")
print("Best validation accuracy:", best_val_acc)

model.load_state_dict(best_weights)