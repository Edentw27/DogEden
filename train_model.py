"""
train_model.py
---------------
MobileNetV2 transfer-learning trainer for the 6 objects. Reads photos from
data/, trains a classifier, saves the model + class names + an accuracy plot
into model/.

(Optional / for the report. The final project uses pure colour+contour vision
in detect_object.py and main.py, so the robot does NOT need this model to run.)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
import json

# ── Settings ──────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "model"
EPOCHS     = 15
BATCH_SIZE = 16
IMG_SIZE   = 224
LR         = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Data transforms ───────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Load dataset ──────────────────────────────────────
print("\n" + "=" * 50)
print("  ROBODOG MODEL TRAINING")
print("=" * 50)
print("\n[1/5] Loading images...")

full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
class_names  = full_dataset.classes
num_classes  = len(class_names)

print(f"  Found {len(full_dataset)} images in {num_classes} classes:")
for i, name in enumerate(class_names):
    print(f"    {i}: {name}")

with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)
print("\n  Class names saved to model/class_names.json")

# ── Split into train / validation ─────────────────────
val_size   = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform   = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"\n  Training images:   {train_size}")
print(f"  Validation images: {val_size}")

# ── Build model (MobileNetV2 + transfer learning) ─────
print("\n[2/5] Building model (MobileNetV2 transfer learning)...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze the feature extractor; only train the new classifier head.
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier with one for our 6 classes.
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

# ── Train ─────────────────────────────────────────────
print("\n[3/5] Training...")
train_acc_history, val_acc_history = [], []

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    train_acc = 100.0 * correct / total

    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, pred = outputs.max(1)
            v_total += labels.size(0)
            v_correct += pred.eq(labels).sum().item()
    val_acc = 100.0 * v_correct / v_total

    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    print(f"  Epoch {epoch+1:2d}/{EPOCHS}  "
          f"loss={running_loss/len(train_loader):.3f}  "
          f"train={train_acc:5.1f}%  val={val_acc:5.1f}%")

# ── Save model ────────────────────────────────────────
print("\n[4/5] Saving model...")
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "robodog_model.pth"))
print("  Saved to model/robodog_model.pth")

# ── Save accuracy plot ────────────────────────────────
print("\n[5/5] Saving accuracy plot...")
plt.figure()
plt.plot(train_acc_history, label="train")
plt.plot(val_acc_history, label="validation")
plt.xlabel("epoch")
plt.ylabel("accuracy (%)")
plt.title("RoboDog training accuracy")
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "training_accuracy.png"))
print("  Saved to model/training_accuracy.png")

print("\n  TRAINING COMPLETE!")
