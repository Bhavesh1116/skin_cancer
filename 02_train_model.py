"""
HAM10000 Skin Cancer Dataset - Model Training Script (PyTorch - V2)
=====================================================
Loads cleaned metadata, splits data preventing lesion leakage,
handles extreme imbalance using WeightedRandomSampler,
fine-tunes MobileNetV2 with 224x224 imgs for better precision/recall.
"""

import os
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = r"d:\skin cancer"
DATA_CSV        = os.path.join(BASE_DIR, "dataset", "cleaned_metadata.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "skin_cancer_model.pth")
BATCH_SIZE      = 32
EPOCHS          = 15
IMG_SIZE        = 224  # Increased to standard 224x224 for MobileNet
OUTPUT_PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA SPLITTING (AVOIDING LESION LEAKAGE)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Splitting data (Train/Val/Test)...")
print("=" * 60)

df = pd.read_csv(DATA_CSV)

# Create consistent mappings
classes = sorted(df['dx'].unique().tolist())
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
idx_to_class = {i: cls_name for cls_name, i in class_to_idx.items()}
num_classes = len(classes)

df['label_idx'] = df['dx'].map(class_to_idx)

# Data Split by lesion_id
unique_lesions = df.drop_duplicates(subset=['lesion_id'])
train_val_lesions, test_lesions = train_test_split(
    unique_lesions, test_size=0.20, random_state=42, stratify=unique_lesions['dx']
)
train_lesions, val_lesions = train_test_split(
    train_val_lesions, test_size=0.20, random_state=42, stratify=train_val_lesions['dx']
)

df_train = df[df['lesion_id'].isin(train_lesions['lesion_id'])].copy()
df_val = df[df['lesion_id'].isin(val_lesions['lesion_id'])].copy()
df_test = df[df['lesion_id'].isin(test_lesions['lesion_id'])].copy()

print(f"Train images: {len(df_train)} | Val images: {len(df_val)} | Test images: {len(df_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — WEIGHTED RANDOM SAMPLER FOR CLASS IMBALANCE
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 2: Configuring Weighted Random Sampler...")

# Calculate class distribution
class_counts = df_train['label_idx'].value_counts().sort_index().values
print("Training Class Counts:", dict(zip(classes, class_counts)))

# Weight = 1.0 / count
class_weights = 1.0 / class_counts

# Assign weight to each individual sample in the training set
sample_weights = [class_weights[label] for label in df_train['label_idx']]

# Create sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — PYTORCH DATASET AND DATALOADER
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 3: Preparing DataLoaders...")

class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.loc[idx, 'label_idx']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Enhanced Transforms for Medical Images
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),       # Added vertical flip
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SkinCancerDataset(df_train, transform=train_transform)
val_dataset = SkinCancerDataset(df_val, transform=val_test_transform)
test_dataset = SkinCancerDataset(df_test, transform=val_test_transform)

# Removed shuffle=True since we are using a sampler
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — MODEL BUILDING & FINE TUNING (MobileNetV2)
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 4: Building & Fine-Tuning Model...")

model = models.mobilenet_v2(pretrained=True)

# Unfreeze the last few blocks (layer 14+) of MobileNetV2 to learn medical features
for idx, child in enumerate(model.features.children()):
    if idx < 14:
        for param in child.parameters():
            param.requires_grad = False
    else:
        for param in child.parameters():
            param.requires_grad = True

# Replace classifier head
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256, num_classes)
)

model = model.to(device)

# Plain CrossEntropyLoss since Sampler already balances the batches!
criterion = nn.CrossEntropyLoss()

# Only optimize parameters that require gradients, use lower LR for fine-tuning
opt_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(opt_params, lr=3e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 5: Training Model...")

best_acc = 0.0
patience_counter = 0
early_stop_patience = 5

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_train_loss = running_loss / len(train_dataset)
    epoch_train_acc = running_corrects.double() / len(train_dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_corrects.double() / len(val_dataset)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
          f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
    
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_acc'].append(epoch_train_acc.item())
    history['val_acc'].append(epoch_val_acc.item())
    
    scheduler.step(epoch_val_loss)
    
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= early_stop_patience:
        print("Early stopping triggered due to plateau.")
        break

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — EVALUATION AND PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\nSTEP 6: Evaluation & Reporting...")

# Load best model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, 'training_history.png'))
plt.close()

# Evaluate on Test Set
test_loss = 0.0
test_corrects = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.data)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_dataset)
test_acc = test_corrects.double() / len(test_dataset)

print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# Metrics
print("\nClassification Report (V2 Model):")
print(classification_report(all_labels, all_preds, target_names=classes))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix (V2)')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PLOT_DIR, 'confusion_matrix.png'))
plt.close()

print(f"\n✅ Training V2 complete! Model saved to {MODEL_PATH}")
print(f"✅ Plots saved to {OUTPUT_PLOT_DIR}")
