"""
HAM10000 Skin Cancer Dataset - MEDIUM Model Training Script
===================================================================
Target: 83% - 88% Accuracy.
Uses ResNet18 (Middle ground weight: ~11M params).
Trains efficiently on Local CPU within ~2 to 3 hours.
"""

import os
import time
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

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = r"d:\skin cancer"
DATA_CSV        = os.path.join(BASE_DIR, "dataset", "cleaned_metadata.csv")
MODEL_PATH      = os.path.join(BASE_DIR, "skin_cancer_medium_model.pth")
BATCH_SIZE      = 16  # ResNet18 runs smoother with smaller batches on CPU
EPOCHS          = 12  # Incremental goal, no need for 25
IMG_SIZE        = 224
OUTPUT_PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"MEDIUM TRAINING ACTIVATED! Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA SPLITTING & SAMPLER
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_CSV)
classes = sorted(df['dx'].unique().tolist())
class_to_idx = {cls: i for i, cls in enumerate(classes)}
df['label_idx'] = df['dx'].map(class_to_idx)

unique_lesions = df.drop_duplicates(subset=['lesion_id'])
train_val_lesions, test_lesions = train_test_split(unique_lesions, test_size=0.20, random_state=42, stratify=unique_lesions['dx'])
train_lesions, val_lesions = train_test_split(train_val_lesions, test_size=0.20, random_state=42, stratify=train_val_lesions['dx'])

df_train = df[df['lesion_id'].isin(train_lesions['lesion_id'])].reset_index(drop=True)
df_val = df[df['lesion_id'].isin(val_lesions['lesion_id'])].reset_index(drop=True)
df_test = df[df['lesion_id'].isin(test_lesions['lesion_id'])].reset_index(drop=True)

class_counts = df_train['label_idx'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
sample_weights = [class_weights[label] for label in df_train['label_idx']]

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DATASETS WITH BALANCED AUGMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────
class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
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

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(SkinCancerDataset(df_train, train_transform), batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(SkinCancerDataset(df_val, val_test_transform), batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — MEDIUM ARCHITECTURE (RESNET18)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading ResNet18 Architecture (~11 Million Params)...")
model = models.resnet18(pretrained=True)

# Unfreeze entire network
for param in model.parameters():
    param.requires_grad = True

num_features = model.fc.in_features
num_classes = len(classes)
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.4),
    nn.Linear(256, num_classes)
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
# Still using a low learning rate since we're unfreezing the whole ResNet
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

# Dynamic cyclical speed
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=1)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — THE TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nInitiating Training logic - Target Epochs: {EPOCHS}...")

best_acc = 0.0

for epoch in range(EPOCHS):
    start_time = time.time()
    
    model.train()
    running_loss, running_corrects = 0.0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_train_loss = running_loss / len(train_loader.dataset) # type: ignore
    epoch_train_acc = running_corrects.double() / len(train_loader.dataset) # type: ignore
    
    # Validation
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
            
    epoch_val_loss = val_loss / len(val_loader.dataset) # type: ignore
    epoch_val_acc = val_corrects.double() / len(val_loader.dataset) # type: ignore
    
    epoch_time = time.time() - start_time
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} [{epoch_time/60:.1f}m] | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
    
    scheduler.step()
    
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\n✅ TRAINING COMPLETE! Peak Medium Model saved to: {MODEL_PATH}")
