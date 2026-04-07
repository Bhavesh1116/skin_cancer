"""
HAM10000 Skin Cancer Dataset - ULTIMATE Model Training Script (V3)
===================================================================
Target: >95% Accuracy.
Uses EfficientNet-B4, Complete Network Unfreeze, Aggressive Med Augmentations,
and CosineAnnealingWarmRestarts to break out of local minima.
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
MODEL_PATH      = os.path.join(BASE_DIR, "skin_cancer_ultimate_model.pth")
BATCH_SIZE      = 16  # Reduced batch size because EfficientNet-B4 is huge memory-wise
EPOCHS          = 25  # More epochs needed for 95% target
IMG_SIZE        = 224 # Or 380 if GPU allows, keeping 224 for CPU survivability
OUTPUT_PLOT_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ULTIMATE TRAINING ACTIVATED! Using device: {device}")

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
# STEP 2 — DATASETS WITH ADVANCED MEDICAL AUGMENTATIONS
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

# EXPERT AUGMENTATIONS: Simulates how a doctor takes a bad picture
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
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
test_loader = DataLoader(SkinCancerDataset(df_test, val_test_transform), batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — HEAVY ARCHITECTURE (EFFICIENTNET-B4) & OPTIMIZERS
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading EfficientNet-B4 Architecture...")
model = models.efficientnet_b4(pretrained=True)

# Entire Network Unfrozen for Maximum Adaptability
for param in model.parameters():
    param.requires_grad = True

num_classes = len(classes)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model.classifier[1].in_features, 512),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512, num_classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Very low learning rate (1e-5) because entire network is unfrozen
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# Cosine Annealing bounces the learning rate to pop out of local minima
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — THE ULTIMATE TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nInitiating Training logic - Target Epochs: {EPOCHS}...")

best_acc = 0.0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(EPOCHS):
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
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['train_acc'].append(epoch_train_acc.item())
    history['val_acc'].append(epoch_val_acc.item())
    
    # Step scheduler
    scheduler.step()
    
    # Save best
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\n✅ TRAINING COMPLETE! Peak Ultimate Model saved to: {MODEL_PATH}")
