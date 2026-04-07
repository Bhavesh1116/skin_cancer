"""
Colab Execution Script (KAGGLE FAST DOWNLOAD VERSION)
=====================================================
Instructions for running the Ultimate Model on Google Colab with Free T4 GPU.

1. Go to https://colab.research.google.com/
2. Click "Upload" and choose "New Notebook"
3. Go to Edit -> Notebook Settings -> Hardware Accelerator -> Choose "T4 GPU"
4. Copy the entire code block below into the Colab cell and press Play (Shift+Enter).
5. When the code runs, it will ask for your Kaggle Username and that Token/Key you have!
"""

import sys
IN_COLAB = 'google.colab' in sys.modules

if not IN_COLAB:
    print("Warning: This script is exclusively built for Google Colab. Do not run it locally.")

# --- COPY FROM AFTER THIS LINE INTO GOOGLE COLAB ---

import os
import json

# Setup Kaggle Credentials right inside Colab
print("--- KAGGLE SETUP ---")
kaggle_user = input("Enter your Kaggle Username (from your profile): ").strip()
kaggle_key = input("Enter your Kaggle Token/Key: ").strip()

os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump({"username": kaggle_user, "key": kaggle_key}, f)

os.chmod("/root/.kaggle/kaggle.json", 0o600)

print("\nDownloading Dataset via Kaggle (Super Fast 1000 Mbps)...")
os.system("pip install kaggle -q")
os.system("kaggle datasets download -d kmader/skin-cancer-mnist-ham10000")

print("\nUnpacking the dataset...")
os.makedirs("/content/dataset", exist_ok=True)
os.system("unzip -q -o skin-cancer-mnist-ham10000.zip -d /content/dataset")
print("Dataset ready!\n")

# Now we run the Ultimate Model Code
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
# COLAB PATH CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = "/content"
DATA_CSV        = os.path.join(BASE_DIR, "dataset", "HAM10000_metadata.csv") 
MODEL_PATH      = os.path.join(BASE_DIR, "skin_cancer_ultimate_model.pth")
BATCH_SIZE      = 32  # GPU can handle 32 efficiently
EPOCHS          = 25
IMG_SIZE        = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ULTIMATE TRAINING ACTIVATED! Using device: {device}")
if device.type != 'cuda':
    print("WARNING: YOU ARE NOT USING A GPU. Go to Runtime -> Change Runtime Type -> T4 GPU")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CLEANING & SAMPLER
# ─────────────────────────────────────────────────────────────────────────────
print("Cleaning data...")
df = pd.read_csv(DATA_CSV)
df['age'] = df['age'].fillna(df['age'].median())
df.loc[df['age'] == 0.0, 'age'] = df['age'].median()
df['sex'] = df['sex'].fillna(df['sex'].mode()[0])

def find_image_path(image_id):
    fname = f"{image_id}.jpg"
    p1 = f"/content/dataset/HAM10000_images_part_1/{fname}"
    p2 = f"/content/dataset/HAM10000_images_part_2/{fname}"
    p3 = f"/content/dataset/HAM10000_images_part_1/HAM10000_images_part_1/{fname}" # Fallbacks
    p4 = f"/content/dataset/HAM10000_images_part_2/HAM10000_images_part_2/{fname}"
    
    if os.path.isfile(p1): return p1
    elif os.path.isfile(p2): return p2
    elif os.path.isfile(p3): return p3
    elif os.path.isfile(p4): return p4
    return None

df['image_path'] = df['image_id'].apply(find_image_path)
df = df[df['image_path'].notnull()].copy()

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
# STEP 2 — ADVANCED MEDICAL AUGMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────
class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self): return len(self.dataframe)
    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.loc[idx, 'label_idx']
        if self.transform: image = self.transform(image)
        return image, label

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

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EFFICIENTNET-B4 & TRAINING (COLAB GPU SUPER SPEED)
# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading EfficientNet-B4 Architecture...")
model = models.efficientnet_b4(pretrained=True)
for param in model.parameters(): param.requires_grad = True
model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(model.classifier[1].in_features, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, len(classes)))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4) # Low learning rate as network is unfrozen
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

print(f"\nInitiating Training logic on GPU - Target Epochs: {EPOCHS}...")
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects, val_loss, val_corrects = 0.0, 0, 0.0, 0
    
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
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            val_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)
            
    epoch_val_loss = val_loss / len(val_loader.dataset) # type: ignore
    epoch_val_acc = val_corrects.double() / len(val_loader.dataset) # type: ignore
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    scheduler.step()
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\n✅ TRAINING COMPLETE! Peak Ultimate Model saved to: {MODEL_PATH}")
print("Go to the left file browser in Colab, right click 'skin_cancer_ultimate_model.pth' and hit DOWNLOAD!")
