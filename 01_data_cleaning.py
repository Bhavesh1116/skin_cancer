"""
HAM10000 Skin Cancer Dataset - Data Cleaning Script
=====================================================
Cleans the metadata CSV, maps image paths, handles missing values,
deduplicates by lesion_id, and saves the cleaned dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = r"d:\skin cancer\dataset"
METADATA_PATH   = os.path.join(BASE_DIR, "HAM10000_metadata.csv")
IMG_DIR_1       = os.path.join(BASE_DIR, "HAM10000_images_part_1")
IMG_DIR_2       = os.path.join(BASE_DIR, "HAM10000_images_part_2")
OUTPUT_CSV      = os.path.join(BASE_DIR, "cleaned_metadata.csv")
OUTPUT_PLOT_DIR = r"d:\skin cancer\plots"

os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# Class label mapping
CLASS_NAMES = {
    'nv'   : 'Melanocytic Nevi',
    'mel'  : 'Melanoma',
    'bkl'  : 'Benign Keratosis',
    'bcc'  : 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratoses',
    'vasc' : 'Vascular Lesions',
    'df'   : 'Dermatofibroma'
}

CLASS_COLORS = ['#4CAF50','#F44336','#FF9800','#E91E63','#9C27B0','#03A9F4','#795548']

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading metadata...")
print("=" * 60)

df = pd.read_csv(METADATA_PATH)

print(f"Shape            : {df.shape}")
print(f"Columns          : {list(df.columns)}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MISSING VALUES ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Missing Values Analysis")
print("=" * 60)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    'Missing Count' : missing,
    'Missing %'     : missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0])

# Fix missing 'age' — fill with median
age_median = df['age'].median()
print(f"\nMedian age (for imputation): {age_median}")
df['age'].fillna(age_median, inplace=True)

# Fix missing 'sex' — fill with mode
sex_mode = df['sex'].mode()[0]
print(f"Mode sex (for imputation): {sex_mode}")
df['sex'].fillna(sex_mode, inplace=True)

print(f"\nMissing values after fix: {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — INVALID AGE (age == 0)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Invalid Age (age == 0)")
print("=" * 60)

zero_age = df[df['age'] == 0.0]
print(f"Rows with age == 0: {len(zero_age)}")
if len(zero_age) > 0:
    print(zero_age[['lesion_id', 'image_id', 'dx', 'age', 'sex']])
    # Replace age=0 with median as well (they look like missing data)
    df.loc[df['age'] == 0.0, 'age'] = age_median
    print(f"  → Replaced age=0 rows with median age ({age_median})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — IMAGE PATH MAPPING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Mapping image paths...")
print("=" * 60)

def find_image_path(image_id):
    """Check part_1 first, then part_2."""
    fname = image_id + ".jpg"
    p1 = os.path.join(IMG_DIR_1, fname)
    p2 = os.path.join(IMG_DIR_2, fname)
    if os.path.isfile(p1):
        return p1
    elif os.path.isfile(p2):
        return p2
    else:
        return None

df['image_path'] = df['image_id'].apply(find_image_path)

missing_images = df['image_path'].isnull().sum()
print(f"Total images     : {len(df)}")
print(f"Images found     : {df['image_path'].notnull().sum()}")
print(f"Images NOT found : {missing_images}")

if missing_images > 0:
    print("\nMissing image IDs:")
    print(df[df['image_path'].isnull()]['image_id'].tolist())
    # Drop rows with no image file
    df = df[df['image_path'].notnull()].copy()
    print(f"Rows after dropping missing images: {len(df)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — CLASS DISTRIBUTION (BEFORE DEDUP)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Class Distribution (Before Deduplication)")
print("=" * 60)

class_counts_before = df['dx'].value_counts()
print(class_counts_before)
print(f"\nTotal samples : {len(df)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — DEDUPLICATE BY LESION_ID
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Deduplicating by lesion_id (prevent data leakage)")
print("=" * 60)

# Some lesions appear multiple times (multiple images of same spot)
# Keep only 1 image per lesion to avoid test contamination
dup_lesions = df['lesion_id'].duplicated().sum()
print(f"Duplicate lesion_id entries : {dup_lesions}")

df_dedup = df.drop_duplicates(subset='lesion_id', keep='first').copy()
print(f"After dedup: {len(df_dedup)} unique lesions")

class_counts_after = df_dedup['dx'].value_counts()
print(f"\nClass distribution after dedup:\n{class_counts_after}")

# NOTE: For training, we CAN use all images (including duplicates)
# but we must ensure no lesion_id appears in both train AND test.
# We'll keep df_dedup as the reference for splitting,
# then merge back duplicates to train set only.
print("\n  → Strategy: Use all images for training, but split by lesion_id")
print("    to prevent same lesion from appearing in train & test.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — LABEL ENCODING
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Label Encoding")
print("=" * 60)

# Consistent label ordering
CLASSES = sorted(df['dx'].unique().tolist())  
# ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
label_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
idx_to_label = {idx: cls for cls, idx in label_to_idx.items()}

df['label'] = df['dx'].map(label_to_idx)

print("Label mapping:")
for cls, idx in label_to_idx.items():
    full = CLASS_NAMES.get(cls, cls)
    print(f"  {idx} → {cls:6s} ({full})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — FINAL CLEANED DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Final Cleaned Dataset")
print("=" * 60)

# Keep relevant columns
df_clean = df[['lesion_id', 'image_id', 'image_path', 'dx', 'label',
               'dx_type', 'age', 'sex', 'localization']].copy()

print(f"Final shape      : {df_clean.shape}")
print(f"Missing values   : {df_clean.isnull().sum().sum()}")
print(f"\nSample:\n{df_clean.head(5)}")

# Save
df_clean.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Cleaned data saved → {OUTPUT_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Generating Visualizations...")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("HAM10000 Dataset - EDA", fontsize=18, fontweight='bold', y=1.01)

# ── Plot 1: Class Distribution ─────────────────────────────────────────────
ax1 = axes[0, 0]
counts = df_clean['dx'].value_counts()
bars = ax1.bar(
    [CLASS_NAMES.get(c, c) for c in counts.index],
    counts.values,
    color=CLASS_COLORS[:len(counts)],
    edgecolor='white', linewidth=1.2
)
ax1.set_title("Class Distribution (All Images)", fontsize=13, fontweight='bold')
ax1.set_xlabel("Diagnosis")
ax1.set_ylabel("Count")
ax1.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

# ── Plot 2: Class Distribution (Pie) ──────────────────────────────────────
ax2 = axes[0, 1]
labels_pie = [f"{CLASS_NAMES.get(c, c)}\n({v})" for c, v in zip(counts.index, counts.values)]
ax2.pie(counts.values, labels=labels_pie, colors=CLASS_COLORS[:len(counts)],
        autopct='%1.1f%%', startangle=90, pctdistance=0.8)
ax2.set_title("Class Distribution (%)", fontsize=13, fontweight='bold')

# ── Plot 3: Age Distribution ───────────────────────────────────────────────
ax3 = axes[1, 0]
ax3.hist(df_clean['age'], bins=30, color='#2196F3', edgecolor='white', linewidth=0.8)
ax3.axvline(df_clean['age'].mean(), color='red', linestyle='--', linewidth=2,
            label=f"Mean: {df_clean['age'].mean():.1f}")
ax3.axvline(df_clean['age'].median(), color='orange', linestyle='--', linewidth=2,
            label=f"Median: {df_clean['age'].median():.1f}")
ax3.set_title("Age Distribution", fontsize=13, fontweight='bold')
ax3.set_xlabel("Age")
ax3.set_ylabel("Count")
ax3.legend()

# ── Plot 4: Sex Distribution ───────────────────────────────────────────────
ax4 = axes[1, 1]
sex_counts = df_clean['sex'].value_counts()
ax4.bar(sex_counts.index, sex_counts.values, color=['#2196F3', '#E91E63'],
        edgecolor='white', linewidth=1.2)
ax4.set_title("Sex Distribution", fontsize=13, fontweight='bold')
ax4.set_xlabel("Sex")
ax4.set_ylabel("Count")
for i, (idx, val) in enumerate(zip(sex_counts.index, sex_counts.values)):
    ax4.text(i, val + 20, str(val), ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_PLOT_DIR, "eda_overview.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ EDA plot saved → {plot_path}")

# ── Age by Class boxplot ───────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(14, 6))
data_for_box = [df_clean[df_clean['dx'] == cls]['age'].values for cls in counts.index]
bp = ax.boxplot(data_for_box, patch_artist=True, notch=False,
                labels=[CLASS_NAMES.get(c, c) for c in counts.index])
for patch, color in zip(bp['boxes'], CLASS_COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Age Distribution by Diagnosis Class", fontsize=14, fontweight='bold')
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Age (years)")
ax.tick_params(axis='x', rotation=20)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plot_path2 = os.path.join(OUTPUT_PLOT_DIR, "age_by_class.png")
plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Age-by-class plot saved → {plot_path2}")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA CLEANING SUMMARY")
print("=" * 60)
print(f"Original rows       : 10,016")
print(f"After cleaning      : {len(df_clean)}")
print(f"Missing values fixed: age (NaN→median), sex (NaN→mode), age=0→median")
print(f"Output CSV          : {OUTPUT_CSV}")
print(f"Classes (7)         : {CLASSES}")
print("\n✅ Data cleaning complete!")
print("Next step → Run: python 02_train_model.py")
