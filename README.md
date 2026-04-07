# Skin Cancer Classification System 🩺🏥

A comprehensive, end-to-end Deep Learning pipeline built with PyTorch to classify different types of skin cancer lesions using the HAM10000 dataset. 

This project spans from initial data cleaning to deploying various deep learning architectures (like ResNet18 and EfficientNet-B4) tailored for different hardware constraints (Local CPU vs. Cloud GPU).

## 🚀 Key Features

- **Robust Data Preprocessing:** Includes automated removal of duplicates, missing value handling, and robust metadata structuring.
- **Handling Class Imbalance:** Uses PyTorch's `WeightedRandomSampler` to ensure uniform exposure to all lesion classes during training.
- **Multi-Tiered Model Architecture:**
  - **Medium Model (ResNet18):** Designed for local CPU training. Reaches optimal weights within 2-3 hours and allows rapid iteration.
  - **Ultimate Model (EfficientNet-B4):** Deployed on Google Colab GPUs to achieve maximum accuracy.
- **Dynamic Augmentation:** Employs advanced `transforms` including Random Flips, Rotations, and ColorJitters, augmented with custom `Dropout` layers and a `CosineAnnealingWarmRestarts` learning rate scheduler to prevent overfitting.
- **Bulk Inference:** Script to rapidly test predictions on unseen data and output results as a CSV report.

## 📁 Repository Structure

- `01_data_cleaning.py`: Script to process the HAM10000 dataset, creating a clean metadata CSV for training.
- `02_train_model.py`: Initial baseline deep learning training script.
- `03_predict_image.py`: Inference script to run images through the trained models and generate `bulk_prediction_results.csv`.
- `04_train_ultimate_model.py`: Script to train the heavy EfficientNet-B4 model optimized for maximum accuracy.
- `06_colab_train.py`: Modified training script optimized for Google Colab GPU environment.
- `08_train_medium_model.py`: Script to train the ResNet18 model locally on CPU.
- `requirements.txt`: Python package dependencies for the project.

## 🛠️ Tech Stack
- **Deep Learning Framework:** PyTorch, Torchvision
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning Utilities:** Scikit-Learn
- **Image Processing:** PIL (Pillow)
- **Visualization:** Matplotlib, Seaborn

## ⚙️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure you have the HAM10000 dataset inside the `dataset` folder.
```bash
python 01_data_cleaning.py
```

### 3. Train the Model
**For Local CPU Training (ResNet18):**
```bash
python 08_train_medium_model.py
```
**For Cloud GPU Training (EfficientNet-B4):**
Upload the project along with `dataset_colab.zip` to Google Colab and execute:
```bash
python 06_colab_train.py
```

### 4. Run Predictions
Use the designated trained `.pth` models (e.g., `skin_cancer_medium_model.pth`) for predicting new lesion images.
```bash
python 03_predict_image.py
```

## ⚠️ Challenges Overcome
1. **Severe Imbalance:** Mitigated using dynamic `WeightedRandomSampler` to balance batch loads natively.
2. **Hardware Limitations:** Split training architectures between local CPU handling (ResNet18) and Cloud processing (EfficientNet-B4).
3. **Overfitting on Minority Classes:** Controlled via extensive PyTorch data augmentations (`RandomRotation`, `ColorJitter`) and adjusted Fully Connected layers with `Dropout`.

## 🤝 Contribution
Feel free to fork this project, submit pull requests, or open an issue if you encounter any bugs!
