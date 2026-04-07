"""
Skin Cancer Image Predictor (Smart CLI UI)
==========================================
Tests the trained ResNet18 model on any random skin lesion image.
"""

import os
import torch
from torchvision import models, transforms
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = r"d:\skin cancer\skin_cancer_medium_model.pth"
IMG_SIZE = 224

CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_NAMES = {
    'nv': 'Normal Mole (Melanocytic Nevi)',
    'mel': 'Melanoma (Skin Cancer)',
    'bkl': 'Benign Keratosis (Age Spot)',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratoses (Pre-Cancerous)',
    'vasc': 'Vascular Lesion',
    'df': 'Dermatofibroma (Harmless Bump)'
}
SEVERITY = {
    'nv':    '🟢 SAFE (Benign)',
    'mel':   '🔴 HIGH RISK (Malignant/Dangerous)',
    'bkl':   '🟢 SAFE (Benign)',
    'bcc':   '🔴 HIGH RISK (Malignant/Dangerous)',
    'akiec': '🟠 WARNING (Pre-Cancerous / Needs observation)',
    'vasc':  '🟢 SAFE (Benign)',
    'df':    '🟢 SAFE (Benign)'
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"\n[SYSTEM] Loading AI diagnostic engine on {device}...")
    model = models.resnet18(pretrained=False)
    
    import torch.nn as nn
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, len(CLASSES))
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find model at {MODEL_PATH}")
        return None
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    return model

def predict_image(image_path, model):
    if not os.path.exists(image_path):
        print(f"\n❌ Error: Could not find image at {image_path}")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 3)

        predicted_class = CLASSES[top_catid[0]]
        confidence = top_prob[0].item() * 100
        
        print("\n" + "═"*60)
        print(f" 🩺 AI DIAGNOSTIC REPORT")
        print("═"*60)
        print(f" FILE:       {os.path.basename(image_path)}")
        print("─"*60)
        print(f" DIAGNOSIS:  {CLASS_NAMES[predicted_class]}")
        print(f" CONFIDENCE: {confidence:.2f}%")
        print(f" RISK LEVEL: {SEVERITY[predicted_class]}")
        print("─"*60)
        
        print(" TOP 3 ALTERNATIVE POSSIBILITIES:")
        for i in range(3):
            cls = CLASSES[top_catid[i]]
            prob = top_prob[i].item() * 100
            print(f"   {i+1}. {CLASS_NAMES[cls]:<35} | {prob:>6.2f}% | {SEVERITY[cls]}")
        print("═"*60 + "\n")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    model = load_model()
    if model:
        print("\n✅ AI is ready! You can continuously paste image paths.")
        while True:
            img_path = input("📸 Paste Full Path to Image (or 'q' to quit): ").strip()
            img_path = img_path.strip('"').strip("'")
            
            if img_path.lower() == 'q':
                break
                
            predict_image(img_path, model)
