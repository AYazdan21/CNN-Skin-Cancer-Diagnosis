# Enhanced Skin Cancer Diagnosis: Custom CNN vs. EfficientNet-B0 (PyTorch)

An implementation and benchmark of deep learning architectures for dermatological lesion classification on the **HAM10000** dataset, based on and expanding upon:

> **"Enhanced skin cancer diagnosis using optimized CNN architecture and checkpoints for automated dermatological lesion classification"**  
> *M Mohamed Musthafa, Mahesh T R, Vinoth Kumar V, Suresh Guluwadi*  
> **BMC Medical Imaging (2024) 24:201**

---

## 🔬 Supported Architectures

1. **Custom CNN (Paper Architecture from Scratch)**:
   - 4-block Conv2D + MaxPool + Dropout + Dense layers ($132,583$ parameters).
   - Trained strictly from scratch on downscaled images ($28\times28$ to $128\times128$).
   - Reaches **$75.55\%$ accuracy** and **$0.61$ Macro F1** (surpassing the paper's reported Figure 8 Macro F1 of $0.48$).

2. **EfficientNet-B0 (Transfer Learning)**:
   - Pre-trained on ImageNet ($\approx 5.3\text{M}$ parameters).
   - High resolution ($224 \times 224$) with ImageNet normalization.
   - Fine-tuned in just $15$ epochs to achieve high sensitivity and diagnostic accuracy ($86\% - 91\%$).

---

## 📂 Project Structure

```
CNN-Skin-Cancer-Diagnosis/
├── dataverse_files/               # Raw HAM10000 images & metadata (local)
├── src/
│   ├── config.py                  # Hyperparameters, paths, auto-detection for Kaggle
│   ├── dataset.py                 # Resizing, ImageNet/standard normalization, balanced augmentation
│   ├── model.py                   # Custom CNN (from scratch) & EfficientNet-B0 (transfer learning)
│   ├── train.py                   # Multi-GPU training, Adam, ReduceLROnPlateau, EarlyStopping
│   └── evaluate.py                # ROC-AUC, Confusion Matrix, Classification Report, Model Comparison
├── checkpoints/                   # Saved model weights (.pth)
├── outputs/                       # Metrics, classification reports, plots, comparison CSV
├── requirements.txt               # Dependencies
├── main.py                        # CLI runner orchestrating training, evaluation, & comparison
└── README.md
```

---

## 🚀 Usage

### 1. Run Custom CNN (Paper Baseline)
```powershell
python main.py --model custom_cnn --img-size 128 --batch-norm --smooth-balance --epochs 50 --batch-size 128
```

### 2. Run EfficientNet-B0 (Transfer Learning)
```powershell
python main.py --model efficientnet --img-size 224 --epochs 15 --lr 3e-4 --smooth-balance --batch-size 128
```

### 3. Compare Both Models Side-by-Side
```powershell
python main.py --compare
```
This reads the evaluation metrics from `outputs/` and outputs a benchmark comparison table saved to `outputs/model_comparison.csv`.
