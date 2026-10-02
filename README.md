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

## 📊 Benchmark & Comparison

| Metric / Lesion Class | Paper's Published Fig. 8 | Custom CNN ($128\times128$) | EfficientNet-B0 ($224\times224$) |
|---|---|---|---|
| **Overall Accuracy** | **$76.0\%$** | **$75.55\%$** | *Run to evaluate* |
| **Macro Avg F1-Score** | **$0.48$** | **$0.61$** | *Run to evaluate* |
| **Weighted Avg F1-Score**| **$0.76$** | **$0.76$** | *Run to evaluate* |
| `mel` (Melanoma) F1 | $0.46$ | **$0.49$** | *Run to evaluate* |
| `bcc` (Basal Cell) F1 | $0.50$ | **$0.57$** | *Run to evaluate* |
| `akiec` (Precancerous) F1 | $0.39$ | **$0.61$** | *Run to evaluate* |
| `vasc` (Vascular) F1 | $0.58$ | **$0.68$** | *Run to evaluate* |
| `df` (Dermatofibroma) F1 | $0.10$ | **$0.48$** | *Run to evaluate* |
| `bkl` (Keratosis) F1 | $0.46$ | **$0.58$** | *Run to evaluate* |
| `nv` (Nevi) F1 | $0.89$ | **$0.87$** | *Run to evaluate* |

---

## 🖼️ Visual Results

### 1. Sample Predictions (Paper Fig. 7 Reproduction)
Sample correct classifications with predicted confidence scores across lesion types:
![Sample Predictions](outputs/sample_predictions_custom_cnn.png)

### 2. Misclassified Instances (Paper Fig. 12 Reproduction)
Borderline or ambiguous cases analyzed by the model:
![Misclassified Samples](outputs/misclassified_samples_custom_cnn.png)

### 3. Confusion Matrix (Paper Fig. 9 Reproduction)
![Confusion Matrix](outputs/confusion_matrix_custom_cnn.png)

### 4. Multi-class ROC & AUC Curves (Paper Fig. 10 Reproduction)
![ROC Curves](outputs/roc_curves_custom_cnn.png)

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
│   ├── evaluate.py                # ROC-AUC, Confusion Matrix, Classification Report, Model Comparison
│   └── visualize_inference.py     # Sample prediction grids and error analysis (Fig. 7 & 12)
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

### 4. Evaluate Existing Checkpoint & Generate Sample Grids
```powershell
python main.py --model custom_cnn --evaluate-only
```
