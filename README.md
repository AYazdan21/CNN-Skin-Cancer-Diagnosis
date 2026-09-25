# Enhanced Skin Cancer Diagnosis Using Optimized CNN (PyTorch)

An implementation of the paper:
> **"Enhanced skin cancer diagnosis using optimized CNN architecture and checkpoints for automated dermatological lesion classification"**  
> *M Mohamed Musthafa, Mahesh T R, Vinoth Kumar V, Suresh Guluwadi*  
> **BMC Medical Imaging (2024) 24:201**

---

## 📂 Project Structure

```
CNN-Skin-Cancer-Diagnosis/
├── dataverse_files/               # HAM10000 images & metadata
│   ├── HAM10000_images_part_1/    # 5,000 dermatoscopic images
│   ├── HAM10000_images_part_2/    # 5,015 dermatoscopic images
│   └── HAM10000_metadata          # CSV metadata (10,015 records)
├── src/
│   ├── config.py                  # Hyperparameters, paths, class mappings
│   ├── dataset.py                 # Resizing (28x28), normalization, balanced augmentation
│   ├── model.py                   # PyTorch CNN matching Table 2 (132,583 params)
│   ├── train.py                   # Training loop, Adam, ReduceLROnPlateau, EarlyStopping
│   └── evaluate.py                # Test evaluation, Confusion Matrix, ROC-AUC, Regression metrics
├── checkpoints/                   # Saved best model checkpoints (.pth)
├── outputs/                       # Metrics, classification report, and plots
├── requirements.txt               # Dependencies
├── main.py                        # CLI runner
└── README.md
```

---

## ⚙️ Setup & Installation

The virtual environment `.venv` is set up. To activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or run directly with the virtual environment Python:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

---

## 🚀 Usage

### 1. Train the model and evaluate on the test set:
```powershell
.\.venv\Scripts\python.exe main.py --epochs 50 --batch-size 128
```

### 2. Evaluate existing checkpoint:
```powershell
.\.venv\Scripts\python.exe main.py --evaluate-only
```

### 3. Ablation without balancing:
```powershell
.\.venv\Scripts\python.exe main.py --no-balance --epochs 50
```

---

## 📊 Outputs & Artifacts

After training and evaluation, the following artifacts are generated in `outputs/`:
- `training_curves.png`: Training vs Validation Loss & Accuracy curves over epochs.
- `confusion_matrix.png`: Multi-class confusion matrix across all 7 lesion categories.
- `roc_curves.png`: One-vs-Rest ROC curve and AUC score for each class.
- `classification_report.csv`: Precision, Recall, and F1-score for each category.
- `test_metrics.json`: Overall test accuracy and regression metrics (MSE, RMSE, MAE).
