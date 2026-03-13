# UNIDO AfricaRice Quality Assessment Challenge

[![Zindi](https://img.shields.io/badge/Zindi-Competition-orange)](https://zindi.africa/competitions/unido-africarice-quality-assessment-challenge)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Final Standing: 16th place out of all participants**  
> Public Score: `0.8796` | Private Score: `0.8686`

---

## Competition Overview

**Hosted by:** [Zindi Africa](https://zindi.africa/competitions/unido-africarice-quality-assessment-challenge) in partnership with **UNIDO** and **AfricaRice**

Manual rice quality inspection is slow, expensive, and inconsistent — especially for smallholder farmers and food processors in Africa. This challenge asks participants to build a **computer vision model** that can automatically assess rice quality from a single tray image.

### Task
Given an image of a rice sample tray, predict **15 quality measurements simultaneously**:

| Group | Targets |
|-------|---------|
| **Grain Counts** | `Count`, `Broken_Count`, `Long_Count`, `Medium_Count`, `Black_Count`, `Chalky_Count`, `Red_Count`, `Yellow_Count`, `Green_Count` |
| **Dimensions** | `WK_Length_Average`, `WK_Width_Average`, `WK_LW_Ratio_Average` |
| **Color (LAB)** | `Average_L`, `Average_a`, `Average_b` |

### Evaluation Metric
**Mean Absolute Error (MAE)**, averaged across all 15 target columns. Lower is better.

$$\text{Score} = \frac{1}{15} \sum_{j=1}^{15} \frac{1}{n} \sum_{i=1}^{n} |y_{ij} - \hat{y}_{ij}|$$

---

## Repository Structure

```
.
├── africarice-zindi.ipynb              # Main Study Jam notebook (EDA + full pipeline)
├── preprocess_submission.ipynb         # Post-processing & domain rule corrections
├── submission_eva02_adamw_384_tta_100ep_32batch.csv           # Raw model output
├── submission_eva02_adamw_384_tta_100ep_32batch_corrected.csv # Post-processed submission
├── slides/                             # Study Jam presentation slides
│   └── africarice_study_jam_slides.pdf
└── README.md
```

---

## Dataset

The dataset is **not included** in this repository. You can download it from Kaggle:

```python
import kagglehub
path = kagglehub.dataset_download('idsalifou/dataset-africarice')
print('Path to dataset files:', path)
```

Or download manually from the [competition page](https://zindi.africa/competitions/unido-africarice-quality-assessment-challenge) and place the files as follows:

```
data/
└── dataset/
    └── Unido_AfricaRice_Challenge/
        ├── Train.csv
        ├── Test.csv
        ├── SampleSubmission.csv
        └── unido_rice_images/
            ├── IMG_001.png
            ├── IMG_002.png
            └── ...
```

---

## Solution Approach

### Model Architecture — Multi-Head Vision Transformer

The core idea is a **multi-head regression model** built on a pretrained Swin Transformer backbone. The 15 targets are split into 3 groups, each with a dedicated prediction head:

```
Input Image (384×384)
        ↓
  Swin Transformer Backbone
  (swin_base_patch4_window12_384, pretrained on ImageNet-22K)
        ↓ shared features
  ┌─────────────┬──────────────┬─────────────┐
  ↓             ↓              ↓
Head: Counts  Head: Dims   Head: Colors
 (9 outputs)  (3 outputs)  (3 outputs)
  └─────────────┴──────────────┴─────────────┘
        ↓
  Concatenate → 15 predictions
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **Swin Transformer @ 384×384** | Native resolution = no information loss from resizing |
| **ImageNet-22K pretrained weights** | More diverse pretraining → better feature transfer |
| **Multi-head output** | Counts, dimensions, and colors have fundamentally different statistics |
| **`StandardScaler` on targets** | Without scaling, `Count ≈ 1400` dominates the loss vs `LW_Ratio ≈ 0.3` |
| **`HuberLoss(delta=1.0)`** | More robust than pure MAE/L1 when some samples are outliers |
| **5-Fold Cross Validation** | More reliable OOF score than a single train/val split |
| **Mixed precision (`autocast`)** | ~2× faster training, lower VRAM usage |
| **Test Time Augmentation (TTA)** | Averaging 3 augmented views at inference reduces variance |

### Training Configuration

```python
CONFIG = {
    "model_name" : "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "img_size"   : 384,
    "batch_size" : 32,
    "epochs"     : 100,
    "optimizer"  : "AdamW",
    "lr"         : 3e-4,
    "weight_decay": 1e-2,
    "scheduler"  : "CosineAnnealingLR (eta_min=1e-7)",
    "loss"       : "HuberLoss(delta=1.0)",
    "n_fold"     : 5,
    "tta_views"  : 3,   # original + H-flip + V-flip
}
```

### Data Augmentation (Training)

```python
A.HorizontalFlip(p=0.5)        # Tray has no natural orientation
A.VerticalFlip(p=0.5)          # Same
A.RandomRotate90(p=0.5)        # Any 90° rotation is valid
A.CoarseDropout(                # Forces model to count from partial views
    max_holes=8,
    max_height=32,
    max_width=32,
    p=0.3
)
A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Post-Processing — Domain Rules (Free Points!)

The `Test.csv` contains a `Comment` column (`Paddy` / `Milled`) that encodes domain knowledge:

```python
# Rule 1: All count columns must be non-negative integers
submission[count_cols] = submission[count_cols].clip(lower=0).round().astype(int)

# Rule 2: Paddy rice has no chalky grains by definition
submission.loc[paddy_mask, 'Chalky_Count'] = 0

# Rule 3: For Paddy, total Count = sum of color sub-counts
submission.loc[paddy_mask, 'Count'] = (
    submission.loc[paddy_mask, ['Black_Count', 'Red_Count',
                                 'Yellow_Count', 'Green_Count']].sum(axis=1)
)
```

These rules improved the score **without retraining the model**.

---

## Results

| Model | Public score | Private score |
|-------|------------|-------------|
| **Swin-Base + TTA + Post-processing** | **0.8796** | **0.8686** |

> **Final leaderboard position: 16th**

---

## Requirements

```bash
pip install torch torchvision timm albumentations opencv-python-headless
pip install scikit-learn pandas numpy matplotlib seaborn tqdm kagglehub
```

**Key library versions used:**
- `torch >= 2.0`
- `timm >= 0.9`
- `albumentations >= 1.3`

---

## How to Run

### 1. Download the dataset (see above)

### 2. Run the Study Jam notebook
Open `africarice-zindi.ipynb` — it walks through the full pipeline interactively, from EDA to submission.

### 3. Post-process your submission
Open and run `preprocess_submission.ipynb` to apply domain rules to any raw model output.

---

## Study Jam

This repository was presented at a **Zindi Study Jam** session in March 2026, hosted for the AfricaRice/UNIDO challenge community.

The session covered:
1. Understanding the challenge & evaluation metric
2. Exploratory Data Analysis (EDA)
3. Building a baseline model (EfficientNet-B3)
4. Upgrading to the advanced multi-head Swin Transformer
5. Preparing and validating a submission
6. Live follow-along: generating a mean-baseline submission (no GPU needed!)

The presentation slides are available in the `slides/` folder.

---

## Lessons Learned

- **Target scaling is not optional** — a 1400× scale difference between targets will silently destroy your training
- **Multi-head models pay off** — let each head specialize in its group of targets
- **Read the data description carefully** — the Paddy domain rules were worth several MAE points for free
- **TTA is always worth trying** — 3 views at inference time costs nothing and always helps
- **Trust CV score over public leaderboard** — public score is only on ~30% of test data

---

## What I Would Try Next

- [ ] Ensemble of EfficientNet + Swin + EVA-02 (averaging diverse architectures)
- [ ] Higher input resolution (512–768) with gradient accumulation
- [ ] Pseudo-labeling on the test set
- [ ] Stratified K-Fold split by `Comment` (Paddy vs Milled) for balanced folds
- [ ] Separate models per head group, merged at inference time
- [ ] Multi-GPU distributed training

---

## Author

**Uriel Nguefack Yefou**

- 🌐 Zindi: [zindi.africa/users/urielnguefack](https://zindi.africa/users/urielnguefack)
- 💼 GitHub: [github.com/nguefackuriel](https://github.com/nguefackuriel)
- 🔗 LinkedIn: [linkedin.com/in/uriel-nguefack-yefou](https://linkedin.com/in/uriel-nguefack-yefou)

---

## License

This project is released under the [MIT License](LICENSE).  
The dataset belongs to UNIDO and AfricaRice — please refer to the [competition rules](https://zindi.africa/competitions/unido-africarice-quality-assessment-challenge) for data usage terms.

---

*⭐ If this repository helped you learn something, consider starring it!*
