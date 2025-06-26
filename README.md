# Vit-super-token-sampling
Vision Transformer enhanced with Super Token Sampling, RoPE, Conv-MLP, and full training pipeline with evaluation and visualizations.

# 🔬 Vision Transformer with Super Token Sampling (ViT-STS)

This repository contains a custom implementation of a **Vision Transformer** enhanced with:

- 🧩 Multi-scale patch embedding
- 📐 Relative positional embeddings (RoPE)
- 🧠 Conv-based MLP layers
- 🌟 **Super Token Sampling (STS)** for token selection
- 📊 Extensive evaluation pipeline with classification reports, confusion matrices, and model statistics

## 🚀 Highlights

- ✅ Customizable ViT depth, head size, patch size
- 📦 Full training, validation, and testing loop with checkpointing
- 🧠 Advanced metrics: Top-1, Top-3, Precision, Recall, F1, AUC
- 📉 Visualizations: Loss curves, accuracy plots, confusion matrices
- 📊 Outputs:
- `trained_model.pth`
- `results.json`, `detailed_metrics.txt`, `classification_report.txt`
- `.png` plots for loss & confusion matrices

## 📁 Project Structure
vit-super-token-sampling/
├── Sts.py
├── Images/ or SoyMCData/
├── results/
├── README.md
├── requirements.txt
└── .gitignore

## 🛠️ Installation
pip install torch torchvision matplotlib seaborn scikit-learn tqdm ptflops

📈 How to Run
Organize your dataset as:
SoyMCData/
├── train/
├── val/
└── test/
Update save_dir in Sts.py.

Run training:
python Sts.py

🖼️ Example Outputs
loss_curves.png
top1_top3_comparison.png
test_confusion_matrix.png
model_statistics.json
trained_model.pth

