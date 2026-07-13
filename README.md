# GTSRB — German Traffic Sign Recognition Benchmark

A lightweight CNN-based classifier for 43 classes of German road traffic signs, achieving **98.25% test accuracy** with only ~361K parameters.

## Pipeline

1. Data download & EDA (class distribution, sample visualization)
2. Preprocessing: ROI cropping → resize to 32×32 → normalize
3. Class balancing via augmentation-based oversampling
4. Logistic regression baseline (**85.46%**) → Custom CNN (**98.25%**)
5. Evaluation: confusion matrix, per-class metrics, misclassification analysis

## Requirements

Python 3 with `torch`, `pandas`, `scikit-learn`, `seaborn`, `Pillow`, `kagglehub`

## Usage

Run the notebook cells in order (`notebook.ipynb`). The dataset is downloaded automatically from Kaggle.
