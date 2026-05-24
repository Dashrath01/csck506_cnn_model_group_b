# Chest X-Ray Pneumonia Detection Suite from Group B

A comprehensive, production-ready deep learning repository for the binary classification of Pneumonia vs. Normal chest X-rays. This suite spans foundational CNN architectures, advanced Transfer Learning, Explainable AI (XAI), and rigorous statistical model testing.

## Overview
This repository addresses the primary challenges of medical image classification—class imbalance, overfitting, and the "black-box" nature of deep learning. It provides a complete workflow: from exploratory custom CNNs and class-weighting strategies to an advanced **DenseNet121** pipeline, culminating in a robust statistical testing suite to directly compare model efficacies.

## Repository Contents (Notebooks)

### 1. Advanced Transfer Learning & XAI (`transfer-learning-dense_net_cnn_DB.ipynb`)
The flagship model of the repository, featuring a highly tuned two-stage pipeline:
* **Two-Stage Fine-Tuning:** * *Stage 1:* Custom head training (Adam, LR=$10^{-3}$) dynamically controlled via `ReduceLROnPlateau` (factor=0.3, min_lr=$10^{-6}$) and `EarlyStopping` (patience=6).
  * *Stage 2:* Mathematical unfreezing of the final Dense Block (starting precisely at layer `conv5_block1_0_bn`) using a microscopic learning rate ($10^{-5}$) to capture radiographic textures without catastrophic forgetting.
* **Overfitting Mitigation:** Relaxed L2 Regularisation ($\lambda = 10^{-4}$), $0.5$ Dropout, and domain-aware augmentation.
* **Class Balancing:** Algorithmic class weights (Normal: 1.94, Pneumonia: 0.67).
* **Explainability (Grad-CAM):** Utilises strict mathematical thresholding (top 30% activations only) and pure red perceptual overlays to pinpoint pulmonary opacities.

### 2. Custom Augmented CNN Pipeline (`pneumonia_cnn_updated_CE.ipynb`)
A robust baseline model built entirely from scratch to establish performance benchmarks.
* **Architecture:** A Keras `Sequential` model utilising deep `Conv2D`, `MaxPool2D`, and `BatchNormalization` layers.
* **Integrated Preprocessing:** Implements real-time spatial augmentations (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`) directly within the model graph.
* **Rapid Iteration:** Employs aggressive `EarlyStopping` (patience=3) across a 12-epoch run to quickly isolate the best weights and prevent immediate overfitting on the augmented data.

### 3. Foundational CNN & Imbalance Analysis (`CNN_Model_EN.ipynb`)
An exploratory notebook focused on dataset dynamics and foundational strategies.
* **Data Ingestion:** Implements automated Kaggle API setup and dataset extraction natively within the notebook.
* **Class Imbalance Handling:** Explores architectural and algorithmic responses to the dataset's inherent imbalance (e.g., 390 Pneumonia vs. 234 Normal images in the test set) through oversampling/undersampling theory and class weighting.
* **Ensemble Theory:** Establishes the framework for combining multiple architectural predictions to create more resilient diagnostic outputs.

### 4. Statistical Model Testing Suite (`Test_AMR.ipynb`)
A rigorous evaluation battery designed to directly compare the clinical viability of two distinct CNN models (e.g., the CE Custom Model vs. the DB DenseNet Model).
* **Testing Principle:** Evaluates models side-by-side on raw, untouched test images, ensuring each model utilises its natively trained preprocessing pipeline.
* **Statistical Significance:** Executes **McNemar's Test** to generate p-values, mathematically proving whether one model's predictions are statistically superior to the other's.
* **Advanced Evaluation Metrics:** Automatically calculates and contrasts Accuracy differentials, F1 Score differentials, Balanced Accuracy, and Brier Scores.
* **Automated Reporting:** Generates and saves a comprehensive visual summary graphic (`amr_summary_report_wrapped.png`).

## Usage Guide
Ensure the [Kaggle Chest X-Ray Dataset (Paul Mooney)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) is downloaded and base paths are configured.

1. **Establish Baselines:** Run `CNN_Model_EN.ipynb` and `pneumonia_cnn_updated_CE.ipynb` to understand the data distribution and train the custom CNN architectures.
2. **Train the Expert Model:** Execute `transfer-learning-dense_net_cnn_DB.ipynb` sequentially to train the highly-regulated DenseNet model and generate Grad-CAM clinical visualisations.
3. **Compare Models:** Load your trained models (`.keras` or `.h5` files) into `Test_AMR.ipynb` to run the statistical testing battery and determine the definitive best-performing network.

## 🛠️ Requirements
* Python 3.8+
* TensorFlow 2.21.0 / Keras
* NumPy, Pandas, Matplotlib, Seaborn
* Scikit-Learn, Statsmodels (for McNemar's testing)