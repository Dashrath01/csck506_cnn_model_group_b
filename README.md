# DenseNet-Pneumonia-Detector

An advanced, production-ready deep learning pipeline for the binary classification of Pneumonia vs. Normal chest X-rays, built using TensorFlow/Keras.

## Overview
This repository addresses the primary challenges of medical image classification (class imbalance, overfitting, and the "black-box" nature of deep learning) by employing a two-stage Transfer Learning methodology atop the **DenseNet121** architecture, paired with Explainable AI (XAI) visualisations.

## Key Technical Features
* **Architecture:** DenseNet121 (Pre-trained on ImageNet).
* **Two-Stage Fine-Tuning:** Initial head mapping followed by deep-layer unfreezing with a microscopic learning rate ($10^{-5}$) to prevent catastrophic forgetting.
* **Overfitting Mitigation:** Incorporates L2 Regularisation ($\lambda = 0.001$), $0.5$ Dropout, and domain-aware data augmentation (preserving anatomical left/right validity).
* **Dynamic Optimisation:** Uses `ReduceLROnPlateau` to monitor validation gradients and automatically adjust the learning rate to ensure smooth convergence.
* **Class Balancing:** Algorithmic class weights applied during the gradient descent phase to heavily penalise False Positives, dramatically improving the clinical specificity of the model.
* **Explainable AI (Grad-CAM):** Integrated focal heatmap visualisations. Utilises strict mathematical thresholding and pure red perceptual overlays to pinpoint pulmonary opacities, ensuring clinical trust.

## Pipeline Structure
1. `preprocessing.py` / Data Block: Initialises `ImageDataGenerator` with DenseNet-specific pixel normalisation.
2. `Model Construction`: Assembles the frozen base and regularised custom head.
3. `Stage 1 Training`: Rapid convergence of the dense layers using a standard learning rate.
4. `Stage 2 Fine-Tuning`: Unfreezing the top 15 convolutional layers for domain-specific feature extraction.
5. `Evaluation`: Generation of Accuracy/Loss curves, Scikit-Learn Classification Reports, and Seaborn Confusion Matrices.
6. `Explainability (Grad-CAM)`: Generates thresholded heatmap overlays on test images to verify the anatomical regions driving the model's predictions.

## Usage
To run the pipeline, ensure the Kaggle Chest X-Ray dataset is downloaded and the paths (`TRAIN_DIR`, `TEST_DIR`) are configured in the notebook.

Execute the Jupyter Notebook sequentially. The final cells will evaluate the model, generate the focal Grad-CAM visualisations, and export the fully trained model as `pneumonia_densenet_v2_finetuned.keras` for immediate deployment.

## Requirements
* Python 3.8+
* TensorFlow 2.x
* NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn
