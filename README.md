```markdown
# Pneumonia Detection from Chest X-Rays using Deep Learning

This repository contains a modular deep learning pipeline designed to classify chest radiographic images (X-rays) into two categories: **Normal** and **Pneumonia**. 

The project demonstrates the iterative development of Convolutional Neural Networks (CNNs), starting from a shallow baseline model and culminating in a highly robust Transfer Learning architecture utilising DenseNet121.

## Dataset
The dataset utilised is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle. It contains 5,863 JPEG images across three initial splits (train, test, val). 

**Note on Validation:** To ensure statistical significance and prevent validation metric instability, this pipeline programmatically merges the training data and performs a robust 80/20 training/validation split, bypassing the dataset's notoriously small default validation folder.

## Prerequisites
Ensure you have Python 3.8+ installed along with the following dependencies:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn kagglehub

```

## Key Features & Pipeline Modules

### 1. Automated Data Acquisition

The project utilises `kagglehub` to seamlessly download and extract the dataset, eliminating the need for manual API key management or local zip extraction.

### 2. Modular Architecture Factory

The codebase separates architectural construction from the training logic, allowing for rapid experimentation. The models included are:

* **Basic CNN:** A shallow baseline (1 Conv block) for rapid prototyping.
* **Medium CNN:** Increased depth (3 Conv blocks) to capture complex spatial hierarchies.
* **Advanced CNN:** A 4-block network utilising L2 Regularisation ($\lambda = 10^{-4}$), Batch Normalisation, and heavy Dropout ($0.6$) to combat overfitting. It also employs **Class Weights** to mitigate the dataset's heavy bias towards the Pneumonia class.
* **Transfer Learning (DenseNet121):** The flagship model of this repository. It leverages a pre-trained DenseNet base (frozen) with a custom classification head, achieving state-of-the-art precision and recall.

### 3. Advanced Evaluation & Tuning

The evaluation suite automatically generates:

* Training vs. Validation Accuracy and Loss plots.
* Scikit-learn Classification Reports (Precision, Recall, F1-Score).
* Seaborn Confusion Matrices.
* **Decision Threshold Tuning:** Scripts to interactively adjust the classification threshold (e.g., from `0.5` to `0.7`) to optimise the trade-off between True Negatives and False Positives without requiring model retraining.

## Usage Instructions

### Step 1: Download Data and Configure Paths

Run the initial setup to download the data and instantiate the `tf.data.Dataset` objects with an 80/20 split and performance prefetching (`tf.data.AUTOTUNE`).

### Step 2: Build and Train a Model

Call the model factory and the training wrapper. For example, to train the DenseNet Transfer Learning model:

```python
# Build
model_transfer = build_transfer_model(dropout_rate=0.4)

# Train
history_transfer = compile_and_train(
    model_transfer, 
    train_ds, 
    val_ds, 
    learning_rate=0.001, 
    epochs=15
)

```

### Step 3: Evaluate Results

Pass the trained model and history object into the evaluation function to generate the performance metrics and visualisations.

```python
evaluate_model(model_transfer, "DenseNet121_Transfer", history_transfer)

```

## Key Findings

* **Overfitting:** Custom CNNs trained from scratch on this dataset exhibit rapid overfitting. Regularisation (L2, Dropout) successfully smooths the loss curves but reveals a secondary issue: Majority Class Bias.
* **Class Collapse:** Due to class imbalance, basic models often predict "Pneumonia" for almost all inputs, resulting in a high False Positive rate.
* **Transfer Learning Superiority:** The DenseNet121 architecture, combined with Decision Threshold tuning ($\tau = 0.7$), provides the optimal solution. It significantly increases True Negatives (correctly identifying healthy patients) while maintaining near-perfect True Positives (identifying pneumonia).

```

```