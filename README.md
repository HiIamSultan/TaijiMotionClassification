# TaijiMotionClassification

## Overview
This repository contains the implementation of classification models for human motion dataset analysis using both traditional machine learning methods and deep learning approaches. The project focuses on evaluating the impact of feature selection, dimensionality reduction using Linear Discriminant Analysis (LDA), and classifier performance comparisons.

## Project Structure
```
├── classification_final.py            # Main script for classification tasks
├── classification_starter_final.py    # Earlier version of classification script
├── classification_starter_initial.py  # Initial version with minimal preprocessing
├── model_starter.py                   # Traditional machine learning model implementation
├── Taiji_dataset_100.csv               # Dataset with N=100, dataset can be given upon request
├── Taiji_dataset_200.csv               # Dataset with N=200, dataset can be given upon request
├── Taiji_dataset_300.csv               # Dataset with N=300, dataset can be given upon request
├── plots_related_code/                 # Scripts for generating analysis plots
│   ├── averages.py                     # Computes average classification metrics
│   ├── best_class.py                    
│   ├── best_classifier.py
│   ├── best_dataset.py
│   ├── best_feature.py
│   ├── best_subject.py
│   ├── plots1.py
├── results/                            # Stores generated confusion matrices
├── logs/                               # Contains log files from model training (if applicable)
├── README.md                           # Project documentation
```

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
```sh
pip install numpy pandas scikit-learn matplotlib torch
```

## Usage
### Running Classification Models
To run the main classification script, execute:
```sh
python classification_final.py
```
### Dataset Selection
Modify the dataset in the script by setting:
```python
dataset_name = "Taiji_dataset_200.csv"  # Change to the desired dataset file
```

### Running Leave-One-Subject-Out (LOSO) Cross-Validation
The classification scripts automatically implement LOSO cross-validation. Ensure datasets are correctly placed.

## Feature Selection & Dimensionality Reduction
The implementation includes:
- **Filter Methods**: Variance Ratio, Augmented Variance Ratio (AVR), Minimum Redundancy Maximum Relevance (mRMR).
- **Wrapper Methods**: Sequential Forward Selection.
- **Dimensionality Reduction**: Linear Discriminant Analysis (LDA).

## Model Evaluation
After training, the results include:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrices (stored in `results/` directory)
- Plots for classifier performance comparisons

