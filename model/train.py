"""
================================================================================
Machine Learning Assignment 2 - Model Training & Evaluation Script
Program: M.Tech (AIML) - BITS Pilani (WILP)
Course: Machine Learning

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: UCI Machine Learning Repository / Kaggle
- Number of Instances: 569 (>= 500 requirement met)
- Number of Input Features: 30 (>= 12 requirement met)
- Target: Binary (1: Malignant, 0: Benign)

This script:
1. Loads and prepares the dataset.
2. Checks for missing values, duplicates, and class distribution.
3. Splits data into Train (80%) and Test (20%) using stratified split (random_state=42).
4. Saves 'test_data.csv' in the project root for Streamlit app evaluation.
5. Trains all five required classification models:
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbors Classifier (KNN)
   - Naive Bayes Classifier (GaussianNB)
   - Support Vector Machine (SVM)
6. Calculates 6 evaluation metrics for each model:
   - Accuracy, AUC, Precision, Recall, F1 Score, MCC
7. Exports saved models to 'model/' directory.
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

def run_training_pipeline():
    print("=" * 70)
    print("STEP 1: DATASET INGESTION & EXPLORATORY CHECKS")
    print("=" * 70)
    
    # 1. Load dataset
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    
    # In scikit-learn's default target: 0 = malignant, 1 = benign.
    # To conform with medical intuition, we invert: 1 = Malignant (positive), 0 = Benign (negative).
    df['target'] = 1 - df['target']
    
    print(f"Total instances (rows)   : {df.shape[0]}")
    print(f"Total columns (features) : {df.shape[1] - 1} input features + 1 target column")
    print(f"Missing values count     : {df.isnull().sum().sum()}")
    print(f"Duplicate rows count     : {df.duplicated().sum()}")
    
    target_dist = df['target'].value_counts()
    print("\nTarget Class Distribution:")
    print(f"  Class 0 (Benign)    : {target_dist[0]} ({target_dist[0] / len(df) * 100:.2f}%)")
    print(f"  Class 1 (Malignant) : {target_dist[1]} ({target_dist[1] / len(df) * 100:.2f}%)")
    
    feature_cols = [col for col in df.columns if col != 'target']
    
    # Save complete dataset to data folder
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/breast_cancer_complete.csv', index=False)
    print("\nSaved complete dataset to 'data/breast_cancer_complete.csv'")
    
    # 2. Train-Test Split (80% Train, 20% Test)
    X = df[feature_cols]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"\nTrain set shape : {X_train.shape[0]} instances, {X_train.shape[1]} features")
    print(f"Test set shape  : {X_test.shape[0]} instances, {X_test.shape[1]} features")
    
    # 3. Create test_data.csv containing both input features and the actual target column
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv('test_data.csv', index=False)
    print("Saved evaluation test data with actual target column to 'test_data.csv'")
    
    # 4. Define and Train the 5 Required Classification Algorithms
    print("\n" + "=" * 70)
    print("STEP 2: MODEL DEFINITIONS & TRAINING")
    print("=" * 70)
    
    # Pipelines encapsulate preprocessing (StandardScaler) with the classifier to avoid data leakage
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000, C=1.0))
        ]),
        'Decision Tree': Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=4, criterion='gini'))
        ]),
        'KNN': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2))
        ]),
        'Naive Bayes': Pipeline([
            ('classifier', GaussianNB())
        ]),
        'Support Vector Machine': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        ])
    }
    
    file_mapping = {
        'Logistic Regression': 'logistic_regression.joblib',
        'Decision Tree': 'decision_tree.joblib',
        'KNN': 'knn.joblib',
        'Naive Bayes': 'naive_bayes.joblib',
        'Support Vector Machine': 'svm.joblib'
    }
    
    os.makedirs('model', exist_ok=True)
    
    metrics_records = []
    reports = {}
    conf_matrices = {}
    
    for name, pipeline in models.items():
        print(f"\nTraining [{name}]...")
        # Fit pipeline strictly on training data
        pipeline.fit(X_train, y_train)
        
        # Predictions and probabilities on test data
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate 6 mandatory evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'], output_dict=True)
        
        conf_matrices[name] = cm.tolist()
        reports[name] = cr
        
        record = {
            'ML Model Name': name,
            'Accuracy': round(float(acc), 4),
            'AUC': round(float(auc), 4),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'F1': round(float(f1), 4),
            'MCC': round(float(mcc), 4)
        }
        metrics_records.append(record)
        
        # Save pipeline artifact
        model_save_path = os.path.join('model', file_mapping[name])
        joblib.dump(pipeline, model_save_path)
        print(f"  -> Model saved to: {model_save_path}")
    
    # Also save standalone scaler and metadata for flexibility
    fitted_scaler = StandardScaler().fit(X_train)
    joblib.dump(fitted_scaler, os.path.join('model', 'scaler.joblib'))
    
    meta_info = {
        'feature_names': feature_cols,
        'target_col': 'target',
        'class_names': ['Benign', 'Malignant'],
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    with open(os.path.join('model', 'metadata.json'), 'w') as f:
        json.dump(meta_info, f, indent=2)
        
    metrics_df = pd.DataFrame(metrics_records)
    with open(os.path.join('model', 'evaluation_metrics.json'), 'w') as f:
        json.dump({
            'metrics_table': metrics_records,
            'confusion_matrices': conf_matrices,
            'classification_reports': reports
        }, f, indent=2)
    
    print("\n" + "=" * 70)
    print("STEP 3: FINAL EVALUATION METRICS COMPARISON TABLE")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    print("=" * 70)
    print("\nTraining completed successfully! All artifacts generated.")

if __name__ == '__main__':
    run_training_pipeline()
