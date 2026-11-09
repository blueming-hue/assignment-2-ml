# Machine Learning Assignment 2: Classification Models & Streamlit Deployment

**Programme:** M.Tech in Artificial Intelligence & Machine Learning (AIML)  
**Institution:** BITS Pilani (Work Integrated Learning Programmes Division)  
**Course:** Machine Learning  
**Assignment:** Assignment - 2  

---

## a. Problem Statement

The objective of this assignment is to design, implement, evaluate, and deploy an end-to-end Machine Learning classification solution using a real-world public dataset that contains at least 12 input features and 500 instances.

Specifically, the workflow entails:
1. Training and evaluating five distinct classification algorithms on the same dataset:
   - Logistic Regression
   - Decision Tree Classifier
   - K-Nearest Neighbors (KNN)
   - Naive Bayes Classifier (GaussianNB)
   - Support Vector Machine (SVM)
2. Rigorously assessing every model across six mandatory evaluation metrics:
   - Accuracy
   - Area Under the ROC Curve (AUC)
   - Precision
   - Recall
   - F1 Score
   - Matthews Correlation Coefficient (MCC)
3. Persisting the trained model artifacts and preprocessing pipelines using `joblib`.
4. Developing an interactive, production-grade Streamlit web application that enables users to upload test data (`test_data.csv`), select any of the five trained models, inspect real-time performance cards, view interactive confusion matrices, and examine detailed classification reports without retraining the models on the test partition.
5. Deploying the application to Streamlit Community Cloud and maintaining an organized GitHub repository.

In medical oncology diagnostics, classifying breast tumors as Malignant or Benign carries significant clinical consequences: False Negatives delay critical cancer treatments, while False Positives cause unnecessary invasive biopsies and severe psychological distress. Thus, model evaluation requires balancing Precision, Recall, AUC, and MCC.

---

## b. Dataset Description

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source Link:** [UCI Machine Learning Repository (Dataset ID: 17)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) / [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Number of Instances:** 569 instances (satisfies the requirement of $\ge 500$ instances)
- **Number of Input Features:** 30 continuous numeric input features (satisfies the requirement of $\ge 12$ features)
- **Target Variable:** Binary classification (`target`: `1 = Malignant`, `0 = Benign`)
- **Class Distribution:**
  - **Class 0 (Benign):** 357 instances (62.74%)
  - **Class 1 (Malignant):** 212 instances (37.26%)
- **Data Quality & Hygiene:**
  - **Missing Values:** 0 missing values across all columns.
  - **Duplicate Records:** 0 duplicate rows.
- **Feature Characteristics:** Ten real-valued features are computed for each cell nucleus from digitized images of a Fine Needle Aspirate (FNA) of a breast mass:
  1. `radius` (mean of distances from center to contour points)
  2. `texture` (standard deviation of gray-scale values)
  3. `perimeter`
  4. `area`
  5. `smoothness` (local variation in radius lengths)
  6. `compactness` ($\text{perimeter}^2 / \text{area} - 1.0$)
  7. `concavity` (severity of concave portions of the contour)
  8. `concave points` (number of concave portions of the contour)
  9. `symmetry`
  10. `fractal dimension` ("coastline approximation" - 1.0)
  
  The Mean, Standard Error (SE), and "Worst" (mean of the three largest values) were computed for each feature, resulting in $3 \times 10 = 30$ continuous input features.
- **Data Splitting:** Stratified 80:20 Train-Test split (`random_state=42`):
  - Training Set: 455 instances (used solely for pipeline fitting)
  - Test Set: 114 instances (held out and exported as `test_data.csv`)

---

## c. GitHub Repository Link

- **GitHub Repository Link:** [https://github.com/blueming-hue/Assignment-2-](https://github.com/blueming-hue/Assignment-2-)
- **Live Streamlit App Link:** [https://assignment-2-2026aj05278.streamlit.app/](https://assignment-2-2026aj05278.streamlit.app/)

### Project Folder Structure
```text
project-folder/
│-- app.py                          # Interactive Streamlit application
│-- requirements.txt                # Python dependencies for deployment & local execution
│-- README.md                       # Comprehensive project report & documentation
│-- test_data.csv                   # Evaluation test data (114 samples × 31 columns: 30 features + target)
│-- .gitignore                      # Git ignore patterns
│-- data/
│   └── breast_cancer_complete.csv  # Full dataset (569 rows)
│-- model/
│   │-- logistic_regression.joblib  # Trained Logistic Regression pipeline
│   │-- decision_tree.joblib        # Trained Decision Tree classifier
│   │-- knn.joblib                  # Trained K-Nearest Neighbors pipeline
│   │-- naive_bayes.joblib          # Trained Gaussian Naive Bayes classifier
│   │-- svm.joblib                  # Trained Support Vector Machine pipeline
│   │-- scaler.joblib               # Fitted StandardScaler artifact
│   │-- train.py                    # Complete end-to-end training script
│   │-- train_models.ipynb          # Jupyter notebook for training and EDA
│   │-- metadata.json               # Schema metadata and class mappings
│   └── evaluation_metrics.json     # Saved metrics, confusion matrices & reports
```

---

## d. Models Used

Five classification models were implemented using Python and `scikit-learn`:
1. **Logistic Regression**
2. **Decision Tree**
3. **KNN (K-Nearest Neighbors)**
4. **Naive Bayes (GaussianNB)**
5. **Support Vector Machine (SVM)**

All five models were trained on the identical 80% training partition and evaluated strictly on the 20% test partition (`test_data.csv`, 114 samples) using fixed random seeds to ensure full reproducibility and eliminate data leakage.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9676 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed exceptionally well as a linear baseline. With feature standardization via `StandardScaler` and L2 regularization ($C=1.0$), it achieved the highest AUC (0.9960) and very strong Accuracy (96.49%) and F1 (0.9512). It missed only 3 malignant cases out of 42 (Recall 92.86%), showing that linear boundaries in standardized 30-dimensional clinical space separate the classes cleanly. |
| Decision Tree | While simple, fast, and inherently scale-invariant, the Decision Tree exhibited lower recall (83.33%, missing 7 malignant cases) and the lowest MCC (0.8299). Orthogonal axis-aligned hyperplanes struggle to capture smooth continuous boundaries, resulting in higher variance and sub-optimal generalization compared to kernel or linear methods. |
| KNN | K-Nearest Neighbors ($k=5$, Euclidean distance on standardized features) delivered strong performance (Accuracy 95.61%, F1 0.9383, MCC 0.9058). By leveraging localized topological neighborhoods, it classified non-linear sample clusters accurately. However, testing is computationally intensive ($O(N \cdot D)$ distance computations), and performance depends heavily on accurate feature scaling. |
| Naive Bayes | Gaussian Naive Bayes attained a perfect Precision of 1.0000 (0 False Positives, meaning every tumor flagged as malignant was indeed malignant) and an outstanding AUC of 0.9934. However, its Recall dropped to 83.33% because the strict conditional independence assumption is heavily violated by strong collinearities among cell geometry features (e.g., radius, perimeter, and area). |
| Support Vector Machine | SVM with an RBF kernel and standardized features emerged as the top performer overall. It achieved the highest Accuracy (97.37%), highest F1 score (0.9630), highest MCC (0.9442), and a perfect Precision of 1.0000 combined with strong Recall (92.86%). The maximum-margin principle with non-linear RBF projection provided superior generalization with the lowest error rate. |
| Overall, Winner for your dataset? | **Support Vector Machine (SVM with RBF Kernel)** is the unambiguous winner. In oncology classification, the primary clinical imperative is achieving the highest discrimination power while avoiding false alarms and missed malignancies. SVM delivered the highest Accuracy (97.37%), highest F1 Score (0.9630), highest Matthews Correlation Coefficient (0.9442), and zero False Positives (Precision 1.0000) with 92.86% Recall. |

---

## Instructions for Local Setup & Execution

### 1. Clone the Repository
```bash
git clone https://github.com/blueming-hue/Assignment-2-.git
cd Assignment-2-
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (Command Prompt / PowerShell)
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-train and Verify Models
```bash
python model/train.py
```

### 5. Run the Streamlit Application Locally
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## Instructions for Streamlit Community Cloud Deployment

1. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in using your GitHub account.
2. Click **"New App"**.
3. Select your repository: `blueming-hue/Assignment-2-`.
4. Choose the branch: `main`.
5. Set the main file path: `app.py`.
6. Click **"Deploy!"**.
7. Once the app is deployed, verify that the application loads without errors and upload `test_data.csv` to evaluate all five models interactively.
