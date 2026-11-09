"""
================================================================================
Machine Learning Assignment 2 - Streamlit Application
M.Tech (AIML) - BITS Pilani (WILP)
Course: Machine Learning

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Features: 30 continuous clinical features (>= 12 required)
Instances: 569 instances (>= 500 required)
Models Evaluated:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. K-Nearest Neighbor Classifier (KNN)
  4. Naive Bayes Classifier (GaussianNB)
  5. Support Vector Machine (SVM)

Metrics: Accuracy, AUC Score, Precision, Recall, F1 Score, MCC
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ML Classifier Suite | Breast Cancer Wisconsin",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #1E40AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
    }
    .info-box {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONSTANTS & MODEL DEFINITIONS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_TEST_PATH = os.path.join(BASE_DIR, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Linear classification model using a logistic sigmoid link function with L2 regularization and feature standardization.",
    "Decision Tree": "Non-linear decision tree partitioned via Gini impurity with max depth constrained to prevent overfitting.",
    "KNN": "Instance-based non-parametric classifier predicting via Euclidean distance among 5 nearest standardized neighbors.",
    "Naive Bayes": "Probabilistic classifier applying Bayes' Theorem with Gaussian distribution assumptions across continuous features.",
    "Support Vector Machine": "Max-margin classifier projecting features into high-dimensional space via Radial Basis Function (RBF) kernel."
}

# -----------------------------------------------------------------------------
# RESOURCE CACHING (LOAD TRAINED MODELS)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    """Load all pre-trained pipeline models from disk without retraining."""
    loaded = {}
    for name, filename in MODEL_FILES.items():
        filepath = os.path.join(MODEL_DIR, filename)
        if os.path.exists(filepath):
            loaded[name] = joblib.load(filepath)
    return loaded

@st.cache_data
def load_metadata():
    """Load dataset metadata such as feature names and target column."""
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return None

models_dict = load_all_models()
metadata = load_metadata()
expected_features = metadata.get("feature_names", []) if metadata else []
target_col = metadata.get("target_col", "target") if metadata else "target"

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg", width=120)
st.sidebar.title("Configuration")

st.sidebar.markdown("### 1. Test Dataset Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ("Use Default Test Dataset (114 samples)", "Upload Custom CSV File")
)

test_df = None

if data_source == "Use Default Test Dataset (114 samples)":
    if os.path.exists(DEFAULT_TEST_PATH):
        test_df = pd.read_csv(DEFAULT_TEST_PATH)
        st.sidebar.success(f"Loaded default test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
    else:
        st.sidebar.error("Default test_data.csv not found on server.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Test Data CSV (must include 30 features + target)",
        type=["csv"]
    )
    if uploaded_file is not None:
        try:
            test_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Uploaded CSV loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Model Selection")
model_options = list(MODEL_FILES.keys()) + ["Compare All 5 Models"]
selected_model_option = st.sidebar.selectbox("Select Classification Model:", model_options, index=0)

st.sidebar.markdown("---")
st.sidebar.caption("BITS Pilani WILP - M.Tech (AIML)\nMachine Learning Assignment 2")

# -----------------------------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Comparative Machine Learning Classification Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evaluation of 5 Classical ML Classifiers on Breast Cancer Wisconsin (Diagnostic) Dataset</div>', unsafe_allow_html=True)

# Dataset validation check
if test_df is None:
    st.info("👈 Please select or upload a test dataset using the sidebar to begin evaluation.")
    st.stop()

# Validate features and target
missing_features = [f for f in expected_features if f not in test_df.columns]
has_target = target_col in test_df.columns

if missing_features or not has_target:
    st.error("Uploaded test data failed schema validation:")
    if not has_target:
        st.error(f"- Missing target column: `{target_col}`. Test CSV must contain both input features and the actual ground truth target.")
    if missing_features:
        st.error(f"- Missing {len(missing_features)} required input feature(s): {', '.join(missing_features[:5])}...")
    st.info("Tip: Please download and examine `test_data.csv` from the repository for the exact required format.")
    st.stop()

X_test = test_df[expected_features]
y_test = test_df[target_col]

with st.expander("🔍 View Test Dataset Overview & Feature Distribution"):
    st.write(f"**Shape:** {test_df.shape[0]} samples × {test_df.shape[1]} columns")
    class_counts = y_test.value_counts().to_dict()
    benign_count = class_counts.get(0, 0)
    malignant_count = class_counts.get(1, 0)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Test Instances", len(y_test))
    col_b.metric("Class 0 (Benign)", f"{benign_count} ({benign_count/len(y_test)*100:.1f}%)")
    col_c.metric("Class 1 (Malignant)", f"{malignant_count} ({malignant_count/len(y_test)*100:.1f}%)")
    st.dataframe(test_df.head(5), use_container_width=True)

# Helper function to compute all 6 metrics
def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X)
    else:
        y_proba = y_pred

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred, target_names=["Benign", "Malignant"], output_dict=True)

    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "MCC": mcc,
        "Confusion Matrix": cm,
        "Classification Report": cr,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

# -----------------------------------------------------------------------------
# SCENARIO A: COMPARE ALL 5 MODELS
# -----------------------------------------------------------------------------
if selected_model_option == "Compare All 5 Models":
    st.markdown("### 📊 Comprehensive Model Performance Comparison")
    st.markdown("All five pre-trained classifiers evaluated directly on the uploaded test set without retraining:")

    comparison_records = []
    all_results = {}

    for model_name, model in models_dict.items():
        res = compute_metrics(model, X_test, y_test)
        all_results[model_name] = res
        comparison_records.append({
            "ML Model Name": model_name,
            "Accuracy": round(res["Accuracy"], 4),
            "AUC": round(res["AUC"], 4),
            "Precision": round(res["Precision"], 4),
            "Recall": round(res["Recall"], 4),
            "F1": round(res["F1"], 4),
            "MCC": round(res["MCC"], 4)
        })

    comp_df = pd.DataFrame(comparison_records)
    
    # Highlight best in each column
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #DCFCE7; font-weight: bold;' if v else '' for v in is_max]

    numeric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    styled_df = comp_df.style.apply(highlight_max, subset=numeric_cols).format({c: "{:.4f}" for c in numeric_cols})
    st.dataframe(styled_df, use_container_width=True)

    # Multi-metric visual bar chart
    st.markdown("#### Graphical Comparison of Key Metrics")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bar_df = comp_df.set_index("ML Model Name")[numeric_cols]
    bar_df.plot(kind="bar", ax=ax, colormap="viridis", width=0.75)
    plt.title("Performance Comparison Across All 6 Evaluation Metrics", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Score", fontsize=11)
    plt.xlabel("")
    plt.ylim(0.7, 1.02)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# SCENARIO B: SINGLE MODEL DETAILED VIEW
# -----------------------------------------------------------------------------
else:
    model_name = selected_model_option
    model = models_dict[model_name]

    st.markdown(f"### ⚙️ Selected Model: **{model_name}**")
    st.markdown(f"*{MODEL_DESCRIPTIONS.get(model_name, '')}*")

    # Compute metrics for selected model
    metrics = compute_metrics(model, X_test, y_test)

    # 6 Metric Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Accuracy</div>
            <div class="metric-val">{metrics['Accuracy']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AUC Score</div>
            <div class="metric-val">{metrics['AUC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Precision</div>
            <div class="metric-val">{metrics['Precision']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Recall</div>
            <div class="metric-val">{metrics['Recall']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">F1 Score</div>
            <div class="metric-val">{metrics['F1']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MCC Score</div>
            <div class="metric-val">{metrics['MCC']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for Visualizations
    tab1, tab2, tab3 = st.tabs(["🔲 Confusion Matrix", "📋 Classification Report", "📈 ROC Curve & Probabilities"])

    with tab1:
        st.markdown("#### Confusion Matrix")
        col_cm, col_cm_desc = st.columns([1.2, 1])

        with col_cm:
            cm = metrics["Confusion Matrix"]
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Benign (0)", "Malignant (1)"],
                yticklabels=["Benign (0)", "Malignant (1)"],
                cbar=False,
                annot_kws={"size": 13, "weight": "bold"}
            )
            plt.ylabel("Actual Label", fontsize=10, fontweight='bold')
            plt.xlabel("Predicted Label", fontsize=10, fontweight='bold')
            plt.title(f"Confusion Matrix: {model_name}", fontsize=11, fontweight='bold', pad=10)
            plt.tight_layout()
            st.pyplot(fig)

        with col_cm_desc:
            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            **Matrix Breakdown:**
            - **True Negatives (TN):** `{tn}` correctly classified as Benign.
            - **False Positives (FP):** `{fp}` Benign erroneously flagged as Malignant.
            - **False Negatives (FN):** `{fn}` Malignant cases missed (critical error).
            - **True Positives (TP):** `{tp}` correctly identified as Malignant.
            
            *In clinical diagnostics, minimizing False Negatives (high Recall) is critical to prevent missed malignant tumors.*
            """)

    with tab2:
        st.markdown("#### Detailed Classification Report")
        cr_dict = metrics["Classification Report"]
        cr_df = pd.DataFrame(cr_dict).transpose()
        st.dataframe(cr_df.style.format("{:.4f}"), use_container_width=True)

    with tab3:
        st.markdown("#### ROC Curve Analysis")
        col_roc, col_dist = st.columns(2)
        
        with col_roc:
            fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
            fig, ax = plt.subplots(figsize=(5, 3.8))
            ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {metrics['AUC']:.4f})")
            ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=1.5, linestyle="--", label="Random Classifier")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate", fontsize=10)
            ax.set_ylabel("True Positive Rate", fontsize=10)
            ax.set_title(f"ROC Curve - {model_name}", fontsize=11, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with col_dist:
            fig, ax = plt.subplots(figsize=(5, 3.8))
            prob_df = pd.DataFrame({
                "Predicted Probability": metrics["y_proba"],
                "Actual Class": ["Malignant" if y == 1 else "Benign" for y in y_test]
            })
            sns.histplot(
                data=prob_df,
                x="Predicted Probability",
                hue="Actual Class",
                element="step",
                stat="density",
                common_norm=False,
                palette={"Benign": "#10B981", "Malignant": "#EF4444"},
                ax=ax
            )
            ax.set_title("Predicted Probability Distribution by True Class", fontsize=11, fontweight='bold')
            ax.set_xlabel("P(Class = Malignant)", fontsize=10)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

# -----------------------------------------------------------------------------
# FOOTER & ACADEMIC INTEGRITY NOTICE
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<small>BITS Pilani Work Integrated Learning Programmes (WILP) | M.Tech (AIML) Machine Learning Assignment 2<br>"
    "Original work implementing 5 classification models and comparative diagnostic evaluation.</small>",
    unsafe_allow_html=True
)
