# Full Stroke Prediction Pipeline with Oversampling and Explainability
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Oversampling
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler

# SHAP
import shap

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Drop id column
X = df.drop(columns=["id", "stroke"])
y = df["stroke"]

# -----------------------------
# 2. Preprocess
# -----------------------------
# Identify numeric and categorical features
num_cols = ["age", "avg_glucose_level", "bmi"]
cat_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

# Fill missing BMI
X["bmi"].fillna(X["bmi"].median(), inplace=True)

# Convert categorical features to numeric codes (needed for SMOTENC)
for col in cat_cols:
    X[col] = X[col].astype('category').cat.codes

# Scale numeric features
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# -----------------------------
# 3. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Oversampling Techniques
# -----------------------------

# === Select ONE oversampling technique ===

# 1. SMOTE
from imblearn.over_sampling import SMOTENC
cat_idx = [X.columns.get_loc(col) for col in cat_cols]
sm = SMOTENC(categorical_features=cat_idx, random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# 2. Borderline-SMOTE
# from imblearn.over_sampling import BorderlineSMOTE
# sm = BorderlineSMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(X_train, y_train)

# 3. ADASYN
# from imblearn.over_sampling import ADASYN
# sm = ADASYN(random_state=42)
# X_res, y_res = sm.fit_resample(X_train, y_train)

# 4. Random OverSampler
# from imblearn.over_sampling import RandomOverSampler
# sm = RandomOverSampler(random_state=42)
# X_res, y_res = sm.fit_resample(X_train, y_train)

print("Before oversampling:", y_train.value_counts())
print("After oversampling:", y_res.value_counts())

# -----------------------------
# 5. Model Training
# -----------------------------

# === Choose ONE model ===

# 1. Random Forest
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

# 2. Logistic Regression
# clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

# 3. SVM
# clf = SVC(probability=True, class_weight="balanced", kernel="rbf", random_state=42)

clf.fit(X_res, y_res)

# -----------------------------
# 6. Predictions & Metrics
# -----------------------------
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]  # probability for positive class

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC Score:", roc_auc)

# Precision-Recall AUC
prec, rec, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(rec, prec)
print("PR-AUC Score:", pr_auc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# 7. Feature Importance / Reason for Stroke
# -----------------------------
# If tree-based model
if hasattr(clf, "feature_importances_"):
    importances = clf.feature_importances_
    features = X.columns
    plt.figure(figsize=(8,6))
    sns.barplot(x=importances, y=features)
    plt.title("Feature Importance (Reason for Stroke)")
    plt.show()

# SHAP explanations
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_res)

# Summary plot for stroke (class 1)
shap.summary_plot(shap_values[1], X_res)

# -----------------------------
# 8. Other Charts
# -----------------------------
# Distribution of stroke in train/test
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Stroke Distribution in Dataset")
plt.show()

# Example: age vs avg_glucose_level colored by stroke
plt.figure(figsize=(6,4))
sns.scatterplot(x=X_test["age"], y=X_test["avg_glucose_level"], hue=y_test)
plt.title("Age vs Glucose Level by Stroke")
plt.show()