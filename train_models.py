"""
ML Model Training Script for JAMB Score Analytics
Trains regression, classification, and clustering models
Saves models as pickle files for Streamlit app usage
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, silhouette_score
import xgboost as xgb

print("=" * 70)
print("JAMB SCORE ANALYTICS - ML MODEL TRAINING")
print("=" * 70)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("\n[1/8] Loading data...")
df = pd.read_csv("jamb_exam_results.csv")

# Create a copy for processing
data = df.copy()

# Define categorical and numeric columns
categorical_cols = ['School_Type', 'School_Location', 'Extra_Tutorials', 
                   'Access_To_Learning_Materials', 'Parent_Involvement', 
                   'IT_Knowledge', 'Gender', 'Socioeconomic_Status', 'Parent_Education_Level']
numeric_cols = ['Study_Hours_Per_Week', 'Attendance_Rate', 'Teacher_Quality', 
               'Distance_To_School', 'Age', 'Assignments_Completed']

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    print(f"  ✓ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Prepare features and target
X = data.drop(['JAMB_Score', 'Student_ID'], axis=1)
y = data['JAMB_Score']

print(f"\n  ✓ Features shape: {X.shape}")
print(f"  ✓ Target shape: {y.shape}")
print(f"  ✓ Target range: {y.min()} - {y.max()}")

# Scale features
print("\n[2/8] Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Save scaler and encoders
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("  ✓ Scaler and encoders saved")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"  ✓ Train set: {X_train.shape}, Test set: {X_test.shape}")

# ============================================================================
# REGRESSION MODELS - Predict exact JAMB score
# ============================================================================

print("\n" + "=" * 70)
print("REGRESSION MODELS - Predict Exact JAMB Score")
print("=" * 70)

# Linear Regression
print("\n[3/8] Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
print(f"  ✓ R² Score: {lr_r2:.4f}")
print(f"  ✓ MAE: {lr_mae:.2f}")
print(f"  ✓ RMSE: {lr_rmse:.2f}")

with open('models/linear_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

# Random Forest Regressor
print("\n[4/8] Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print(f"  ✓ R² Score: {rf_r2:.4f}")
print(f"  ✓ MAE: {rf_mae:.2f}")
print(f"  ✓ RMSE: {rf_rmse:.2f}")

with open('models/random_forest_regressor.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# XGBoost Regressor
print("\n[5/8] Training XGBoost Regressor...")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, 
                             random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
print(f"  ✓ R² Score: {xgb_r2:.4f}")
print(f"  ✓ MAE: {xgb_mae:.2f}")
print(f"  ✓ RMSE: {xgb_rmse:.2f}")

with open('models/xgboost_regressor.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# ============================================================================
# CLASSIFICATION MODELS - Classify performance tier
# ============================================================================

print("\n" + "=" * 70)
print("CLASSIFICATION MODELS - Classify Performance Tier")
print("=" * 70)

# Create performance tiers: Low (<150), Medium (150-220), High (>220)
y_class = pd.cut(y, bins=[0, 150, 220, 500], labels=['Low', 'Medium', 'High'])
le_target = LabelEncoder()
y_class_encoded = le_target.fit_transform(y_class)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y_class_encoded, test_size=0.2, random_state=42
)

print(f"\nPerformance Classes: {np.unique(y_class, return_counts=True)}")

# Random Forest Classifier
print("\n[6/8] Training Random Forest Classifier...")
rfc_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rfc_model.fit(X_train_c, y_train_c)
rfc_pred = rfc_model.predict(X_test_c)
rfc_acc = accuracy_score(y_test_c, rfc_pred)
rfc_f1 = f1_score(y_test_c, rfc_pred, average='weighted')
print(f"  ✓ Accuracy: {rfc_acc:.4f}")
print(f"  ✓ F1 Score: {rfc_f1:.4f}")

with open('models/random_forest_classifier.pkl', 'wb') as f:
    pickle.dump(rfc_model, f)

# Logistic Regression (Binary: Pass/Fail)
print("\n[7/8] Training Logistic Regression (Binary Classification)...")
y_binary = (y >= 150).astype(int)  # 1 = Pass (>=150), 0 = Fail (<150)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42
)

lr_class_model = LogisticRegression(max_iter=1000, random_state=42)
lr_class_model.fit(X_train_b, y_train_b)
lr_class_pred = lr_class_model.predict(X_test_b)
lr_class_acc = accuracy_score(y_test_b, lr_class_pred)
lr_class_f1 = f1_score(y_test_b, lr_class_pred)
print(f"  ✓ Accuracy: {lr_class_acc:.4f}")
print(f"  ✓ F1 Score: {lr_class_f1:.4f}")

with open('models/logistic_regression_binary.pkl', 'wb') as f:
    pickle.dump(lr_class_model, f)

# Save class labels
with open('models/class_labels.pkl', 'wb') as f:
    pickle.dump({
        'target_encoder': le_target,
        'binary_labels': ['Fail (<150)', 'Pass (≥150)']
    }, f)

# ============================================================================
# UNSUPERVISED LEARNING
# ============================================================================

print("\n" + "=" * 70)
print("UNSUPERVISED LEARNING - Clustering & Dimensionality Reduction")
print("=" * 70)

# K-Means Clustering
print("\n[8/8] Training K-Means Clustering...")
# Find optimal number of clusters using elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Use k=4 as optimal
optimal_k = 4
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_model.fit(X_scaled)
silhouette_avg = silhouette_score(X_scaled, kmeans_model.labels_)

print(f"  ✓ Optimal clusters: {optimal_k}")
print(f"  ✓ Silhouette Score: {silhouette_avg:.4f}")
print(f"  ✓ Cluster distribution: {np.bincount(kmeans_model.labels_)}")

with open('models/kmeans_clusterer.pkl', 'wb') as f:
    pickle.dump(kmeans_model, f)

# PCA for dimensionality reduction
pca_model = PCA(n_components=3)
X_pca = pca_model.fit_transform(X_scaled)
explained_var = pca_model.explained_variance_ratio_.sum()

print(f"  ✓ PCA variance explained (3 components): {explained_var:.4f}")
print(f"  ✓ Component variance ratios: {pca_model.explained_variance_ratio_}")

with open('models/pca_model.pkl', 'wb') as f:
    pickle.dump(pca_model, f)

# ============================================================================
# SAVE FEATURE NAMES AND METRICS
# ============================================================================

print("\n" + "=" * 70)
print("SAVING METADATA AND FEATURE INFORMATION")
print("=" * 70)

metadata = {
    'feature_names': list(X.columns),
    'categorical_cols': categorical_cols,
    'numeric_cols': numeric_cols,
    'regression_metrics': {
        'linear_regression': {'r2': lr_r2, 'mae': lr_mae, 'rmse': lr_rmse},
        'random_forest': {'r2': rf_r2, 'mae': rf_mae, 'rmse': rf_rmse},
        'xgboost': {'r2': xgb_r2, 'mae': xgb_mae, 'rmse': xgb_rmse}
    },
    'classification_metrics': {
        'random_forest': {'accuracy': rfc_acc, 'f1': rfc_f1},
        'logistic_regression': {'accuracy': lr_class_acc, 'f1': lr_class_f1}
    },
    'clustering_metrics': {
        'kmeans': {'optimal_k': optimal_k, 'silhouette_score': silhouette_avg},
        'pca_variance': explained_var
    }
}

with open('models/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# Feature importance from best regression model (XGBoost)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

with open('models/feature_importance.pkl', 'wb') as f:
    pickle.dump(feature_importance, f)

print("\n✓ Feature importance saved")
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE - MODEL SUMMARY")
print("=" * 70)

print("\n📊 REGRESSION MODELS (Score Prediction)")
print(f"  Linear Regression   → R²: {lr_r2:.4f}, MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}")
print(f"  Random Forest       → R²: {rf_r2:.4f}, MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
print(f"  XGBoost (BEST)      → R²: {xgb_r2:.4f}, MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}")

print("\n📈 CLASSIFICATION MODELS (Performance Tier)")
print(f"  Random Forest (3-tier) → Accuracy: {rfc_acc:.4f}, F1: {rfc_f1:.4f}")
print(f"  Logistic Regression    → Accuracy: {lr_class_acc:.4f}, F1: {lr_class_f1:.4f}")

print("\n🔍 UNSUPERVISED LEARNING")
print(f"  K-Means (k={optimal_k})   → Silhouette: {silhouette_avg:.4f}")
print(f"  PCA (3 components)    → Variance explained: {explained_var:.4f}")

print("\n✅ All models saved to 'models/' directory")
print("✅ Ready to use with Streamlit app!")
