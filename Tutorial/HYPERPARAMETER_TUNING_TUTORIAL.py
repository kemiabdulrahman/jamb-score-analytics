"""
COMPREHENSIVE HYPERPARAMETER TUNING TUTORIAL
Optimizing ML Models for Better Performance

This tutorial covers:
1. What are hyperparameters and why tune them?
2. GridSearchCV vs RandomSearchCV vs Bayesian Optimization
3. Cross-validation strategies
4. Parameter grids for each model
5. Practical implementation examples
6. Performance improvement analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HYPERPARAMETER TUNING FOR JAMB SCORE ANALYTICS")
print("=" * 80)

# ============================================================================
# SECTION 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n[SECTION 1] Loading and preparing data...")

df = pd.read_csv("jamb_exam_results.csv")
data = df.copy()

# Encode categorical variables
categorical_cols = ['School_Type', 'School_Location', 'Extra_Tutorials', 
                   'Access_To_Learning_Materials', 'Parent_Involvement', 
                   'IT_Knowledge', 'Gender', 'Socioeconomic_Status', 'Parent_Education_Level']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Prepare features and target
feature_cols = [col for col in data.columns if col != 'Student_ID' and col != 'JAMB_Score']
X = data[feature_cols].copy()
y_regression = df['JAMB_Score'].copy()

# Create classification targets
y_binary = (y_regression >= 140).astype(int)  # Pass/Fail
y_multiclass = pd.cut(y_regression, bins=3, labels=[0, 1, 2])  # Low/Medium/High

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_regression, test_size=0.2, random_state=42
)
_, _, y_bin_train, y_bin_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)
_, _, y_mul_train, y_mul_test = train_test_split(
    X, y_multiclass, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print(f"✓ Features: {X_train.shape[1]}")

# ============================================================================
# SECTION 2: UNDERSTANDING HYPERPARAMETERS
# ============================================================================

print("\n" + "=" * 80)
print("[SECTION 2] HYPERPARAMETER REFERENCE GUIDE")
print("=" * 80)

hyperparameter_guide = """
1. LINEAR REGRESSION
   - fit_intercept: Include intercept term (default: True)
   - positive: Force positive coefficients (default: False)
   
2. LOGISTIC REGRESSION (Binary Classification)
   - C: Inverse regularization strength (lower = stronger regularization)
     Typical: [0.001, 0.01, 0.1, 1, 10, 100]
   - penalty: Type of regularization (l1, l2, elasticnet)
   - solver: Algorithm to optimize (lbfgs, saga, liblinear)
   - max_iter: Maximum iterations (default: 100)
   
3. RANDOM FOREST REGRESSOR/CLASSIFIER
   - n_estimators: Number of trees (50-500, more = better but slower)
     Typical: [50, 100, 200, 300, 500]
   - max_depth: Maximum tree depth (3-20, None = no limit)
     Typical: [5, 10, 15, 20, None]
   - min_samples_split: Min samples to split node (2-20)
     Typical: [2, 5, 10]
   - min_samples_leaf: Min samples in leaf node (1-20)
     Typical: [1, 2, 4]
   - max_features: Features to consider at split ('sqrt', 'log2', None)
   
4. XGBOOST REGRESSOR/CLASSIFIER
   - learning_rate: Step shrinkage (0.001-0.5)
     Typical: [0.01, 0.05, 0.1, 0.2]
   - n_estimators: Number of boosting rounds (50-500)
     Typical: [50, 100, 200, 300]
   - max_depth: Maximum tree depth (3-10)
     Typical: [3, 5, 7, 10]
   - subsample: Fraction of samples for each tree (0.5-1.0)
     Typical: [0.6, 0.8, 1.0]
   - colsample_bytree: Fraction of features for each tree (0.5-1.0)
     Typical: [0.6, 0.8, 1.0]
   - gamma: Minimum loss reduction for split (0-5)
     Typical: [0, 1, 5]
   - reg_alpha: L1 regularization (0-1)
   - reg_lambda: L2 regularization (0-1)
"""

print(hyperparameter_guide)

# ============================================================================
# SECTION 3: GRID SEARCH VS RANDOM SEARCH
# ============================================================================

print("\n" + "=" * 80)
print("[SECTION 3] GRID SEARCH VS RANDOM SEARCH EXPLANATION")
print("=" * 80)

comparison_text = """
GRID SEARCH (GridSearchCV)
✓ Exhaustively search every combination
✓ Guarantees finding best combination in parameter space
✓ Works well for small parameter spaces (2-3 hyperparameters)
✗ Very slow with large parameter spaces (exponential growth)
✗ Example: 3 parameters × 5 values each = 125 combinations

RANDOM SEARCH (RandomizedSearchCV)
✓ Randomly sample parameter combinations
✓ Much faster than grid search
✓ Can find good solutions with fewer iterations
✓ Better for large parameter spaces (5+ hyperparameters)
✗ Doesn't guarantee optimal solution
✗ May miss good combinations by chance
✓ Example: 3 parameters × 5 values each = can sample 50 combinations

WHEN TO USE:
- GridSearch: Small grids (< 100 combinations), important parameters
- RandomSearch: Large grids (> 100 combinations), computational constraints

RECOMMENDATION FOR THIS PROJECT:
- Use GridSearch for relatively simple models (Linear Regression, Logistic)
- Use RandomSearch for complex models (XGBoost, Random Forest) with many parameters
"""

print(comparison_text)

# ============================================================================
# SECTION 4: REGRESSION MODEL TUNING
# ============================================================================

print("\n" + "=" * 80)
print("[SECTION 4] REGRESSION MODEL HYPERPARAMETER TUNING")
print("=" * 80)

results_regression = {}

# 4.1: Linear Regression (Baseline - simple model, limited tuning)
print("\n[4.1] Linear Regression")
print("-" * 80)

lr_params = {'fit_intercept': [True, False]}
lr_grid = GridSearchCV(LinearRegression(), lr_params, cv=5, n_jobs=-1)
start_time = time.time()
lr_grid.fit(X_train_scaled, y_reg_train)
tuning_time = time.time() - start_time

lr_pred = lr_grid.predict(X_test_scaled)
lr_r2 = r2_score(y_reg_test, lr_pred)
lr_mae = mean_absolute_error(y_reg_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_reg_test, lr_pred))

print(f"✓ Best parameters: {lr_grid.best_params_}")
print(f"✓ Best CV score: {lr_grid.best_score_:.4f}")
print(f"✓ Test R²: {lr_r2:.4f}")
print(f"✓ Test MAE: {lr_mae:.2f}")
print(f"✓ Test RMSE: {lr_rmse:.2f}")
print(f"✓ Tuning time: {tuning_time:.2f}s")

results_regression['Linear Regression'] = {
    'best_params': lr_grid.best_params_,
    'r2': lr_r2,
    'mae': lr_mae,
    'rmse': lr_rmse
}

# 4.2: Random Forest Regressor
print("\n[4.2] Random Forest Regressor")
print("-" * 80)

rf_reg_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use RandomSearch due to large parameter space (3*4*3*3 = 108 combinations)
rf_reg_grid = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_reg_params,
    n_iter=20,  # Sample 20 combinations
    cv=5,
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
rf_reg_grid.fit(X_train, y_reg_train)
tuning_time = time.time() - start_time

rf_reg_pred = rf_reg_grid.predict(X_test)
rf_reg_r2 = r2_score(y_reg_test, rf_reg_pred)
rf_reg_mae = mean_absolute_error(y_reg_test, rf_reg_pred)
rf_reg_rmse = np.sqrt(mean_squared_error(y_reg_test, rf_reg_pred))

print(f"✓ Best parameters: {rf_reg_grid.best_params_}")
print(f"✓ Best CV score: {rf_reg_grid.best_score_:.4f}")
print(f"✓ Test R²: {rf_reg_r2:.4f}")
print(f"✓ Test MAE: {rf_reg_mae:.2f}")
print(f"✓ Test RMSE: {rf_reg_rmse:.2f}")
print(f"✓ Tuning time: {tuning_time:.2f}s")

results_regression['Random Forest'] = {
    'best_params': rf_reg_grid.best_params_,
    'r2': rf_reg_r2,
    'mae': rf_reg_mae,
    'rmse': rf_reg_rmse
}

# 4.3: XGBoost Regressor
print("\n[4.3] XGBoost Regressor")
print("-" * 80)

xgb_reg_params = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_reg_grid = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    xgb_reg_params,
    n_iter=20,  # Sample 20 combinations
    cv=5,
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
xgb_reg_grid.fit(X_train, y_reg_train)
tuning_time = time.time() - start_time

xgb_reg_pred = xgb_reg_grid.predict(X_test)
xgb_reg_r2 = r2_score(y_reg_test, xgb_reg_pred)
xgb_reg_mae = mean_absolute_error(y_reg_test, xgb_reg_pred)
xgb_reg_rmse = np.sqrt(mean_squared_error(y_reg_test, xgb_reg_pred))

print(f"✓ Best parameters: {xgb_reg_grid.best_params_}")
print(f"✓ Best CV score: {xgb_reg_grid.best_score_:.4f}")
print(f"✓ Test R²: {xgb_reg_r2:.4f}")
print(f"✓ Test MAE: {xgb_reg_mae:.2f}")
print(f"✓ Test RMSE: {xgb_reg_rmse:.2f}")
print(f"✓ Tuning time: {tuning_time:.2f}s")

results_regression['XGBoost'] = {
    'best_params': xgb_reg_grid.best_params_,
    'r2': xgb_reg_r2,
    'mae': xgb_reg_mae,
    'rmse': xgb_reg_rmse
}

# ============================================================================
# SECTION 5: CLASSIFICATION MODEL TUNING
# ============================================================================

print("\n" + "=" * 80)
print("[SECTION 5] CLASSIFICATION MODEL HYPERPARAMETER TUNING")
print("=" * 80)

results_classification = {}

# 5.1: Logistic Regression (Binary)
print("\n[5.1] Logistic Regression (Binary Classification)")
print("-" * 80)

lr_clf_params = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'max_iter': [200]
}

lr_clf_grid = GridSearchCV(LogisticRegression(random_state=42), lr_clf_params, cv=5, n_jobs=-1)
start_time = time.time()
lr_clf_grid.fit(X_train_scaled, y_bin_train)
tuning_time = time.time() - start_time

lr_clf_pred = lr_clf_grid.predict(X_test_scaled)
lr_clf_acc = accuracy_score(y_bin_test, lr_clf_pred)
lr_clf_f1 = f1_score(y_bin_test, lr_clf_pred)

print(f"✓ Best parameters: {lr_clf_grid.best_params_}")
print(f"✓ Best CV score: {lr_clf_grid.best_score_:.4f}")
print(f"✓ Test Accuracy: {lr_clf_acc:.4f}")
print(f"✓ Test F1-Score: {lr_clf_f1:.4f}")
print(f"✓ Tuning time: {tuning_time:.2f}s")

results_classification['Logistic Regression'] = {
    'best_params': lr_clf_grid.best_params_,
    'accuracy': lr_clf_acc,
    'f1': lr_clf_f1
}

# 5.2: Random Forest Classifier (Binary)
print("\n[5.2] Random Forest Classifier (Binary Classification)")
print("-" * 80)

rf_clf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_clf_grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_clf_params,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
rf_clf_grid.fit(X_train, y_bin_train)
tuning_time = time.time() - start_time

rf_clf_pred = rf_clf_grid.predict(X_test)
rf_clf_acc = accuracy_score(y_bin_test, rf_clf_pred)
rf_clf_f1 = f1_score(y_bin_test, rf_clf_pred)

print(f"✓ Best parameters: {rf_clf_grid.best_params_}")
print(f"✓ Best CV score: {rf_clf_grid.best_score_:.4f}")
print(f"✓ Test Accuracy: {rf_clf_acc:.4f}")
print(f"✓ Test F1-Score: {rf_clf_f1:.4f}")
print(f"✓ Tuning time: {tuning_time:.2f}s")

results_classification['Random Forest (Binary)'] = {
    'best_params': rf_clf_grid.best_params_,
    'accuracy': rf_clf_acc,
    'f1': rf_clf_f1
}

# 5.3: Random Forest Classifier (Multi-class)
print("\n[5.3] Random Forest Classifier (Multi-class - Low/Medium/High)")
print("-" * 80)

rf_mc_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_mc_grid = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_mc_params,
    n_iter=15,
    cv=5,
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
rf_mc_grid.fit(X_train, y_mul_train)
tuning_time = time.time() - start_time

rf_mc_pred = rf_mc_grid.predict(X_test)
rf_mc_acc = accuracy_score(y_mul_test, rf_mc_pred)
rf_mc_f1 = f1_score(y_mul_test, rf_mc_pred, average='weighted')

print(f"✓ Best parameters: {rf_mc_grid.best_params_}")
print(f"✓ Best CV score: {rf_mc_grid.best_score_:.4f}")
print(f"✓ Test Accuracy: {rf_mc_acc:.4f}")
print(f"✓ Test F1-Score (weighted): {rf_mc_f1:.4f}")
print(f"✓ Tuning time: {tuning_time:.2f}s")

results_classification['Random Forest (Multi-class)'] = {
    'best_params': rf_mc_grid.best_params_,
    'accuracy': rf_mc_acc,
    'f1': rf_mc_f1
}

# 5.4: XGBoost Classifier
print("\n[5.4] XGBoost Classifier (Binary Classification)")
print("-" * 80)

xgb_clf_params = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_clf_grid = RandomizedSearchCV(
    XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False, eval_metric='logloss'),
    xgb_clf_params,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    random_state=42
)

start_time = time.time()
xgb_clf_grid.fit(X_train, y_bin_train)
tuning_time = time.time() - start_time

xgb_clf_pred = xgb_clf_grid.predict(X_test)
xgb_clf_acc = accuracy_score(y_bin_test, xgb_clf_pred)
xgb_clf_f1 = f1_score(y_bin_test, xgb_clf_pred)

print(f"✓ Best parameters: {xgb_clf_grid.best_params_}")
print(f"✓ Best CV score: {xgb_clf_grid.best_score_:.4f}")
print(f"✓ Test Accuracy: {xgb_clf_acc:.4f}")
print(f"✓ Test F1-Score: {xgb_clf_f1:.4f}")
print(f"✓ Tuning time: {tuning_time:.2f}s")

results_classification['XGBoost'] = {
    'best_params': xgb_clf_grid.best_params_,
    'accuracy': xgb_clf_acc,
    'f1': xgb_clf_f1
}

# ============================================================================
# SECTION 6: RESULTS SUMMARY AND VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("[SECTION 6] SUMMARY OF TUNING RESULTS")
print("=" * 80)

# Create results dataframe
print("\nREGRESSION MODELS:")
print("-" * 80)
reg_df = pd.DataFrame(results_regression).T
print(reg_df[['r2', 'mae', 'rmse']])

print("\nCLASSIFICATION MODELS:")
print("-" * 80)
clf_df = pd.DataFrame(results_classification).T
print(clf_df[['accuracy', 'f1']])

# Save all models
print("\n" + "=" * 80)
print("[SECTION 7] SAVING OPTIMIZED MODELS")
print("=" * 80)

import pickle

models_to_save = {
    'linear_regression_tuned': lr_grid.best_estimator_,
    'random_forest_regressor_tuned': rf_reg_grid.best_estimator_,
    'xgboost_regressor_tuned': xgb_reg_grid.best_estimator_,
    'logistic_regression_tuned': lr_clf_grid.best_estimator_,
    'random_forest_classifier_binary_tuned': rf_clf_grid.best_estimator_,
    'random_forest_classifier_multiclass_tuned': rf_mc_grid.best_estimator_,
    'xgboost_classifier_tuned': xgb_clf_grid.best_estimator_
}

for name, model in models_to_save.items():
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved {name}")

# Save results
with open('models/tuning_results.pkl', 'wb') as f:
    pickle.dump({
        'regression': results_regression,
        'classification': results_classification
    }, f)
print("✓ Saved tuning results")

print("\n" + "=" * 80)
print("✓ HYPERPARAMETER TUNING COMPLETE!")
print("=" * 80)

print(f"""
KEY TAKEAWAYS:
1. GridSearch: Exhaustive search - good for small parameter spaces
2. RandomSearch: Efficient sampling - good for large parameter spaces
3. Cross-validation: Prevents overfitting by averaging across folds
4. Trade-offs: Better accuracy often requires more training time
5. Parameter importance varies by model and dataset

BEST MODELS:
- Regression: {reg_df['r2'].idxmax()} (R² = {reg_df['r2'].max():.4f})
- Classification: {clf_df['accuracy'].idxmax()} (Accuracy = {clf_df['accuracy'].max():.4f})

All tuned models saved to 'models/' directory!
""")
