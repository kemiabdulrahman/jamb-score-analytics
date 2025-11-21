"""
Anomaly Detection Implementation
Detects outliers and unusual patterns in JAMB scores
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("JAMB SCORE ANALYTICS - ANOMALY DETECTION")
print("=" * 70)

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("\n[1/7] Loading and preparing data...")
df = pd.read_csv("jamb_exam_results.csv")

# Create a copy for processing
data = df.copy()

# Encode categorical variables (for multivariate anomaly detection)
categorical_cols = ['School_Type', 'School_Location', 'Extra_Tutorials', 
                   'Access_To_Learning_Materials', 'Parent_Involvement', 
                   'IT_Knowledge', 'Gender', 'Socioeconomic_Status', 'Parent_Education_Level']

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Select numeric features for anomaly detection
numeric_cols = ['JAMB_Score', 'Study_Hours_Per_Week', 'Attendance_Rate', 
               'Teacher_Quality', 'Distance_To_School', 'Age', 'Assignments_Completed']

# Additional features (all numeric except JAMB_Score)
feature_cols = numeric_cols + categorical_cols

X = data[feature_cols].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"  ✓ Data shape: {X.shape}")
print(f"  ✓ Features: {len(feature_cols)}")

# ============================================================================
# STATISTICAL ANOMALY DETECTION
# ============================================================================

print("\n[2/7] Statistical anomaly detection...")

scores = df['JAMB_Score'].values

# Z-Score method
z_scores = np.abs(stats.zscore(scores))
z_anomalies = z_scores > 3

print(f"  ✓ Z-Score anomalies (|z| > 3): {z_anomalies.sum()}")

# IQR method
Q1 = df['JAMB_Score'].quantile(0.25)
Q3 = df['JAMB_Score'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

iqr_anomalies = (df['JAMB_Score'] < lower_bound) | (df['JAMB_Score'] > upper_bound)

print(f"  ✓ IQR anomalies: {iqr_anomalies.sum()}")
print(f"    IQR Bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")

# Add to dataframe
df['Z_Score'] = z_scores
df['Z_Anomaly'] = z_anomalies
df['IQR_Anomaly'] = iqr_anomalies

# ============================================================================
# ISOLATION FOREST
# ============================================================================

print("\n[3/7] Isolation Forest anomaly detection...")

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.05,  # Expect ~5% anomalies
    random_state=42,
    n_jobs=-1
)

iso_predictions = iso_forest.fit_predict(X_scaled)
iso_scores = iso_forest.score_samples(X_scaled)

df['ISO_Anomaly'] = iso_predictions
df['ISO_Score'] = iso_scores

iso_anomaly_count = (iso_predictions == -1).sum()
print(f"  ✓ Isolation Forest anomalies: {iso_anomaly_count}")

# Save model
with open('models/isolation_forest.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)
print("  ✓ Model saved")

# ============================================================================
# LOCAL OUTLIER FACTOR
# ============================================================================

print("\n[4/7] Local Outlier Factor anomaly detection...")

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=False,
    n_jobs=-1
)

lof_predictions = lof.fit_predict(X_scaled)
lof_scores = lof.negative_outlier_factor_

df['LOF_Anomaly'] = lof_predictions
df['LOF_Score'] = lof_scores

lof_anomaly_count = (lof_predictions == -1).sum()
print(f"  ✓ LOF anomalies: {lof_anomaly_count}")

# Save model
with open('models/lof_model.pkl', 'wb') as f:
    pickle.dump(lof, f)
print("  ✓ Model saved")

# ============================================================================
# CONTEXTUAL ANOMALIES
# ============================================================================

print("\n[5/7] Contextual anomaly detection...")

# Anomaly 1: High effort, low results
effort = df['Study_Hours_Per_Week'] + df['Attendance_Rate']/10
high_effort_low_score = (effort > effort.quantile(0.75)) & \
                        (df['JAMB_Score'] < df['JAMB_Score'].quantile(0.25))

print(f"  ✓ High effort, low score: {high_effort_low_score.sum()}")

# Anomaly 2: Low effort, high results (exceptional efficiency)
low_effort_high_score = (effort < effort.quantile(0.25)) & \
                        (df['JAMB_Score'] > df['JAMB_Score'].quantile(0.75))

print(f"  ✓ Low effort, high score: {low_effort_high_score.sum()}")

# Anomaly 3: High attendance but very low score
high_attend_low_score = (df['Attendance_Rate'] > 90) & \
                        (df['JAMB_Score'] < df['JAMB_Score'].quantile(0.1))

print(f"  ✓ High attendance, low score: {high_attend_low_score.sum()}")

# Add to dataframe
df['Context_Anomaly_1'] = high_effort_low_score
df['Context_Anomaly_2'] = low_effort_high_score
df['Context_Anomaly_3'] = high_attend_low_score

# ============================================================================
# ENSEMBLE ANOMALY DETECTION
# ============================================================================

print("\n[6/7] Ensemble anomaly detection...")

# Combine multiple detectors
df['Anomaly_Count'] = (
    (df['Z_Anomaly']).astype(int) +
    (df['IQR_Anomaly']).astype(int) +
    (df['ISO_Anomaly'] == -1).astype(int) +
    (df['LOF_Anomaly'] == -1).astype(int) +
    (df['Context_Anomaly_1']).astype(int) +
    (df['Context_Anomaly_2']).astype(int) +
    (df['Context_Anomaly_3']).astype(int)
)

# Classification
df['Anomaly_Level'] = df['Anomaly_Count'].apply(
    lambda x: 'High' if x >= 4 else 'Medium' if x >= 2 else 'Low'
)

high_conf_anomalies = (df['Anomaly_Count'] >= 4).sum()
medium_conf_anomalies = ((df['Anomaly_Count'] >= 2) & (df['Anomaly_Count'] < 4)).sum()
low_conf_anomalies = ((df['Anomaly_Count'] >= 1) & (df['Anomaly_Count'] < 2)).sum()

print(f"  ✓ High confidence anomalies (≥4 detectors): {high_conf_anomalies}")
print(f"  ✓ Medium confidence anomalies (2-3 detectors): {medium_conf_anomalies}")
print(f"  ✓ Low confidence anomalies (1 detector): {low_conf_anomalies}")

# Save ensemble results
ensemble_results = {
    'high_confidence': high_conf_anomalies,
    'medium_confidence': medium_conf_anomalies,
    'low_confidence': low_conf_anomalies,
    'total_anomalies': (df['Anomaly_Count'] >= 1).sum()
}

with open('models/anomaly_ensemble.pkl', 'wb') as f:
    pickle.dump(ensemble_results, f)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[7/7] Creating visualizations...")

# Plot 1: Z-Score Distribution
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].hist(z_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(3, color='red', linestyle='--', label='Threshold')
axes[0, 0].set_title('Z-Score Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Z-Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# Plot 2: IQR Box Plot
axes[0, 1].boxplot(df['JAMB_Score'], vert=True)
axes[0, 1].scatter([1]*iqr_anomalies.sum(), df['JAMB_Score'][iqr_anomalies], 
                   color='red', s=100, marker='X', label='Anomalies')
axes[0, 1].set_title('IQR Method Box Plot', fontweight='bold')
axes[0, 1].set_ylabel('JAMB Score')
axes[0, 1].legend()

# Plot 3: Isolation Forest Scores
scatter3 = axes[0, 2].scatter(df.index, df['ISO_Score'], c=df['ISO_Anomaly'], 
                             cmap='RdYlGn', s=50, alpha=0.6)
axes[0, 2].set_title('Isolation Forest Anomaly Scores', fontweight='bold')
axes[0, 2].set_xlabel('Student Index')
axes[0, 2].set_ylabel('Anomaly Score')
plt.colorbar(scatter3, ax=axes[0, 2], label='Normal/Anomaly')

# Plot 4: LOF Scores
scatter4 = axes[1, 0].scatter(df.index, df['LOF_Score'], c=df['LOF_Anomaly'], 
                             cmap='RdYlGn', s=50, alpha=0.6)
axes[1, 0].set_title('LOF Anomaly Scores', fontweight='bold')
axes[1, 0].set_xlabel('Student Index')
axes[1, 0].set_ylabel('LOF Score')
plt.colorbar(scatter4, ax=axes[1, 0], label='Normal/Anomaly')

# Plot 5: Anomaly Count Distribution
anomaly_counts = df['Anomaly_Count'].value_counts().sort_index()
axes[1, 1].bar(anomaly_counts.index, anomaly_counts.values, 
              color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Anomaly Count Distribution', fontweight='bold')
axes[1, 1].set_xlabel('Number of Detectors Flagging as Anomaly')
axes[1, 1].set_ylabel('Number of Students')

# Plot 6: Anomaly Level Pie Chart
anomaly_levels = df['Anomaly_Level'].value_counts()
colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
axes[1, 2].pie(anomaly_levels.values, labels=anomaly_levels.index, 
              autopct='%1.1f%%', colors=colors, startangle=90)
axes[1, 2].set_title('Anomaly Confidence Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig("plots/anomaly_detection.png", dpi=300, bbox_inches='tight')
print("  ✓ Anomaly detection plot saved")
plt.close()

# ============================================================================
# ANOMALY ANALYSIS REPORT
# ============================================================================

print("\n" + "=" * 70)
print("ANOMALY DETECTION REPORT")
print("=" * 70)

# High confidence anomalies
high_conf = df[df['Anomaly_Count'] >= 4]

print(f"\nHIGH CONFIDENCE ANOMALIES ({len(high_conf)} students):")
print("-" * 70)

if len(high_conf) > 0:
    print(high_conf[['Student_ID', 'JAMB_Score', 'Study_Hours_Per_Week', 
                     'Attendance_Rate', 'Anomaly_Count']].to_string(index=False))
else:
    print("None detected")

# Specific anomaly types
print(f"\n\nANOMALY TYPE BREAKDOWN:")
print("-" * 70)

print(f"Z-Score Anomalies: {df['Z_Anomaly'].sum()}")
print(f"IQR Anomalies: {df['IQR_Anomaly'].sum()}")
print(f"Isolation Forest Anomalies: {(df['ISO_Anomaly'] == -1).sum()}")
print(f"LOF Anomalies: {(df['LOF_Anomaly'] == -1).sum()}")
print(f"\nContextual Anomalies:")
print(f"  - High effort, low score: {df['Context_Anomaly_1'].sum()}")
print(f"  - Low effort, high score: {df['Context_Anomaly_2'].sum()}")
print(f"  - High attendance, low score: {df['Context_Anomaly_3'].sum()}")

# Save detailed report
detailed_anomalies = df[df['Anomaly_Count'] >= 1][
    ['Student_ID', 'JAMB_Score', 'Study_Hours_Per_Week', 'Attendance_Rate',
     'Teacher_Quality', 'Z_Score', 'ISO_Score', 'LOF_Score', 'Anomaly_Count', 'Anomaly_Level']
].sort_values('Anomaly_Count', ascending=False)

detailed_anomalies.to_csv('anomaly_report.csv', index=False)
print(f"\n✓ Detailed anomaly report saved to 'anomaly_report.csv'")

# Save cleaned dataset (anomalies removed)
df_clean = df[df['Anomaly_Count'] <= 1].copy()
print(f"\n✓ Cleaned dataset: {len(df)} → {len(df_clean)} samples")
print(f"  Removed: {len(df) - len(df_clean)} samples ({(len(df) - len(df_clean))/len(df)*100:.1f}%)")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(f"\nOriginal Data:")
print(f"  Mean JAMB Score: {df['JAMB_Score'].mean():.2f}")
print(f"  Std Dev: {df['JAMB_Score'].std():.2f}")
print(f"  Min: {df['JAMB_Score'].min():.0f}, Max: {df['JAMB_Score'].max():.0f}")

print(f"\nAfter Removing High Confidence Anomalies:")
df_no_high = df[df['Anomaly_Count'] < 4]
print(f"  Mean JAMB Score: {df_no_high['JAMB_Score'].mean():.2f}")
print(f"  Std Dev: {df_no_high['JAMB_Score'].std():.2f}")
print(f"  Min: {df_no_high['JAMB_Score'].min():.0f}, Max: {df_no_high['JAMB_Score'].max():.0f}")

print(f"\n✓ Anomaly detection complete!")
print(f"✓ All models and results saved to 'models/' directory")
