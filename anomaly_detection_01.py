"""
ANOMALY DETECTION - COMPREHENSIVE TUTORIAL
Detecting unusual patterns and outliers in JAMB score data
"""

# ============================================================================
# 1. WHAT IS ANOMALY DETECTION?
# ============================================================================

"""
ANOMALY DETECTION: Identifying data points that deviate significantly from 
the normal pattern. These are called "outliers" or "anomalies".

REAL-WORLD EXAMPLES:
- Fraud detection: Unusual credit card transactions
- Intrusion detection: Suspicious network activity
- Equipment failure: Abnormal sensor readings
- Quality control: Defective products

IN JAMB CONTEXT:
- Students with suspiciously high scores
- Students with low scores despite high attendance
- Unusual study patterns
- Grade inflation or deflation

WHY DETECT ANOMALIES?
1. Data Quality: Remove erroneous data
2. Fraud Detection: Identify cheating or data manipulation
3. Insights: Find exceptional students or cases needing attention
4. Model Improvement: Exclude outliers for better training
"""

# ============================================================================
# 2. TYPES OF ANOMALIES
# ============================================================================

"""
1. POINT ANOMALY
   - Single data point is abnormal
   - Example: Student with score 298 (highest possible)
   - Most common type

2. CONTEXTUAL ANOMALY
   - Normal in general but unusual in context
   - Example: Score 100 in urban school (context is low-resource)
   - Example: 40 study hours but score 120 (unusual efficiency)

3. COLLECTIVE ANOMALY
   - Subset of points forms anomalous pattern
   - Example: Group of students with identical high scores
   - Suggests data manipulation

DETECTION APPROACHES:
1. Statistical: Based on distribution (z-score, IQR)
2. Distance-based: Isolation Forest, Local Outlier Factor
3. Density-based: DBSCAN
4. Model-based: Auto-encoder Neural Networks
"""

# ============================================================================
# 3. STATISTICAL ANOMALY DETECTION
# ============================================================================

"""
Using statistics to identify outliers.

METHOD 1: Z-SCORE
- How many standard deviations from mean?
- Formula: z = (x - mean) / std_dev
- Threshold: |z| > 3 (99.7% of data within ±3σ)

METHOD 2: INTERQUARTILE RANGE (IQR)
- Q1: 25th percentile
- Q3: 75th percentile
- IQR = Q3 - Q1
- Lower Bound: Q1 - 1.5 * IQR
- Upper Bound: Q3 + 1.5 * IQR
- Points outside bounds are outliers

METHOD 3: MODIFIED Z-SCORE
- Uses median instead of mean (robust to outliers)
- Better for skewed distributions
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("jamb_exam_results.csv")

# ============================================================================
# STATISTICAL APPROACH 1: Z-SCORE
# ============================================================================

scores = df['JAMB_Score']

# Calculate z-scores
z_scores = np.abs(stats.zscore(scores))

# Identify anomalies (|z| > 3)
anomaly_mask_zscore = z_scores > 3
anomalies_zscore = df[anomaly_mask_zscore]

print(f"Anomalies detected (Z-score): {anomaly_mask_zscore.sum()}")
print(f"Anomaly indices: {anomalies_zscore.index.tolist()}")

# ============================================================================
# STATISTICAL APPROACH 2: INTERQUARTILE RANGE (IQR)
# ============================================================================

Q1 = scores.quantile(0.25)  # 25th percentile
Q3 = scores.quantile(0.75)  # 75th percentile
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nIQR Method:")
print(f"Q1: {Q1:.0f}, Q3: {Q3:.0f}, IQR: {IQR:.0f}")
print(f"Bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")

anomaly_mask_iqr = (scores < lower_bound) | (scores > upper_bound)
anomalies_iqr = df[anomaly_mask_iqr]

print(f"Anomalies detected (IQR): {anomaly_mask_iqr.sum()}")

# ============================================================================
# VISUALIZATION OF ANOMALIES
# ============================================================================

"""
WHAT TO VISUALIZE:
1. Box plot with outliers highlighted
2. Histogram with anomaly regions shaded
3. Scatter plot with anomalies marked
"""

import matplotlib.pyplot as plt

# Box plot with marked anomalies
fig, ax = plt.subplots(figsize=(10, 6))

bp = ax.boxplot(scores, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')

# Overlay anomalies
anomaly_x = [1] * anomaly_mask_iqr.sum()
anomaly_y = scores[anomaly_mask_iqr]
ax.scatter(anomaly_x, anomaly_y, color='red', s=100, marker='X', 
          label='Anomalies', zorder=5)

ax.set_ylabel('JAMB Score')
ax.set_title('Box Plot with Anomalies Highlighted (IQR Method)')
ax.legend()
plt.show()

# ============================================================================
# 4. DISTANCE-BASED ANOMALY DETECTION: ISOLATION FOREST
# ============================================================================

"""
ISOLATION FOREST: Modern, efficient anomaly detection

HOW IT WORKS:
1. Randomly select a feature
2. Randomly select split value
3. Build isolation trees
4. Anomalies are isolated quickly (short paths)
5. Normal points need more splits

ADVANTAGES:
- Fast and scalable
- No distance computation (efficient in high dimensions)
- Doesn't assume data distribution
- Good for multivariate anomalies

PARAMETERS:
- n_estimators: Number of trees (default 100)
- contamination: Expected % of anomalies (0-1)
  - If contamination=0.1, top 10% are marked anomalies
  - Helps with imbalanced data
- max_samples: Samples per tree
- random_state: For reproducibility
"""

from sklearn.ensemble import IsolationForest

# Prepare data for Isolation Forest
# Need to select features to use for anomaly detection
feature_cols = ['JAMB_Score', 'Study_Hours_Per_Week', 'Attendance_Rate', 
                'Teacher_Quality', 'Age', 'Distance_To_School']

X = df[feature_cols].copy()

# Initialize Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,      # Number of trees
    contamination=0.05,    # Expected 5% anomalies
    random_state=42,
    n_jobs=-1             # Use all processors
)

# Fit and predict
# Returns: 1 for normal, -1 for anomaly
predictions = iso_forest.fit_predict(X)
anomaly_scores = iso_forest.score_samples(X)  # Anomaly score (-1 to 1)

# Add to dataframe
df['Anomaly'] = predictions
df['Anomaly_Score'] = anomaly_scores

# Extract anomalies
anomalies_iso = df[df['Anomaly'] == -1]

print(f"\nIsolation Forest:")
print(f"Anomalies detected: {len(anomalies_iso)}")
print(f"\nAnomaly details:")
print(anomalies_iso[['JAMB_Score', 'Study_Hours_Per_Week', 'Attendance_Rate', 'Anomaly_Score']])

# ============================================================================
# 5. ANOMALY SCORE ANALYSIS
# ============================================================================

"""
ANOMALY SCORE:
- Range: -1 to 1 (approximately)
- More negative = More anomalous
- Threshold for binary classification depends on use case

INTERPRETING SCORES:
- score < -0.3: Clearly anomalous
- -0.3 to -0.1: Borderline
- > -0.1: Normal
"""

# Threshold analysis
thresholds = [-0.5, -0.3, -0.1]

for threshold in thresholds:
    count = (df['Anomaly_Score'] < threshold).sum()
    percentage = count / len(df) * 100
    print(f"Anomalies below {threshold}: {count} ({percentage:.1f}%)")

# ============================================================================
# 6. CONTEXTUAL ANOMALIES - MULTIVARIATE ANALYSIS
# ============================================================================

"""
Finding students with unusual combinations of factors.

EXAMPLES OF CONTEXTUAL ANOMALIES:
- Very high score but very low attendance
- Very high study hours but low score
- High teacher quality but low scores
- High parent involvement but low scores
"""

# Example: High effort but low results
df['Effort'] = df['Study_Hours_Per_Week'] + df['Attendance_Rate']/10
df['Contextual_Anomaly_1'] = (df['Effort'] > df['Effort'].quantile(0.75)) & \
                             (df['JAMB_Score'] < df['JAMB_Score'].quantile(0.25))

contextual_anomalies_1 = df[df['Contextual_Anomaly_1']]
print(f"\nStudents with high effort but low scores: {len(contextual_anomalies_1)}")

# Example: Very efficient (high score, low hours)
df['Study_Efficiency'] = df['JAMB_Score'] / (df['Study_Hours_Per_Week'] + 1)
df['Contextual_Anomaly_2'] = df['Study_Efficiency'] > df['Study_Efficiency'].quantile(0.95)

contextual_anomalies_2 = df[df['Contextual_Anomaly_2']]
print(f"Exceptionally efficient students: {len(contextual_anomalies_2)}")

# ============================================================================
# 7. LOCAL OUTLIER FACTOR (LOF)
# ============================================================================

"""
LOF: Density-based anomaly detection

HOW IT WORKS:
1. Compare local density of point with neighbors
2. Isolated points have low density
3. Score based on density ratio

ADVANTAGES:
- Detects local anomalies (density-based)
- Good for clusters with varying densities

DISADVANTAGES:
- Slower than Isolation Forest
- More parameters to tune
- Harder to interpret
"""

from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,        # Number of neighbors
    contamination=0.05,    # Expected % anomalies
    novelty=False          # Detect on training data
)

lof_predictions = lof.fit_predict(X)
lof_scores = lof.negative_outlier_factor_

df['LOF_Anomaly'] = lof_predictions
df['LOF_Score'] = lof_scores

lof_anomalies = df[df['LOF_Anomaly'] == -1]
print(f"\nLOF Anomalies detected: {len(lof_anomalies)}")

# ============================================================================
# 8. COMBINING MULTIPLE ANOMALY DETECTORS
# ============================================================================

"""
Ensemble approach: Use multiple detectors and vote.

LOGIC:
- If multiple detectors flag a point, it's likely anomalous
- More robust than single method
"""

# Combine detectors
df['Combined_Score'] = (
    (df['Anomaly'] == -1).astype(int) +  # Isolation Forest
    (df['LOF_Anomaly'] == -1).astype(int)  # LOF
)

# Points flagged by both detectors
high_confidence_anomalies = df[df['Combined_Score'] == 2]
print(f"\nHigh-confidence anomalies (flagged by 2+ detectors): {len(high_confidence_anomalies)}")

# ============================================================================
# 9. HANDLING ANOMALIES IN ML PIPELINES
# ============================================================================

"""
WHAT TO DO WITH DETECTED ANOMALIES?

OPTION 1: REMOVE THEM
- Pro: Cleaner training data
- Con: Lose information

OPTION 2: CAP/FLOOR THEM
- Pro: Keep data, reduce extreme values
- Con: Artificial modification

OPTION 3: SEPARATE TREATMENT
- Pro: Learn patterns for both normal and anomalous
- Con: Requires separate models

OPTION 4: FLAG FOR REVIEW
- Pro: Let domain experts decide
- Con: Manual process
"""

# Option 1: Remove anomalies before training
df_clean = df[df['Anomaly'] == 1].copy()

print(f"\nOriginal dataset: {len(df)} samples")
print(f"After removing anomalies: {len(df_clean)} samples")
print(f"Removed: {len(df) - len(df_clean)} samples ({(len(df) - len(df_clean))/len(df)*100:.1f}%)")

# Option 2: Cap extreme values
df_capped = df.copy()

for col in ['JAMB_Score', 'Study_Hours_Per_Week', 'Attendance_Rate']:
    Q1 = df_capped[col].quantile(0.25)
    Q3 = df_capped[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    df_capped[col] = df_capped[col].clip(lower, upper)

print(f"\nValues modified by capping: {(df != df_capped).sum().sum()}")

# ============================================================================
# 10. VISUALIZATION OF ALL METHODS
# ============================================================================

"""
Comparing anomaly detection methods visually.
"""

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Z-Score
axes[0, 0].scatter(df.index, z_scores, alpha=0.6, label='Normal', c='blue')
axes[0, 0].scatter(df[z_scores > 3].index, z_scores[z_scores > 3], 
                   c='red', s=100, marker='X', label='Anomaly')
axes[0, 0].axhline(y=3, color='red', linestyle='--', label='Threshold')
axes[0, 0].set_title('Z-Score Method')
axes[0, 0].set_ylabel('Z-Score')
axes[0, 0].legend()

# Plot 2: IQR Method
axes[0, 1].boxplot(scores, vert=False)
axes[0, 1].scatter(scores[anomaly_mask_iqr], [1]*anomaly_mask_iqr.sum(), 
                   c='red', s=100, marker='X', label='Anomaly')
axes[0, 1].set_title('IQR Method')
axes[0, 1].set_xlabel('JAMB Score')

# Plot 3: Isolation Forest
axes[1, 0].scatter(df.index, df['Anomaly_Score'], c=df['Anomaly'], 
                   cmap='RdYlGn', s=50, alpha=0.6)
axes[1, 0].set_title('Isolation Forest Anomaly Scores')
axes[1, 0].set_ylabel('Anomaly Score')
axes[1, 0].set_xlabel('Student Index')

# Plot 4: LOF
axes[1, 1].scatter(df.index, df['LOF_Score'], c=df['LOF_Anomaly'], 
                   cmap='RdYlGn', s=50, alpha=0.6)
axes[1, 1].set_title('LOF Anomaly Scores')
axes[1, 1].set_ylabel('LOF Score')
axes[1, 1].set_xlabel('Student Index')

plt.tight_layout()
plt.savefig('anomaly_detection_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 11. ANOMALY DETECTION USE CASES
# ============================================================================

"""
PRACTICAL APPLICATIONS IN JAMB ANALYSIS:

USE CASE 1: DATA QUALITY CHECK
- Identify data entry errors
- Remove before model training

USE CASE 2: IDENTIFY EXCEPTIONAL CASES
- High scorers with low attendance
- Low scorers with high resources
- Needs investigation/intervention

USE CASE 3: CHEATING DETECTION
- Identical scores across students
- Unusual score patterns
- Impossible improvements

USE CASE 4: RESOURCE ALLOCATION
- Identify students needing help (low score anomalies)
- Support high performers (exceptional efficiency)
- Focus on borderline cases

USE CASE 5: MODEL IMPROVEMENT
- Remove outliers that confuse model
- Separate models for normal vs anomalous
- Better generalization
"""

# ============================================================================
# 12. KEY CONCEPTS SUMMARY
# ============================================================================

"""
STATISTICAL ANOMALY DETECTION:
- Z-Score: How many std devs from mean?
- IQR: Box plot method, practical and interpretable
- Good for univariate (single variable) detection

DISTANCE-BASED:
- Isolation Forest: Fast, efficient, no distribution assumptions
- Best for multivariate detection

DENSITY-BASED:
- LOF: Local density comparison
- Good for varying density clusters

CONTEXTUAL:
- Combine multiple features
- Find unusual combinations
- More domain knowledge needed

ENSEMBLE:
- Combine multiple detectors
- Vote-based system
- More robust

WHEN TO USE WHICH:
1. Quick check → Z-Score or IQR
2. Multivariate → Isolation Forest
3. Varying densities → LOF
4. Production system → Ensemble
5. Domain specific → Contextual + manual review
"""

# ============================================================================
# 13. PRACTICE EXERCISE
# ============================================================================

"""
EXERCISE: Detect students with anomalous study patterns

Find students where:
- Study hours > 30 (high effort)
- Attendance < 70 (low engagement)
- Score < 150 (low result)

This is unusual - high effort + low attendance + low score suggests:
- Possible cheating (inconsistent data)
- Personal issues affecting both attendance and results
- Gaming the system (counting unproductive hours)
"""

anomalous_students = df[
    (df['Study_Hours_Per_Week'] > 30) &
    (df['Attendance_Rate'] < 70) &
    (df['JAMB_Score'] < 150)
]

print(f"\nStudents with anomalous study patterns: {len(anomalous_students)}")
if len(anomalous_students) > 0:
    print(anomalous_students[['Study_Hours_Per_Week', 'Attendance_Rate', 'JAMB_Score']])
