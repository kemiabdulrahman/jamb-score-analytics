"""
JAMB SCORE ANALYTICS - COMPREHENSIVE TUTORIAL
Master Guide to Understanding All Concepts and Code Patterns
"""

# ============================================================================
# TABLE OF CONTENTS
# ============================================================================
"""
1. PANDAS FUNDAMENTALS
   1.1 DataFrames and Series
   1.2 Loading and Exploring Data
   1.3 Data Selection and Filtering
   1.4 GroupBy Operations
   1.5 Data Aggregation

2. DATA PREPROCESSING
   2.1 Encoding Categorical Variables
   2.2 Feature Scaling
   2.3 Train-Test Split

3. VISUALIZATION CONCEPTS
   3.1 Matplotlib Basics
   3.2 Seaborn for Statistical Plots
   3.3 Plot Customization
   3.4 Multiple Subplots

4. MACHINE LEARNING FUNDAMENTALS
   4.1 Supervised Learning (Regression & Classification)
   4.2 Unsupervised Learning (Clustering)
   4.3 Model Training and Evaluation
   4.4 Feature Importance

5. STREAMLIT DASHBOARD CONCEPTS
   5.1 Page Configuration
   5.2 Caching and Performance
   5.3 Interactive Components
   5.4 Layout and Styling
"""

# ============================================================================
# 1. PANDAS FUNDAMENTALS
# ============================================================================

"""
PANDAS is a powerful library for data manipulation and analysis.
It provides two main data structures: Series (1D) and DataFrame (2D).

WHY PANDAS?
- Easy data loading from CSV, JSON, Excel
- Powerful filtering and selection
- Built-in statistical functions
- GroupBy for aggregation operations
- Seamless integration with NumPy and Matplotlib
"""

# 1.1 LOADING AND EXPLORING DATA
# ============================================================================

import pandas as pd
import numpy as np

# Load CSV file into DataFrame
df = pd.read_csv("jamb_exam_results.csv")

# View first few rows
print(df.head())  # Shows first 5 rows
print(df.head(10))  # Shows first 10 rows

# Get basic info
print(df.info())  # Data types, non-null counts, memory usage
print(df.shape)  # (rows, columns) → (200, 17)
print(df.columns)  # List all column names
print(df.describe())  # Statistics: mean, std, min, max, quartiles

# Data types
print(df.dtypes)  # Shows type of each column
print(df['JAMB_Score'].dtype)  # Check specific column type


# 1.2 SELECTING AND ACCESSING DATA
# ============================================================================

"""
Different ways to access data in pandas:
"""

# Access by column (returns Series - 1D)
scores = df['JAMB_Score']  # Using bracket notation
scores = df.JAMB_Score     # Using dot notation (if no spaces in name)

# Access multiple columns (returns DataFrame - 2D)
subset = df[['JAMB_Score', 'Study_Hours_Per_Week', 'Attendance_Rate']]

# Access by row using iloc (integer location)
first_row = df.iloc[0]      # First row as Series
first_five_rows = df.iloc[0:5]  # First 5 rows as DataFrame

# Access by row using loc (label-based)
row_by_id = df.loc[0]  # Row with index 0

# Access specific cell
value = df.iloc[0, 0]  # First row, first column
value = df.loc[0, 'JAMB_Score']  # Row 0, JAMB_Score column


# 1.3 FILTERING DATA
# ============================================================================

"""
Filtering means selecting rows that meet certain conditions.
Returns a Boolean mask (True/False for each row)
"""

# Filter high scorers (>220)
high_scorers = df[df['JAMB_Score'] > 220]
print(f"Students with score > 220: {len(high_scorers)}")

# Multiple conditions using & (AND), | (OR), ~ (NOT)
urban_high_achievers = df[(df['School_Location'] == 'Urban') & 
                          (df['JAMB_Score'] > 220)]

# Filter with ~NOT
not_urban = df[~(df['School_Location'] == 'Urban')]

# Filter by column values
public_school_students = df[df['School_Type'] == 'Public']

# Using .isin() for multiple values
science_students = df[df['Parent_Education_Level'].isin(['Tertiary', 'Secondary'])]


# 1.4 GROUPBY OPERATIONS - THE MOST IMPORTANT CONCEPT!
# ============================================================================

"""
GROUPBY: Split-Apply-Combine Strategy

Purpose: Group rows by category and apply operations to each group
Common operations: mean, sum, count, std, min, max, etc.

WHY USE GROUPBY?
- Calculate statistics for each category
- Compare across groups
- Aggregate data efficiently
- Foundation for statistical visualizations

SYNTAX: df.groupby('column_name')['target_column'].operation()
"""

# Example 1: Average JAMB score by school type
avg_by_school = df.groupby('School_Type')['JAMB_Score'].mean()
"""
Output:
School_Type
Private    185.4
Public     190.2
"""

# Example 2: Count students in each cluster
students_per_school = df.groupby('School_Type').size()
# or
students_per_school = df.groupby('School_Type')['JAMB_Score'].count()

# Example 3: Multiple statistics at once
score_stats = df.groupby('School_Type')['JAMB_Score'].agg(['mean', 'std', 'count', 'min', 'max'])
"""
Output:
              mean   std  count  min  max
School_Type                            
Private      185.4  25.3     45   100  298
Public       190.2  28.1    155   100  274
"""

# Example 4: Groupby multiple columns
stats_by_location_gender = df.groupby(['School_Location', 'Gender'])['JAMB_Score'].mean()
"""
Output:
School_Location  Gender
Rural            Female    180.5
                 Male      175.3
Urban            Female    195.2
                 Male      192.1
"""

# Example 5: Using transform (returns same length as original)
# Add average score for each school type as new column
df['avg_score_by_school'] = df.groupby('School_Type')['JAMB_Score'].transform('mean')

# Example 6: Using apply with custom function
def categorize_score(score):
    if score > 220:
        return 'High'
    elif score >= 150:
        return 'Medium'
    else:
        return 'Low'

df['Performance_Tier'] = df['JAMB_Score'].apply(categorize_score)

# USED IN OUR CODE:
# ────────────────
# involvement_data = df.groupby("Parent_Involvement")["JAMB_Score"].agg(['mean', 'std', 'count'])
# This groups by Parent_Involvement and calculates mean, std, count for each group


# 1.5 DATA AGGREGATION METHODS
# ============================================================================

"""
Common aggregation functions:
- mean(): Average value
- sum(): Total sum
- count(): Number of non-null values
- std(): Standard deviation
- min(): Minimum value
- max(): Maximum value
- median(): Middle value
"""

# Single aggregation
mean_score = df['JAMB_Score'].mean()  # 175.5
total_hours = df['Study_Hours_Per_Week'].sum()  # 4532

# Multiple aggregations
stats = df['JAMB_Score'].agg(['mean', 'median', 'std', 'min', 'max', 'count'])

# Custom aggregation
from scipy.stats import skew, kurtosis

stats = df['JAMB_Score'].agg(['mean', 'median', skew, kurtosis])

# Create new DataFrame from aggregation
score_analysis = pd.DataFrame({
    'Mean': [df['JAMB_Score'].mean()],
    'Median': [df['JAMB_Score'].median()],
    'Std': [df['JAMB_Score'].std()],
    'Min': [df['JAMB_Score'].min()],
    'Max': [df['JAMB_Score'].max()]
})


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================

"""
Preprocessing is crucial for machine learning.
Prepare raw data into format suitable for models.

Steps:
1. Handle missing values
2. Encode categorical variables (text → numbers)
3. Scale/normalize numeric features
4. Split into train/test sets
"""

# 2.1 ENCODING CATEGORICAL VARIABLES
# ============================================================================

"""
WHY ENCODE?
Machine learning models require numeric input.
Categorical variables (text) must be converted to numbers.

METHODS:
1. LabelEncoder: Maps categories to 0, 1, 2, ... (ordinal)
2. OneHotEncoder: Creates binary columns for each category
3. OrdinalEncoder: Similar to LabelEncoder

WHEN TO USE:
- LabelEncoder: When order doesn't matter or is ordinal
- OneHotEncoder: When categories are independent (nominal)
"""

from sklearn.preprocessing import LabelEncoder

# EXAMPLE 1: Encoding a single column
le = LabelEncoder()
data['School_Type_Encoded'] = le.fit_transform(data['School_Type'])
# 'Public' → 0, 'Private' → 1

# Get mapping
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
# {'Public': 0, 'Private': 1}

# Transform new data
new_value = le.transform(['Public'])  # Returns [0]

# EXAMPLE 2: Encode multiple columns (as in our code)
categorical_cols = ['School_Type', 'School_Location', 'Gender', ...]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoder for future use

# Later, when making predictions, use same encoder:
new_school = label_encoders['School_Type'].transform(['Private'])

# USED IN OUR CODE:
# ────────────────
# for col in categorical_cols:
#     le = LabelEncoder()
#     data[col] = le.fit_transform(data[col])
#     label_encoders[col] = le
# This encodes all categorical columns and saves encoders for later use


# 2.2 FEATURE SCALING
# ============================================================================

"""
WHY SCALE?
Different features have different ranges:
- Study_Hours_Per_Week: 0-40
- Attendance_Rate: 50-100
- Age: 15-25

This can bias machine learning models toward larger-range features.

SCALING METHODS:
1. StandardScaler: (X - mean) / std → Range: approximately -3 to 3
2. MinMaxScaler: (X - min) / (max - min) → Range: 0 to 1
3. RobustScaler: Uses median and IQR, robust to outliers

WHEN TO SCALE:
- Before: Linear Regression, Logistic Regression, KNN, SVM, Neural Networks
- NOT needed: Tree-based models (Random Forest, XGBoost)
"""

from sklearn.preprocessing import StandardScaler

# Create scaler object
scaler = StandardScaler()

# Fit on training data (learn mean and std)
scaler.fit(X_train)

# Transform training data
X_train_scaled = scaler.transform(X_train)

# Transform test data (use training statistics!)
X_test_scaled = scaler.transform(X_test)

# Convert to DataFrame (for readability)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# IMPORTANT: Never fit scaler on test data!
# Reason: Prevents data leakage and unrealistic performance metrics

# SAVE SCALER for production
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# LOAD SCALER later
with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# Use saved scaler on new data
new_data_scaled = loaded_scaler.transform(new_data)


# 2.3 TRAIN-TEST SPLIT
# ============================================================================

"""
WHY SPLIT?
- Training set: Teach model patterns
- Test set: Evaluate on unseen data (avoid overfitting)

TYPICAL SPLIT:
- 80/20: 80% train, 20% test
- 70/30: Common alternative
- 60/40: For small datasets

IMPORTANT CONCEPTS:
- Stratification: Maintain class distribution in splits
- Random state: Reproducibility (same split every time)
"""

from sklearn.model_selection import train_test_split

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,              # Features and target
    test_size=0.2,     # 20% test, 80% train
    random_state=42    # For reproducibility
)

# Stratified split (maintain class distribution)
# Useful for imbalanced datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain y distribution in both sets
)

# Check split
print(f"Training set size: {X_train.shape}")  # (160, 17)
print(f"Test set size: {X_test.shape}")       # (40, 17)


# ============================================================================
# 3. VISUALIZATION CONCEPTS
# ============================================================================

"""
DATA VISUALIZATION libraries:
1. Matplotlib: Low-level, flexible, lots of control
2. Seaborn: High-level, statistical visualizations
3. Plotly: Interactive, web-based visualizations
4. Pandas plot: Quick plotting from DataFrames

VISUALIZATION WORKFLOW:
1. Create figure and axes
2. Plot data
3. Customize (labels, title, colors)
4. Save or show
"""

import matplotlib.pyplot as plt
import seaborn as sns

# 3.1 MATPLOTLIB BASICS
# ============================================================================

"""
Figure: Canvas for plot (like a page)
Axes: Area inside figure where data is plotted (like a graph)

SYNTAX:
fig, ax = plt.subplots(figsize=(width, height))
"""

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data
ax.plot([1, 2, 3, 4], [10, 20, 25, 30])

# Customize
ax.set_title("Example Plot", fontsize=14, fontweight='bold')
ax.set_xlabel("X-axis label", fontsize=12)
ax.set_ylabel("Y-axis label", fontsize=12)

# Save
plt.savefig("plot.png", dpi=300, bbox_inches='tight')

# Show
plt.show()

# Close (important for memory management)
plt.close()

# TYPES OF PLOTS:
# ──────────────

# 1. Line plot
ax.plot(x_data, y_data, color='blue', linewidth=2, marker='o')

# 2. Histogram (distribution)
ax.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
# bins: number of bars
# alpha: transparency (0-1, where 1 is opaque)
# edgecolor: outline color

# 3. Scatter plot
scatter = ax.scatter(x, y, s=100, c=colors, cmap='viridis', alpha=0.6)
# s: marker size
# c: color values
# cmap: color map (viridis, coolwarm, plasma, etc.)
# Add colorbar: cbar = plt.colorbar(scatter, ax=ax)

# 4. Bar plot
ax.bar(categories, values, color=['red', 'green', 'blue'], alpha=0.7)
ax.barh(categories, values)  # Horizontal

# 5. Box plot
ax.boxplot(data)  # Shows median, quartiles, outliers

# CUSTOMIZATION:
# ──────────────

# Axes labels and title
ax.set_xlabel("Label", fontsize=12)
ax.set_ylabel("Label", fontsize=12)
ax.set_title("Title", fontsize=14, fontweight='bold')

# Set axis limits
ax.set_xlim(0, 100)
ax.set_ylim(0, 300)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Grid
ax.grid(True, alpha=0.3)

# Legend
ax.legend(loc='upper right', fontsize=10)

# Add text box
ax.text(0.05, 0.95, 'Text content', transform=ax.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


# 3.2 SEABORN - STATISTICAL VISUALIZATIONS
# ============================================================================

"""
SEABORN makes statistical plots easier and prettier than matplotlib.
Built on top of matplotlib.

COMMON PLOTS:
1. violinplot: Distribution by category
2. boxplot: Box-and-whisker plot
3. barplot: Bar chart with error bars
4. histplot: Histogram with distribution
5. scatterplot: Scatter with options
6. heatmap: Correlation matrix
"""

import seaborn as sns

# Set style
sns.set_style("whitegrid")  # Other options: darkgrid, dark, white, ticks

# 1. VIOLIN PLOT (shows full distribution)
sns.violinplot(data=df, x='School_Type', y='JAMB_Score', palette='muted')
# palette: color scheme (muted, deep, pastel, husl, etc.)

# 2. BOX PLOT
sns.boxplot(data=df, x='School_Location', y='JAMB_Score', hue='Gender')
# hue: adds another dimension using color

# 3. BAR PLOT (with error bars)
sns.barplot(data=df, x='Socioeconomic_Status', y='JAMB_Score', errorbar='sd')
# errorbar: 'sd' (std dev), 'se' (std error), None

# 4. SCATTER PLOT
sns.scatterplot(data=df, x='Study_Hours_Per_Week', y='JAMB_Score', 
                hue='School_Type', size='Age')

# 5. HISTOGRAM
sns.histplot(data=df, x='JAMB_Score', bins=20, kde=True)
# kde: add kernel density estimate line

# 6. HEATMAP (correlation matrix)
correlation = df[['JAMB_Score', 'Study_Hours_Per_Week', 'Attendance_Rate']].corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0)
# annot: show values in cells
# fmt: number format
# center: center colormap at value


# 3.3 MULTIPLE SUBPLOTS
# ============================================================================

"""
Create multiple plots in one figure for comparison.
"""

# Create grid of subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# (rows, columns)

# Access each subplot
axes[0].plot(x, y)
axes[0].set_title("Plot 1")

axes[1].bar(categories, values)
axes[1].set_title("Plot 2")

# Or more explicitly
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(data1)  # Top-left
axes[0, 1].plot(data2)  # Top-right
axes[1, 0].plot(data3)  # Bottom-left
axes[1, 1].plot(data4)  # Bottom-right

# Used in our code:
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# This creates 1 row, 2 columns of subplots


# ============================================================================
# 4. MACHINE LEARNING FUNDAMENTALS
# ============================================================================

"""
MACHINE LEARNING: Systems that learn from data and make predictions

TYPES:
1. SUPERVISED: Learn from labeled data (input → output)
2. UNSUPERVISED: Find patterns in unlabeled data
3. REINFORCEMENT: Learn through rewards

WORKFLOW:
1. Load data
2. Preprocess (encode, scale)
3. Split train/test
4. Choose model
5. Train on training data
6. Evaluate on test data
7. Tune hyperparameters
8. Deploy
"""

# 4.1 SUPERVISED LEARNING
# ============================================================================

"""
Two main types:
1. REGRESSION: Predict continuous values (0-300)
2. CLASSIFICATION: Predict categories (Low/Medium/High)
"""

# REGRESSION - Predicting JAMB Scores
# ──────────────────────────────────────

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# 1. LINEAR REGRESSION
# ────────────────────
# Assumes linear relationship: y = m*x + b
# Equation: JAMB_Score = w0 + w1*Study_Hours + w2*Attendance + ...

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)  # Learn weights (w0, w1, w2, ...)
predictions = lr_model.predict(X_test)  # Make predictions

# Get coefficients (weights)
coefficients = lr_model.coef_  # [w1, w2, w3, ...]
intercept = lr_model.intercept_  # w0

# 2. RANDOM FOREST REGRESSION
# ─────────────────────────────
# Ensemble of decision trees
# Each tree makes predictions, average them

rf_model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Maximum depth of tree
    random_state=42,       # For reproducibility
    n_jobs=-1             # Use all processors
)

rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)

# Feature importance (how much each feature contributes)
importances = rf_model.feature_importances_  # [0.15, 0.20, 0.10, ...]

# 3. XGBOOST REGRESSION
# ──────────────────────
# Gradient Boosting: Sequential trees, each corrects previous

xgb_model = xgb.XGBRegressor(
    n_estimators=100,      # Number of trees
    max_depth=5,           # Tree depth
    learning_rate=0.1,     # Step size
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)

# CLASSIFICATION - Predicting Performance Tier
# ────────────────────────────────────────────

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# First, create target classes
y_class = pd.cut(y, bins=[0, 150, 220, 500], labels=['Low', 'Medium', 'High'])

# 1. LOGISTIC REGRESSION (Binary)
# ────────────────────────────────
# Predicts probability (0-1) of class

y_binary = (y >= 150).astype(int)  # 1 if Pass, 0 if Fail

lr_class = LogisticRegression(max_iter=1000, random_state=42)
lr_class.fit(X_train, y_train_binary)

predictions = lr_class.predict(X_test)  # [0, 1, 1, 0, ...]
probabilities = lr_class.predict_proba(X_test)  # [[0.9, 0.1], [0.2, 0.8], ...]

# 2. RANDOM FOREST CLASSIFIER (Multi-class)
# ──────────────────────────────────────────
# Handles multiple classes naturally

rf_class = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_class.fit(X_train_class, y_train_class)

predictions = rf_class.predict(X_test_class)  # ['Low', 'High', 'Medium', ...]
probabilities = rf_class.predict_proba(X_test_class)  # [[0.2, 0.5, 0.3], ...]


# 4.2 UNSUPERVISED LEARNING
# ============================================================================

"""
No labels provided. Goal: Find natural groupings (clusters) or patterns.
"""

# K-MEANS CLUSTERING
# ──────────────────
# Groups data into K clusters based on similarity

from sklearn.cluster import KMeans

# Create model with k=4 clusters
kmeans = KMeans(
    n_clusters=4,          # Number of clusters
    random_state=42,
    n_init=10              # Number of times to run
)

# Fit and predict
clusters = kmeans.fit_predict(X_scaled)  # Returns cluster labels [0, 1, 2, 3, ...]

# Cluster centers
centers = kmeans.cluster_centers_  # Coordinates of cluster centers

# Inertia (within-cluster sum of squares - lower is better)
inertia = kmeans.inertia_

# DIMENSIONALITY REDUCTION - PCA
# ───────────────────────────────
# Reduce features while preserving information

from sklearn.decomposition import PCA

# Reduce 17 features to 3 principal components
pca = PCA(n_components=3)

# Fit and transform
X_pca = pca.fit_transform(X_scaled)  # (200, 3) instead of (200, 17)

# Explained variance
explained_variance = pca.explained_variance_ratio_
# [0.35, 0.25, 0.18]  → first 3 components explain 78% of variance

# Use for visualization (reduce to 2D)
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)


# 4.3 MODEL EVALUATION
# ============================================================================

"""
HOW TO MEASURE IF MODEL IS GOOD?

REGRESSION METRICS:
1. R² (R-squared): Percentage of variance explained (0-1, higher better)
   - 0.9 = Excellent, 0.7 = Good, 0.5 = Okay, <0.3 = Poor

2. MAE (Mean Absolute Error): Average absolute difference
   - Lower is better
   - In same units as target

3. RMSE (Root Mean Squared Error): Similar to MAE but penalizes large errors
   - Lower is better
   - Same units as target
   - More sensitive to outliers

4. MSE (Mean Squared Error): Average squared error
   - Lower is better
"""

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Predictions vs actual
y_pred = model.predict(X_test)
y_actual = y_test

# R² Score (higher is better, max 1.0)
r2 = r2_score(y_actual, y_pred)

# MAE (lower is better)
mae = mean_absolute_error(y_actual, y_pred)

# RMSE (lower is better)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

"""
CLASSIFICATION METRICS:
1. Accuracy: % of correct predictions (not good for imbalanced data)
2. Precision: Of predicted positive, how many were correct?
3. Recall (Sensitivity): Of actual positive, how many did we find?
4. F1 Score: Harmonic mean of Precision and Recall (balanced metric)
5. Confusion Matrix: Breakdown of correct/incorrect predictions
"""

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)

y_pred = model.predict(X_test)

# Accuracy (higher is better)
accuracy = accuracy_score(y_test, y_pred)

# Precision (higher is better)
precision = precision_score(y_test, y_pred)

# Recall (higher is better)
recall = recall_score(y_test, y_pred)

# F1 Score (higher is better, balanced)
f1 = f1_score(y_test, y_pred)

# Weighted F1 (for multi-class)
f1_weighted = f1_score(y_test, y_pred, average='weighted')

# Detailed report
print(classification_report(y_test, y_pred))

"""
CLUSTERING METRICS:
1. Silhouette Score: -1 to 1 (higher is better)
   - >0.5: Good clusters
   - 0.25-0.5: Okay
   - <0.25: Overlapping clusters
"""

from sklearn.metrics import silhouette_score

silhouette = silhouette_score(X_scaled, clusters)


# ============================================================================
# 5. STREAMLIT DASHBOARD CONCEPTS
# ============================================================================

"""
STREAMLIT: Turn Python scripts into interactive web apps
No HTML/CSS/JavaScript needed!

WORKFLOW:
1. Write Python functions
2. Use st.* functions for UI
3. Run with: streamlit run app.py
4. Automatically creates web app
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# 5.1 PAGE CONFIGURATION
# ============================================================================

# Set page title, icon, layout
st.set_page_config(
    page_title="JAMB Analytics",
    page_icon="📊",
    layout="wide",              # wide or centered
    initial_sidebar_state="expanded"  # expanded or collapsed
)

# 5.2 DISPLAYING CONTENT
# ============================================================================

# Text
st.title("Main Title")
st.header("Header")
st.subheader("Subheader")
st.write("Normal text")
st.text("Monospace text")

# Markdown
st.markdown("# Title in Markdown")
st.markdown("**Bold** and *italic* text")

# Metrics (KPI cards)
st.metric(label="Average Score", value=175, delta="+5")

# Columns (layout)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Students", 200)

with col2:
    st.metric("Avg Score", 175)

with col3:
    st.metric("High Performers", 45)

# 5.3 USER INPUT
# ============================================================================

# Slider
score = st.slider("Select score", 100, 300, 200)  # min, max, default

# Selectbox (dropdown)
school = st.selectbox("Choose school", ["Public", "Private"])

# Radio (single choice)
choice = st.radio("Choose option", ["Option 1", "Option 2"])

# Text input
name = st.text_input("Enter name")

# Button
if st.button("Click me"):
    st.write("Button clicked!")

# 5.4 DISPLAYING DATA & VISUALIZATIONS
# ============================================================================

# DataFrame
st.dataframe(df)

# Table
st.table(df.head())

# Matplotlib plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 2, 3])
st.pyplot(fig)

# Plotly (interactive)
fig = px.scatter(df, x='Study_Hours', y='JAMB_Score')
st.plotly_chart(fig, use_container_width=True)

# 5.5 CACHING (PERFORMANCE OPTIMIZATION)
# ============================================================================

"""
CACHING: Store function results to avoid recomputation
Speeds up app when running multiple times

@st.cache_resource: For models, databases (not time-sensitive)
@st.cache_data: For datasets (time-sensitive)
"""

@st.cache_resource
def load_models():
    """Load ML models once at startup"""
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    """Load dataset once at startup"""
    df = pd.read_csv("data.csv")
    return df

# Usage
models = load_models()
data = load_data()

# 5.6 NAVIGATION & TABS
# ============================================================================

# Sidebar for navigation
page = st.sidebar.radio("Navigate", ["Home", "Predictions", "Analysis"])

if page == "Home":
    st.write("Home page")
elif page == "Predictions":
    st.write("Predictions page")
else:
    st.write("Analysis page")

# Tabs
tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])

with tab1:
    st.write("Content 1")

with tab2:
    st.write("Content 2")

with tab3:
    st.write("Content 3")

# 5.7 COMMON PATTERNS
# ============================================================================

# Pattern 1: Take input and predict
# ─────────────────────────────────
input_data = {}
input_data['Study_Hours'] = st.slider("Study Hours", 0, 40)
input_data['Attendance'] = st.slider("Attendance", 50, 100)

if st.button("Predict"):
    # Prepare data
    scaled_data = scaler.transform([input_data.values()])
    
    # Predict
    prediction = model.predict(scaled_data)[0]
    
    # Display
    st.metric("Predicted Score", f"{prediction:.0f}")

# Pattern 2: Display DataFrame filtered by user selection
# ────────────────────────────────────────────────────────
selected_type = st.selectbox("School Type", df['School_Type'].unique())
filtered_df = df[df['School_Type'] == selected_type]
st.dataframe(filtered_df)

# Pattern 3: Multiple visualizations
# ───────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x='JAMB_Score')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(df, x='Study_Hours', y='JAMB_Score')
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PUTTING IT ALL TOGETHER - WORKFLOW
# ============================================================================

"""
COMPLETE ML PROJECT WORKFLOW:

1. LOAD DATA
   └─ df = pd.read_csv("data.csv")
   └─ Check shape, types, missing values

2. EXPLORATORY DATA ANALYSIS (EDA)
   └─ Describe data with statistics
   └─ Visualize distributions
   └─ Find correlations
   └─ Identify outliers

3. DATA PREPROCESSING
   ├─ Handle missing values
   ├─ Encode categorical variables
   ├─ Scale numeric features
   └─ Create new features if needed

4. FEATURE ENGINEERING
   ├─ Select relevant features
   ├─ Create polynomial features
   └─ Drop highly correlated features

5. SPLIT DATA
   └─ X_train, X_test, y_train, y_test

6. TRAIN MODELS
   ├─ Train multiple models
   ├─ Linear Regression
   ├─ Random Forest
   ├─ XGBoost
   └─ Compare performance

7. EVALUATE MODELS
   ├─ Calculate metrics (R², MAE, RMSE)
   ├─ Cross-validation
   └─ Select best model

8. HYPERPARAMETER TUNING
   ├─ Grid search
   ├─ Random search
   └─ Optimization

9. DEPLOY & MONITOR
   ├─ Save model as pickle
   ├─ Create prediction interface
   ├─ Deploy to production
   └─ Monitor performance

10. ITERATE
    └─ Collect feedback, improve
"""

# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

"""
PANDAS GROUPBY:
df.groupby('column')['target'].operation()
→ Groups rows by 'column' and applies operation to 'target' in each group
→ Used for: avg score by school, count by class, etc.

ENCODING:
LabelEncoder transforms text to numbers (0, 1, 2, ...)
→ Needed for ML models

SCALING:
StandardScaler transforms features to mean=0, std=1
→ Prevents large-range features from dominating

TRAIN-TEST SPLIT:
Divide data to train on known examples, evaluate on unseen
→ Prevents overfitting, gives realistic performance

REGRESSION vs CLASSIFICATION:
→ Regression: Predict continuous values (price, score)
→ Classification: Predict categories (pass/fail, tier)

SUPERVISED vs UNSUPERVISED:
→ Supervised: Learn from labeled examples
→ Unsupervised: Find patterns in data

MODEL EVALUATION:
→ R² Score: How well model fits (regression)
→ Accuracy: % correct (classification)
→ F1 Score: Balanced metric (classification)

STREAMLIT:
→ Convert Python scripts to web apps
→ Use st.* functions for UI
→ Automatic rerun on input changes
"""

# ============================================================================
# NEXT STEPS FOR LEARNING
# ============================================================================

"""
1. EXPERIMENT
   - Try different models
   - Adjust hyperparameters
   - Create new features
   - Use different datasets

2. DEEP LEARNING
   - Neural Networks (PyTorch, TensorFlow)
   - CNNs for images
   - RNNs for sequences

3. PRODUCTION
   - Containerization (Docker)
   - API creation (FastAPI)
   - Cloud deployment (AWS, Google Cloud, Azure)

4. ADVANCED ML
   - Ensemble methods
   - XGBoost, LightGBM
   - Time series forecasting
   - NLP

5. BEST PRACTICES
   - Version control (Git)
   - Code documentation
   - Testing
   - Monitoring
"""
