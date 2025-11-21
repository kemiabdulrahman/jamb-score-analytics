# PRACTICAL EXERCISES - LEARN BY DOING

This guide provides hands-on exercises to reinforce your understanding of each concept.

---

## SECTION 1: PANDAS & DATA MANIPULATION

### Exercise 1.1: GroupBy Mastery
**Objective**: Master the GroupBy operation

**Task 1**: Calculate average scores by Gender
```python
import pandas as pd

df = pd.read_csv("jamb_exam_results.csv")

# TODO: Calculate average JAMB_Score for each Gender
# Hint: df.groupby('Gender')['JAMB_Score'].mean()

# Expected Output:
# Gender
# Female    185.2
# Male      172.8
```

**Solution**:
```python
avg_by_gender = df.groupby('Gender')['JAMB_Score'].mean()
print(avg_by_gender)
```

**Task 2**: Count students in each Socioeconomic_Status
```python
# TODO: Count students in each socioeconomic status
# Hint: Use .size() or .count()

# Expected Output:
# Socioeconomic_Status
# High      45
# Low       60
# Medium    95
```

**Solution**:
```python
count_by_ses = df.groupby('Socioeconomic_Status').size()
print(count_by_ses)
```

**Task 3**: Get multiple statistics at once
```python
# TODO: For each School_Type, get:
#   - Mean JAMB_Score
#   - Min JAMB_Score
#   - Max JAMB_Score
#   - Count of students

# Expected Output:
#             mean  min  max  count
# School_Type
# Private    185.4  100  298     45
# Public     190.2  100  274    155
```

**Solution**:
```python
stats_by_school = df.groupby('School_Type')['JAMB_Score'].agg(['mean', 'min', 'max', 'count'])
print(stats_by_school)
```

### Exercise 1.2: Data Filtering
**Objective**: Filter data based on conditions

**Task 1**: Find all high performers (score > 220)
```python
# TODO: Filter students with JAMB_Score > 220

# Expected: About 30-40 students
```

**Solution**:
```python
high_performers = df[df['JAMB_Score'] > 220]
print(f"High performers: {len(high_performers)}")
```

**Task 2**: Find urban students with high attendance (>90%)
```python
# TODO: Filter students where:
#   - School_Location == 'Urban'
#   - Attendance_Rate > 90

# Expected: About 15-20 students
```

**Solution**:
```python
urban_attentive = df[(df['School_Location'] == 'Urban') & 
                     (df['Attendance_Rate'] > 90)]
print(f"Urban + high attendance: {len(urban_attentive)}")
```

**Task 3**: Find students with Extra Tutorials who scored low
```python
# TODO: Filter students where:
#   - Extra_Tutorials == 'Yes'
#   - JAMB_Score < 150

# Question: How many students got low scores despite extra tutorials?
```

**Solution**:
```python
low_despite_tutoring = df[(df['Extra_Tutorials'] == 'Yes') & 
                          (df['JAMB_Score'] < 150)]
print(f"Low score + extra tutorials: {len(low_despite_tutoring)}")
```

### Exercise 1.3: Create New Features
**Objective**: Add new columns using calculations

**Task 1**: Create performance tier column
```python
# TODO: Create new column 'Tier' with values:
#   - 'High' if score > 220
#   - 'Medium' if 150 <= score <= 220
#   - 'Low' if score < 150

df['Tier'] = df['JAMB_Score'].apply(lambda score: 
    'High' if score > 220 else 'Medium' if score >= 150 else 'Low')

# Check
print(df[['JAMB_Score', 'Tier']].head(10))
```

**Task 2**: Calculate study efficiency (score / study hours)
```python
# TODO: Create new column 'Efficiency' = JAMB_Score / Study_Hours_Per_Week
# Handle division by zero

df['Efficiency'] = df.apply(
    lambda row: row['JAMB_Score'] / row['Study_Hours_Per_Week'] 
    if row['Study_Hours_Per_Week'] > 0 else 0,
    axis=1
)

print(df[['JAMB_Score', 'Study_Hours_Per_Week', 'Efficiency']].head())
```

---

## SECTION 2: ENCODING & PREPROCESSING

### Exercise 2.1: Encoding Categorical Variables
**Objective**: Convert text to numbers

**Task 1**: Manually encode School_Type
```python
from sklearn.preprocessing import LabelEncoder

# Create encoder
le = LabelEncoder()

# Fit and transform
df['School_Type_Code'] = le.fit_transform(df['School_Type'])

# Show mapping
print("Classes:", le.classes_)
print("Transformed values:", le.transform(['Public', 'Private']))

# Expected:
# Classes: ['Private' 'Public']
# Transformed values: [1 0]
```

**Task 2**: Encode multiple columns at once
```python
categorical_cols = ['Gender', 'School_Location', 'Parent_Involvement']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_Code'] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
```

### Exercise 2.2: Feature Scaling
**Objective**: Normalize features to same range

**Task 1**: Scale a single column
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit on study hours
study_hours = df['Study_Hours_Per_Week'].values.reshape(-1, 1)
study_scaled = scaler.fit_transform(study_hours)

# Check results
print(f"Original: mean={study_hours.mean():.2f}, std={study_hours.std():.2f}")
print(f"Scaled: mean={study_scaled.mean():.2f}, std={study_scaled.std():.2f}")

# Expected: Scaled mean ≈ 0, std ≈ 1
```

**Task 2**: Scale multiple numeric columns
```python
from sklearn.preprocessing import StandardScaler

numeric_cols = ['Study_Hours_Per_Week', 'Attendance_Rate', 'Age', 'Distance_To_School']

scaler = StandardScaler()
X_numeric = df[numeric_cols]
X_scaled = scaler.fit_transform(X_numeric)

# Verify
print(f"Shape: {X_scaled.shape}")
print(f"First row (scaled): {X_scaled[0]}")
print(f"All means close to 0? {np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)}")
```

---

## SECTION 3: VISUALIZATION

### Exercise 3.1: Create Custom Plots
**Objective**: Build visualizations from scratch

**Task 1**: Create histogram with statistics
```python
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Your code here
scores = df['JAMB_Score']

fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram
# TODO: Add histogram

# Add statistics box
# TODO: Calculate mean, median, std, skewness
# TODO: Add text box with stats

plt.savefig('my_histogram.png')
plt.show()
```

**Solution**:
```python
scores = df['JAMB_Score']

stats = {
    'mean': scores.mean(),
    'median': scores.median(),
    'std': scores.std(),
    'skewness': skew(scores)
}

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(scores, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax.set_title('JAMB Score Distribution')
ax.set_xlabel('Score')
ax.set_ylabel('Frequency')

stats_text = f"Mean: {stats['mean']:.1f}\nMedian: {stats['median']:.0f}\nStd: {stats['std']:.2f}"
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.savefig('my_histogram.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Task 2**: Create comparison plot with subplots
```python
import seaborn as sns

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# TODO: Plot 1 (top-left): Distribution by School_Type
# Hint: Use sns.violinplot

# TODO: Plot 2 (top-right): Correlation heatmap
# Hint: Use numeric columns only

# TODO: Plot 3 (bottom-left): Attendance vs Score
# Hint: Use scatter plot

# TODO: Plot 4 (bottom-right): Average score by Gender
# Hint: Use bar plot

plt.tight_layout()
plt.savefig('comparison.png', dpi=300)
plt.show()
```

---

## SECTION 4: MACHINE LEARNING

### Exercise 4.1: Train Your First Model
**Objective**: Build and evaluate a regression model

**Task**: Train Linear Regression
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Step 1: Prepare data
df = pd.read_csv("jamb_exam_results.csv")

# TODO: Encode categorical columns
# TODO: Select numeric features
# TODO: Create X (features) and y (target)

# Step 2: Scale features
# TODO: Create StandardScaler, fit and transform

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                      test_size=0.2, 
                                                      random_state=42)

# Step 4: Train model
# TODO: Create LinearRegression model
# TODO: Fit on training data

# Step 5: Evaluate
# TODO: Make predictions on test data
# TODO: Calculate R² score
# TODO: Calculate MAE

print(f"R² Score: {r2_score(y_test, predictions):.4f}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
```

### Exercise 4.2: Compare Multiple Models
**Objective**: Train and compare 3 regression models

**Task**: Train Linear Regression, Random Forest, and XGBoost
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Assume X_train, X_test, y_train, y_test already split

results = {}

# Model 1: Linear Regression
# TODO: Create, train, predict, evaluate

# Model 2: Random Forest
# TODO: Create with n_estimators=100, max_depth=10
# TODO: Train, predict, evaluate

# Model 3: XGBoost
# TODO: Create with n_estimators=100, max_depth=5
# TODO: Train, predict, evaluate

# Compare results
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
```

### Exercise 4.3: Classification
**Objective**: Build a classifier

**Task**: Binary classification (Pass/Fail)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Create binary target: 1 if score >= 150, 0 otherwise
y_binary = (y >= 150).astype(int)

# Split data
X_train, X_test, y_train_b, y_test_b = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42)

# Train Logistic Regression
# TODO: Create model
# TODO: Fit on training data

# Evaluate
# TODO: Make predictions
# TODO: Calculate accuracy and F1 score
# TODO: Print confusion matrix

print(f"Accuracy: {accuracy_score(y_test_b, predictions):.4f}")
print(f"F1 Score: {f1_score(y_test_b, predictions):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test_b, predictions)}")
```

### Exercise 4.4: Clustering
**Objective**: Perform K-Means clustering

**Task**: Find optimal K and analyze clusters
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Find optimal K
silhouette_scores = []

# TODO: Loop through k from 2 to 8
# TODO: For each k, train KMeans and calculate silhouette score

# Find best k
best_k = silhouette_scores.index(max(silhouette_scores)) + 2

# Train final model with best k
# TODO: Create KMeans with best_k

# Analyze clusters
for cluster_id in range(best_k):
    cluster_data = df[clusters == cluster_id]
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {len(cluster_data)}")
    print(f"  Avg Score: {cluster_data['JAMB_Score'].mean():.0f}")
    print(f"  Avg Study Hours: {cluster_data['Study_Hours_Per_Week'].mean():.1f}")
```

---

## SECTION 5: STREAMLIT PRACTICE

### Exercise 5.1: Create Simple Dashboard
**Objective**: Build first Streamlit app

**Task**: Create app.py
```python
import streamlit as st
import pandas as pd

# Load data
df = pd.read_csv("jamb_exam_results.csv")

# Page config
# TODO: Set page title, icon, layout

# Title
# TODO: Add main title

# Metrics row
col1, col2, col3 = st.columns(3)

# TODO: Display total students in col1
# TODO: Display average score in col2
# TODO: Display high performers in col3

# Data explorer
st.subheader("Data Explorer")

# TODO: Create dropdown to select school type
# TODO: Filter and display students from selected school

# Visualization
st.subheader("Score Distribution")

# TODO: Create and display histogram

# Run with: streamlit run app.py
```

### Exercise 5.2: Add Prediction Interface
**Objective**: Create prediction input form

**Task**: Extend your Streamlit app
```python
import streamlit as st
import pandas as pd
import pickle

# Load models
with open('models/linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Score Predictor")

# Input section
st.subheader("Student Information")

# TODO: Create sliders and selectboxes for:
# - Study_Hours_Per_Week (0-40)
# - Attendance_Rate (50-100)
# - Teacher_Quality (1-5)
# - School_Type (dropdown)
# - Gender (dropdown)
# - ... other features

# Prediction section
if st.button("Predict Score"):
    # TODO: Create input array
    # TODO: Scale input
    # TODO: Make prediction
    # TODO: Display result with formatting
    
    st.success(f"Predicted Score: {prediction:.0f}")
```

---

## CHALLENGE PROJECTS

### Challenge 1: Enhanced Visualization
Create a dashboard showing:
- Score distribution by school type
- Correlation heatmap
- Top 5 factors affecting scores
- Student demographics

### Challenge 2: Hyperparameter Tuning
Use GridSearchCV to find best parameters for Random Forest:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor()

grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### Challenge 3: Feature Engineering
Create new features:
- Study efficiency = Score / Study_Hours
- School resources = (Extra_Tutorials + Access_To_Learning_Materials) / 2
- Family support = (Parent_Involvement + Parent_Education_Level) / 2

Train model with new features and compare performance.

### Challenge 4: Cross-Validation
Implement k-fold cross-validation:
```python
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor()

scores = cross_val_score(model, X_scaled, y, cv=5, 
                        scoring='r2')

print(f"CV Scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
print(f"Std: {scores.std():.4f}")
```

---

## SOLUTIONS

Solutions to all exercises are available in the `SOLUTIONS.md` file.

---

## TIPS FOR SUCCESS

1. **Start Small**: Begin with simple operations, build complexity
2. **Print Everything**: Check data shape and values at each step
3. **Read Error Messages**: They tell you exactly what's wrong
4. **Experiment**: Try different parameters and see the effect
5. **Compare Results**: Understand why one approach works better
6. **Ask Questions**: Use print() statements to understand behavior
7. **Document Your Code**: Add comments explaining your logic
8. **Version Control**: Save your work with meaningful commit messages

---

## COMMON MISTAKES TO AVOID

❌ **Fitting scaler on test data**
```python
# WRONG
scaler.fit(X_test)  # Leaks information!
```

✅ **Correct**
```python
scaler.fit(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

❌ **Using same train/test split every time**
```python
# WRONG
random_state variable  # Inconsistent results
```

✅ **Correct**
```python
random_state=42  # Reproducible
```

---

❌ **Forgetting to scale features**
```python
# WRONG
model.fit(X_train, y_train)  # Unscaled features
```

✅ **Correct**
```python
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)
```

---

## NEXT STEPS

After mastering these exercises:
1. Try with different datasets
2. Combine multiple techniques
3. Deploy your models
4. Build production systems
5. Explore deep learning

Happy learning! 🚀
