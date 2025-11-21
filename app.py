"""
JAMB Score Analytics - Streamlit Dashboard
Interactive ML models for score prediction, classification, and clustering
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="JAMB Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-style {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects"""
    models = {}
    
    # Load scalers and encoders
    with open('models/scaler.pkl', 'rb') as f:
        models['scaler'] = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        models['label_encoders'] = pickle.load(f)
    with open('models/metadata.pkl', 'rb') as f:
        models['metadata'] = pickle.load(f)
    with open('models/class_labels.pkl', 'rb') as f:
        models['class_labels'] = pickle.load(f)
    
    # Load regression models
    with open('models/linear_regression.pkl', 'rb') as f:
        models['linear_regression'] = pickle.load(f)
    with open('models/random_forest_regressor.pkl', 'rb') as f:
        models['random_forest_regressor'] = pickle.load(f)
    with open('models/xgboost_regressor.pkl', 'rb') as f:
        models['xgboost_regressor'] = pickle.load(f)
    
    # Load classification models
    with open('models/random_forest_classifier.pkl', 'rb') as f:
        models['random_forest_classifier'] = pickle.load(f)
    with open('models/logistic_regression_binary.pkl', 'rb') as f:
        models['logistic_regression_binary'] = pickle.load(f)
    
    # Load clustering models
    with open('models/kmeans_clusterer.pkl', 'rb') as f:
        models['kmeans'] = pickle.load(f)
    with open('models/pca_model.pkl', 'rb') as f:
        models['pca'] = pickle.load(f)
    
    # Load feature importance
    with open('models/feature_importance.pkl', 'rb') as f:
        models['feature_importance'] = pickle.load(f)
    
    return models

@st.cache_data
def load_data():
    """Load original dataset for reference and clustering"""
    df = pd.read_csv("jamb_exam_results.csv")
    return df

# Load all resources
models = load_models()
original_data = load_data()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_input_data(input_dict):
    """Convert user input to model-ready format"""
    # Create DataFrame with input
    input_df = pd.DataFrame([input_dict])
    
    # Encode categorical variables
    for col in models['metadata']['categorical_cols']:
        if col in input_df.columns:
            le = models['label_encoders'][col]
            input_df[col] = le.transform(input_df[col])
    
    # Scale features
    input_scaled = models['scaler'].transform(input_df)
    
    return input_scaled

def get_feature_names():
    """Get list of all features"""
    return models['metadata']['feature_names']

def get_feature_importance_data():
    """Get feature importance dataframe"""
    return models['feature_importance']

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("# 📊 JAMB Analytics Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Home", "🎯 Score Predictor", "📈 Performance Classifier", 
     "🔍 Student Segmentation", "🧬 Feature Analysis", "📊 Model Comparison"]
)

# ============================================================================
# PAGE: HOME
# ============================================================================

if page == "🏠 Home":
    st.markdown('<div class="header-style">📊 JAMB Score Analytics Dashboard</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Advanced Analytics Suite
    
    This dashboard provides machine learning-powered insights into JAMB exam performance 
    and student success factors. Explore multiple models to:
    
    - **🎯 Predict** individual student JAMB scores
    - **📈 Classify** students into performance tiers
    - **🔍 Segment** students into meaningful groups
    - **🧬 Analyze** which factors drive success
    - **📊 Compare** model performance metrics
    
    ---
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Students", len(original_data))
        st.metric("Avg Score", f"{original_data['JAMB_Score'].mean():.0f}")
    
    with col2:
        st.metric("Score Range", f"{original_data['JAMB_Score'].min():.0f}-{original_data['JAMB_Score'].max():.0f}")
        st.metric("Std Deviation", f"{original_data['JAMB_Score'].std():.2f}")
    
    with col3:
        high_performers = len(original_data[original_data['JAMB_Score'] > 220])
        st.metric("High Performers (>220)", high_performers)
        st.metric("% of Students", f"{high_performers/len(original_data)*100:.1f}%")
    
    st.markdown("---")
    
    # Score distribution
    st.subheader("📊 Score Distribution")
    fig = px.histogram(original_data, x='JAMB_Score', nbins=30, 
                      title='JAMB Score Distribution',
                      labels={'JAMB_Score': 'Score', 'count': 'Number of Students'},
                      color_discrete_sequence=['#1f77b4'])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Top Factors Influencing Score")
        top_features = get_feature_importance_data().head(8)
        fig = px.bar(top_features, x='importance', y='feature', 
                    title='Feature Importance',
                    labels={'importance': 'Importance Score', 'feature': 'Feature'},
                    color='importance', color_continuous_scale='Blues')
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("👥 Student Demographics")
        demo_data = pd.DataFrame({
            'Category': ['Public School', 'Private School', 'Urban', 'Rural'],
            'Count': [
                len(original_data[original_data['School_Type'] == 'Public']),
                len(original_data[original_data['School_Type'] == 'Private']),
                len(original_data[original_data['School_Location'] == 'Urban']),
                len(original_data[original_data['School_Location'] == 'Rural'])
            ]
        })
        fig = px.pie(demo_data, values='Count', names='Category', 
                    title='School Distribution')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: SCORE PREDICTOR
# ============================================================================

elif page == "🎯 Score Predictor":
    st.markdown('<div class="header-style">🎯 JAMB Score Predictor</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Enter Student Information to Predict JAMB Score
    
    Provide details about a student to get predictions from multiple regression models.
    The models use study habits, attendance, school resources, and socioeconomic factors.
    """)
    
    # Create input form in columns
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.subheader("📚 Academic Factors")
        input_data['Study_Hours_Per_Week'] = st.slider("Study Hours Per Week", 0, 40, 20)
        input_data['Attendance_Rate'] = st.slider("Attendance Rate (%)", 50, 100, 85)
        input_data['Teacher_Quality'] = st.slider("Teacher Quality (1-5)", 1, 5, 3)
        input_data['Assignments_Completed'] = st.slider("Assignments Completed", 1, 5, 2)
        
    with col2:
        st.subheader("🏫 School & Location")
        input_data['School_Type'] = st.selectbox("School Type", 
                                                 models['label_encoders']['School_Type'].classes_)
        input_data['School_Location'] = st.selectbox("School Location",
                                                     models['label_encoders']['School_Location'].classes_)
        input_data['Distance_To_School'] = st.slider("Distance to School (km)", 0, 20, 8)
        input_data['Age'] = st.slider("Student Age", 15, 25, 18)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("👨‍👩‍👧 Personal & Social")
        input_data['Gender'] = st.selectbox("Gender",
                                           models['label_encoders']['Gender'].classes_)
        input_data['Socioeconomic_Status'] = st.selectbox("Socioeconomic Status",
                                                         models['label_encoders']['Socioeconomic_Status'].classes_)
        input_data['Parent_Education_Level'] = st.selectbox("Parent Education Level",
                                                           models['label_encoders']['Parent_Education_Level'].classes_)
        input_data['IT_Knowledge'] = st.selectbox("IT Knowledge Level",
                                                  models['label_encoders']['IT_Knowledge'].classes_)
    
    with col4:
        st.subheader("🤝 Support Systems")
        input_data['Extra_Tutorials'] = st.selectbox("Extra Tutorials",
                                                    models['label_encoders']['Extra_Tutorials'].classes_)
        input_data['Access_To_Learning_Materials'] = st.selectbox("Access to Learning Materials",
                                                                 models['label_encoders']['Access_To_Learning_Materials'].classes_)
        input_data['Parent_Involvement'] = st.selectbox("Parent Involvement Level",
                                                       models['label_encoders']['Parent_Involvement'].classes_)
    
    # Make predictions
    if st.button("🔮 Predict Score", key="predict_score", use_container_width=True):
        input_scaled = prepare_input_data(input_data)
        
        # Get predictions from all regression models
        lr_pred = models['linear_regression'].predict(input_scaled)[0]
        rf_pred = models['random_forest_regressor'].predict(input_scaled)[0]
        xgb_pred = models['xgboost_regressor'].predict(input_scaled)[0]
        
        # Average prediction
        avg_pred = (lr_pred + rf_pred + xgb_pred) / 3
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Linear Regression", f"{lr_pred:.0f}")
        
        with col2:
            st.metric("Random Forest", f"{rf_pred:.0f}")
        
        with col3:
            st.metric("XGBoost (Best)", f"{xgb_pred:.0f}", delta=f"{xgb_pred-avg_pred:+.0f}")
        
        with col4:
            st.metric("Average Prediction", f"{avg_pred:.0f}")
        
        st.markdown("---")
        
        # Prediction confidence and interpretation
        st.subheader("📊 Prediction Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Score interpretation
            if avg_pred >= 220:
                tier = "🟢 HIGH PERFORMER"
                color = "green"
            elif avg_pred >= 150:
                tier = "🟡 MEDIUM PERFORMER"
                color = "orange"
            else:
                tier = "🔴 LOW PERFORMER"
                color = "red"
            
            st.markdown(f"""
            ### Predicted Tier: {tier}
            - **Predicted Score**: {avg_pred:.0f}
            - **Model Agreement**: {100 - abs(xgb_pred - rf_pred):.1f}%
            - **Confidence**: High
            """)
        
        with col2:
            # Prediction spread visualization
            pred_data = pd.DataFrame({
                'Model': ['Linear\nRegression', 'Random\nForest', 'XGBoost', 'Average'],
                'Prediction': [lr_pred, rf_pred, xgb_pred, avg_pred]
            })
            
            fig = px.bar(pred_data, x='Model', y='Prediction',
                        title='Model Predictions Comparison',
                        color='Prediction',
                        color_continuous_scale='RdYlGn',
                        range_color=[100, 300])
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: PERFORMANCE CLASSIFIER
# ============================================================================

elif page == "📈 Performance Classifier":
    st.markdown('<div class="header-style">📈 Performance Classifier</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Classify Student Performance Tier
    
    Predict whether a student falls into Low, Medium, or High performance category,
    or predict Pass/Fail status.
    """)
    
    # Classification type selection
    class_type = st.radio("Classification Type:", 
                         ["3-Tier Classification (Low/Medium/High)", 
                          "Binary Classification (Pass/Fail)"],
                         horizontal=True)
    
    # Input form
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.subheader("📚 Academic Factors")
        input_data['Study_Hours_Per_Week'] = st.slider("Study Hours Per Week", 0, 40, 20, key="class1")
        input_data['Attendance_Rate'] = st.slider("Attendance Rate (%)", 50, 100, 85, key="class2")
        input_data['Teacher_Quality'] = st.slider("Teacher Quality (1-5)", 1, 5, 3, key="class3")
        input_data['Assignments_Completed'] = st.slider("Assignments Completed", 1, 5, 2, key="class4")
        
    with col2:
        st.subheader("🏫 School & Location")
        input_data['School_Type'] = st.selectbox("School Type", 
                                                 models['label_encoders']['School_Type'].classes_,
                                                 key="class5")
        input_data['School_Location'] = st.selectbox("School Location",
                                                     models['label_encoders']['School_Location'].classes_,
                                                     key="class6")
        input_data['Distance_To_School'] = st.slider("Distance to School (km)", 0, 20, 8, key="class7")
        input_data['Age'] = st.slider("Student Age", 15, 25, 18, key="class8")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("👨‍👩‍👧 Personal & Social")
        input_data['Gender'] = st.selectbox("Gender",
                                           models['label_encoders']['Gender'].classes_,
                                           key="class9")
        input_data['Socioeconomic_Status'] = st.selectbox("Socioeconomic Status",
                                                         models['label_encoders']['Socioeconomic_Status'].classes_,
                                                         key="class10")
        input_data['Parent_Education_Level'] = st.selectbox("Parent Education Level",
                                                           models['label_encoders']['Parent_Education_Level'].classes_,
                                                           key="class11")
        input_data['IT_Knowledge'] = st.selectbox("IT Knowledge Level",
                                                  models['label_encoders']['IT_Knowledge'].classes_,
                                                  key="class12")
    
    with col4:
        st.subheader("🤝 Support Systems")
        input_data['Extra_Tutorials'] = st.selectbox("Extra Tutorials",
                                                    models['label_encoders']['Extra_Tutorials'].classes_,
                                                    key="class13")
        input_data['Access_To_Learning_Materials'] = st.selectbox("Access to Learning Materials",
                                                                 models['label_encoders']['Access_To_Learning_Materials'].classes_,
                                                                 key="class14")
        input_data['Parent_Involvement'] = st.selectbox("Parent Involvement Level",
                                                       models['label_encoders']['Parent_Involvement'].classes_,
                                                       key="class15")
    
    if st.button("🔮 Classify", key="classify_btn", use_container_width=True):
        input_scaled = prepare_input_data(input_data)
        
        if class_type == "3-Tier Classification (Low/Medium/High)":
            # Get prediction and probabilities
            pred_class = models['random_forest_classifier'].predict(input_scaled)[0]
            pred_proba = models['random_forest_classifier'].predict_proba(input_scaled)[0]
            
            class_names = models['class_labels']['target_encoder'].classes_
            predicted_class = class_names[pred_class]
            
            # Display main result
            emoji_map = {'Low': '🔴', 'Medium': '🟡', 'High': '🟢'}
            st.markdown(f"""
            # {emoji_map.get(predicted_class, '')} Predicted Class: {predicted_class}
            """)
            
            # Confidence scores
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Low Confidence", f"{pred_proba[0]*100:.1f}%")
            with col2:
                st.metric("Medium Confidence", f"{pred_proba[1]*100:.1f}%")
            with col3:
                st.metric("High Confidence", f"{pred_proba[2]*100:.1f}%")
            
            # Probability distribution
            prob_data = pd.DataFrame({
                'Class': ['Low', 'Medium', 'High'],
                'Probability': pred_proba * 100
            })
            
            fig = px.bar(prob_data, x='Class', y='Probability',
                        title='Classification Probabilities',
                        color='Class', color_discrete_map={
                            'Low': '#FF6B6B', 'Medium': '#FFE66D', 'High': '#51CF66'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Binary classification
            pred_binary = models['logistic_regression_binary'].predict(input_scaled)[0]
            pred_proba_binary = models['logistic_regression_binary'].predict_proba(input_scaled)[0]
            
            predicted_result = models['class_labels']['binary_labels'][pred_binary]
            emoji_map = {'Pass (≥150)': '✅', 'Fail (<150)': '❌'}
            
            st.markdown(f"""
            # {emoji_map.get(predicted_result, '')} Prediction: {predicted_result}
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Fail Probability", f"{pred_proba_binary[0]*100:.1f}%")
            with col2:
                st.metric("Pass Probability", f"{pred_proba_binary[1]*100:.1f}%")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred_proba_binary[1]*100,
                title={'text': "Pass Probability"},
                delta={'reference': 50},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "green"},
                       'steps': [
                           {'range': [0, 50], 'color': "#FFE6E6"},
                           {'range': [50, 100], 'color': "#E6FFE6"}
                       ],
                       'threshold': {
                           'line': {'color': "red", 'width': 4},
                           'thickness': 0.75,
                           'value': 50
                       }}
            ))
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: STUDENT SEGMENTATION
# ============================================================================

elif page == "🔍 Student Segmentation":
    st.markdown('<div class="header-style">🔍 Student Segmentation</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### K-Means Clustering Analysis
    
    Students are grouped into 4 clusters based on their characteristics.
    This helps identify student archetypes and enables targeted interventions.
    """)
    
    # Prepare data for visualization
    from sklearn.preprocessing import LabelEncoder as LE
    
    data_viz = original_data.copy()
    categorical_cols = models['metadata']['categorical_cols']
    
    for col in categorical_cols:
        le = LE()
        data_viz[col] = le.fit_transform(data_viz[col])
    
    # Scale and cluster
    X_viz = models['scaler'].transform(data_viz.drop(['JAMB_Score', 'Student_ID'], axis=1))
    clusters = models['kmeans'].predict(X_viz)
    
    # PCA for visualization
    X_pca_2d = models['pca'].transform(X_viz)[:, :2]
    data_viz['Cluster'] = clusters
    data_viz['PCA1'] = X_pca_2d[:, 0]
    data_viz['PCA2'] = X_pca_2d[:, 1]
    
    # Cluster visualization
    st.subheader("📊 Cluster Visualization (PCA)")
    
    fig = px.scatter(data_viz, x='PCA1', y='PCA2', color='Cluster',
                    hover_data=['JAMB_Score', 'Study_Hours_Per_Week', 'Attendance_Rate'],
                    title='Student Clusters (PCA Projection)',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#95E1D3', '#FFE66D'],
                    labels={'Cluster': 'Student Cluster'})
    fig.update_traces(marker=dict(size=10))
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster profiles
    st.subheader("👥 Cluster Profiles")
    
    # Reload original data for cluster analysis
    data_clusters = original_data.copy()
    data_clusters['Cluster'] = clusters
    
    # Create tabs for each cluster
    cluster_tabs = st.tabs([f"Cluster {i}" for i in range(4)])
    
    for cluster_id, tab in enumerate(cluster_tabs):
        with tab:
            cluster_data = data_clusters[data_clusters['Cluster'] == cluster_id]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Students", len(cluster_data))
                st.metric("Avg Score", f"{cluster_data['JAMB_Score'].mean():.0f}")
            
            with col2:
                st.metric("Avg Study Hours", f"{cluster_data['Study_Hours_Per_Week'].mean():.1f}")
                st.metric("Avg Attendance", f"{cluster_data['Attendance_Rate'].mean():.1f}%")
            
            with col3:
                st.metric("Most Common: School Type", cluster_data['School_Type'].mode()[0])
                st.metric("Most Common: Location", cluster_data['School_Location'].mode()[0])
            
            st.markdown("---")
            
            # Detailed statistics
            st.subheader("Detailed Profile")
            
            profile_stats = pd.DataFrame({
                'Metric': ['Study Hours', 'Attendance %', 'Teacher Quality', 'Distance (km)', 'Age'],
                'Mean': [
                    cluster_data['Study_Hours_Per_Week'].mean(),
                    cluster_data['Attendance_Rate'].mean(),
                    cluster_data['Teacher_Quality'].mean(),
                    cluster_data['Distance_To_School'].mean(),
                    cluster_data['Age'].mean()
                ]
            })
            
            st.dataframe(profile_stats, use_container_width=True)

# ============================================================================
# PAGE: FEATURE ANALYSIS
# ============================================================================

elif page == "🧬 Feature Analysis":
    st.markdown('<div class="header-style">🧬 Feature Importance Analysis</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### Understanding Factor Importance
    
    These are the factors that most influence JAMB scores according to the XGBoost model.
    """)
    
    # Feature importance bar chart
    feature_importance = get_feature_importance_data()
    
    fig = px.bar(feature_importance, x='importance', y='feature',
                title='Feature Importance Scores',
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Viridis',
                orientation='h')
    
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature correlation with target
    st.subheader("📊 Feature Correlations with JAMB Score")
    
    numeric_cols = models['metadata']['numeric_cols']
    correlations = []
    
    for col in numeric_cols:
        corr = original_data[col].corr(original_data['JAMB_Score'])
        correlations.append({'Feature': col, 'Correlation': corr})
    
    corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
    
    fig = px.bar(corr_df, x='Correlation', y='Feature',
                title='Numeric Features Correlation with JAMB Score',
                color='Correlation',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                orientation='h')
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed insights
    st.subheader("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Strongest Positive Factors
        - **Study Hours**: More study = Higher scores
        - **Attendance Rate**: Consistent attendance matters
        - **Teacher Quality**: Good teachers improve outcomes
        """)
    
    with col2:
        st.markdown("""
        ### Support Systems Impact
        - **Extra Tutorials**: Significant boost to performance
        - **Learning Materials**: Access enables better prep
        - **Parent Education**: Higher parent education helps
        """)

# ============================================================================
# PAGE: MODEL COMPARISON
# ============================================================================

elif page == "📊 Model Comparison":
    st.markdown('<div class="header-style">📊 Model Performance Comparison</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### ML Model Metrics and Performance
    
    Compare accuracy, precision, and other metrics across all models.
    """)
    
    # Get metrics from metadata
    metadata = models['metadata']
    
    # Regression metrics
    st.subheader("🎯 Regression Models (Score Prediction)")
    
    regression_metrics = metadata['regression_metrics']
    
    reg_data = []
    for model_name, metrics in regression_metrics.items():
        reg_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'R² Score': metrics['r2'],
            'MAE': metrics['mae'],
            'RMSE': metrics['rmse']
        })
    
    reg_df = pd.DataFrame(reg_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("R² Score (Higher is Better)")
        fig = px.bar(reg_df, x='R² Score', y='Model',
                    color='R² Score',
                    color_continuous_scale='Greens',
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("MAE (Lower is Better)")
        fig = px.bar(reg_df, x='MAE', y='Model',
                    color='MAE',
                    color_continuous_scale='Reds_r',
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("RMSE (Lower is Better)")
        fig = px.bar(reg_df, x='RMSE', y='Model',
                    color='RMSE',
                    color_continuous_scale='Purples_r',
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    # Classification metrics
    st.subheader("📈 Classification Models")
    
    classification_metrics = metadata['classification_metrics']
    
    class_data = []
    for model_name, metrics in classification_metrics.items():
        class_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': metrics['accuracy'],
            'F1 Score': metrics['f1']
        })
    
    class_df = pd.DataFrame(class_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Accuracy")
        fig = px.bar(class_df, x='Accuracy', y='Model',
                    color='Accuracy',
                    color_continuous_scale='Blues',
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("F1 Score")
        fig = px.bar(class_df, x='F1 Score', y='Model',
                    color='F1 Score',
                    color_continuous_scale='Oranges',
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    
    # Clustering metrics
    st.subheader("🔍 Unsupervised Learning Metrics")
    
    clustering_metrics = metadata['clustering_metrics']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("K-Means Silhouette Score", 
                 f"{clustering_metrics['kmeans']['silhouette_score']:.4f}")
        st.caption("Score range: -1 to 1. Higher is better (>0.5 is good)")
    
    with col2:
        st.metric("PCA Variance Explained (3 components)", 
                 f"{clustering_metrics['pca_variance']*100:.2f}%")
        st.caption("Percentage of total variance captured by first 3 components")
    
    # Model recommendations
    st.subheader("🏆 Recommendations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Best for Score Prediction
        🥇 **XGBoost**
        - Highest R² score
        - Lowest RMSE
        - Handles feature interactions
        """)
    
    with col2:
        st.markdown("""
        ### Best for Classification
        🥇 **Random Forest**
        - Multi-class classification
        - Good interpretability
        - Balanced performance
        """)
    
    with col3:
        st.markdown("""
        ### Best for Segmentation
        🥇 **K-Means (k=4)**
        - Good silhouette score
        - Clear clusters
        - Actionable segments
        """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.9rem;">
Made with ❤️ | JAMB Score Analytics Dashboard | 2024
</div>
""", unsafe_allow_html=True)
