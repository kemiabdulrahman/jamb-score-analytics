"""
JAMB Score Analytics - Data Visualization Suite
Analyzes student performance factors and their impact on exam scores
"""

from scipy.stats import skew, kurtosis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress seaborn point placement warnings (many overlapping points)
warnings.filterwarnings('ignore', category=UserWarning, message='.*cannot be placed.*')

# Load data
dataframe = pd.read_csv("jamb_exam_results.csv")

# Configure visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


# ============================================================================
# 1. HISTOGRAM - Score Distribution Analysis
# ============================================================================

def plot_histogram(df):
    """
    Visualizes JAMB score distribution with statistical metrics.
    Calculates and displays mean, median, mode, standard deviation, 
    skewness, and kurtosis.
    """
    scores = df["JAMB_Score"]
    
    # Calculate statistics
    stats = {
        'mean': scores.mean(),
        'median': scores.median(),
        'mode': scores.mode()[0],
        'std': scores.std(),
        'skewness': skew(scores),
        'kurtosis': kurtosis(scores)
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_title("JAMB Score Distribution", fontsize=14, fontweight='bold')
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    
    # Add statistics box
    stats_text = "\n".join([
        f"Mean: {stats['mean']:.2f}",
        f"Median: {stats['median']:.0f}",
        f"Mode: {stats['mode']:.0f}",
        f"Std Dev: {stats['std']:.2f}",
        f"Skewness: {stats['skewness']:.2f}",
        f"Kurtosis: {stats['kurtosis']:.2f}"
    ])
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("plots/histogram.png", dpi=300, bbox_inches='tight')
    plt.close()




# ============================================================================
# 2. SCATTER PLOT - Study Hours vs Performance
# ============================================================================

def plot_scatter(df):
    """
    Shows relationship between study hours and JAMB score with 
    regression trend line and key statistics.
    """
    # Calculate statistics
    scores = df["JAMB_Score"]
    study_hours = df["Study_Hours_Per_Week"]
    
    stats = {
        'score_mean': scores.mean(),
        'score_median': scores.median(),
        'study_mean': study_hours.mean(),
        'attendance_mean': df["Attendance_Rate"].mean()
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(scores, study_hours, alpha=0.6, s=100, 
                        c=df["Attendance_Rate"], cmap='viridis', edgecolors='black')
    
    # Add trend line
    z = np.polyfit(scores, study_hours, 1)
    p = np.poly1d(z)
    ax.plot(scores, p(scores), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    ax.set_title("Study Hours vs JAMB Score", fontsize=14, fontweight='bold')
    ax.set_xlabel("JAMB Score", fontsize=12)
    ax.set_ylabel("Study Hours Per Week", fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Attendance Rate (%)', fontsize=10)
    
    # Add statistics box
    stats_text = "\n".join([
        f"Avg Score: {stats['score_mean']:.2f}",
        f"Median Score: {stats['score_median']:.0f}",
        f"Avg Study Hours: {stats['study_mean']:.2f}",
        f"Avg Attendance: {stats['attendance_mean']:.1f}%"
    ])
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.8))
    
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/scatter.png", dpi=300, bbox_inches='tight')
    plt.close()




# ============================================================================
# 3. BOX PLOT - Statistical Summary by Variable
# ============================================================================

def plot_boxplot(df):
    """
    Displays distribution of all numeric variables using box plots.
    Shows median, quartiles, and outliers for each variable.
    """
    numeric_df = df.select_dtypes(include=["number"])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=numeric_df, orient="h", palette="Set2", ax=ax)
    
    ax.set_title("Statistical Summary - All Numeric Variables", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Value", fontsize=12)
    ax.set_ylabel("Variables", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("plots/boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# 4. CORRELATION HEATMAP - Factor Relationships
# ============================================================================

def plot_correlation_heatmap(df):
    """
    Shows correlation matrix of all numeric features.
    Identifies strong relationships affecting JAMB scores.
    """
    numeric_df = df.select_dtypes(include=["number"])
    correlation = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0, square=True, linewidths=1, ax=ax, cbar_kws={"label": "Correlation"})
    
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================ 
# 5. SCORE BY CATEGORICAL FACTORS - FIXED FOR SEABORN v0.14+
# ============================================================================

def plot_scores_by_school_type(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df, 
        x="School_Type", 
        y="JAMB_Score", 
        hue="School_Type",
        palette="muted", 
        ax=ax,
        legend=False
    )
    sns.swarmplot(
        data=df, 
        x="School_Type", 
        y="JAMB_Score", 
        color='black', 
        alpha=0.3,  # Reduced from 0.5
        size=2,     # Reduced from 4
        ax=ax
    )

    ax.set_title("JAMB Score Distribution by School Type", fontsize=14, fontweight='bold')
    ax.set_xlabel("School Type", fontsize=12)
    ax.set_ylabel("JAMB Score", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("plots/scores_by_school_type.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_scores_by_location_gender(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df, 
        x="School_Location", 
        y="JAMB_Score", 
        hue="Gender", 
        palette="husl",
        ax=ax
    )

    ax.set_title("JAMB Scores by School Location and Gender", fontsize=14, fontweight='bold')
    ax.set_xlabel("School Location", fontsize=12)
    ax.set_ylabel("JAMB Score", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("plots/scores_by_location_gender.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_scores_by_socioeconomic(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df, 
        x="Socioeconomic_Status", 
        y="JAMB_Score", 
        hue="Socioeconomic_Status",
        palette="viridis", 
        ax=ax, 
        errorbar='sd',
        legend=False
    )

    ax.set_title("Average JAMB Score by Socioeconomic Status", fontsize=14, fontweight='bold')
    ax.set_xlabel("Socioeconomic Status", fontsize=12)
    ax.set_ylabel("Average JAMB Score", fontsize=12)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f')

    plt.tight_layout()
    plt.savefig("plots/scores_by_socioeconomic.png", dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# 6. FEATURE IMPORTANCE - Regression Analysis
# ============================================================================

def plot_attendance_vs_score(df):
    """
    Analyzes relationship between attendance rate and exam performance.
    Includes scatter plot with trend line.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(df["Attendance_Rate"], df["JAMB_Score"], 
                        alpha=0.6, s=100, c=df["Age"], cmap='plasma', 
                        edgecolors='black')
    
    # Add trend line
    z = np.polyfit(df["Attendance_Rate"], df["JAMB_Score"], 1)
    p = np.poly1d(z)
    ax.plot(df["Attendance_Rate"], p(df["Attendance_Rate"]), 
            "r--", alpha=0.8, linewidth=2, label='Trend Line')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Age', fontsize=10)
    
    ax.set_title("Attendance Rate vs JAMB Score", fontsize=14, fontweight='bold')
    ax.set_xlabel("Attendance Rate (%)", fontsize=12)
    ax.set_ylabel("JAMB Score", fontsize=12)
    ax.legend()
    
    # Add correlation
    correlation = df["Attendance_Rate"].corr(df["JAMB_Score"])
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("plots/attendance_vs_score.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_teacher_quality_vs_score(df):
    """
    Analyzes teacher quality impact on student performance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Teacher_Quality", y="JAMB_Score", 
                palette="coolwarm", ax=ax)
    sns.stripplot(data=df, x="Teacher_Quality", y="JAMB_Score", 
                  color='black', alpha=0.2, size=3, ax=ax, jitter=True)
    
    ax.set_title("JAMB Score by Teacher Quality", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Teacher Quality Rating", fontsize=12)
    ax.set_ylabel("JAMB Score", fontsize=12)
    
    plt.tight_layout()
    plt.savefig("plots/teacher_quality_vs_score.png", dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# 7. SUPPORT FACTORS ANALYSIS
# ============================================================================

def plot_tutorials_and_materials_impact(df):
    """
    Compares average scores for students with/without tutorials and materials.
    Shows combined impact of support factors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Extra Tutorials Impact
    tutorial_data = df.groupby("Extra_Tutorials")["JAMB_Score"].agg(['mean', 'std', 'count'])
    axes[0].bar(tutorial_data.index, tutorial_data['mean'], 
               yerr=tutorial_data['std'], capsize=5, 
               color=['#FF6B6B', '#4ECDC4'], alpha=0.7, edgecolor='black')
    axes[0].set_title("Impact of Extra Tutorials", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Average JAMB Score", fontsize=11)
    axes[0].set_xlabel("Extra Tutorials", fontsize=11)
    for i, v in enumerate(tutorial_data['mean']):
        axes[0].text(i, v + tutorial_data['std'].iloc[i] + 5, f'{v:.1f}', 
                    ha='center', fontweight='bold')
    
    # Learning Materials Impact
    materials_data = df.groupby("Access_To_Learning_Materials")["JAMB_Score"].agg(['mean', 'std', 'count'])
    axes[1].bar(materials_data.index, materials_data['mean'], 
               yerr=materials_data['std'], capsize=5, 
               color=['#FFE66D', '#95E1D3'], alpha=0.7, edgecolor='black')
    axes[1].set_title("Impact of Learning Materials", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Average JAMB Score", fontsize=11)
    axes[1].set_xlabel("Access to Materials", fontsize=11)
    for i, v in enumerate(materials_data['mean']):
        axes[1].text(i, v + materials_data['std'].iloc[i] + 5, f'{v:.1f}', 
                    ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("plots/tutorials_and_materials_impact.png", dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# 8. PARENT INVOLVEMENT ANALYSIS
# ============================================================================

def plot_parent_involvement_analysis(df):
    """
    Shows comprehensive analysis of parent education and involvement levels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Parent Involvement
    involvement_data = df.groupby("Parent_Involvement")["JAMB_Score"].agg(['mean', 'std', 'count'])
    involvement_data = involvement_data.sort_values('mean', ascending=False)
    axes[0].barh(involvement_data.index, involvement_data['mean'], 
                xerr=involvement_data['std'], capsize=5,
                color=['#2ECC71', '#F39C12', '#E74C3C'], alpha=0.7, edgecolor='black')
    axes[0].set_title("JAMB Score by Parent Involvement", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Average JAMB Score", fontsize=11)
    for i, (idx, row) in enumerate(involvement_data.iterrows()):
        axes[0].text(row['mean'] + 5, i, f"{row['mean']:.1f}", va='center', fontweight='bold')
    
    # Parent Education Level
    education_data = df.groupby("Parent_Education_Level")["JAMB_Score"].agg(['mean', 'std', 'count'])
    education_data = education_data.sort_values('mean', ascending=False)
    colors_edu = ['#9B59B6', '#3498DB', '#E67E22', '#95A5A6']
    axes[1].barh(education_data.index, education_data['mean'], 
                xerr=education_data['std'], capsize=5,
                color=colors_edu[:len(education_data)], alpha=0.7, edgecolor='black')
    axes[1].set_title("JAMB Score by Parent Education", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Average JAMB Score", fontsize=11)
    for i, (idx, row) in enumerate(education_data.iterrows()):
        axes[1].text(row['mean'] + 5, i, f"{row['mean']:.1f}", va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("plots/parent_involvement_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Generating JAMB Score Analytics visualizations...")
    
    plot_histogram(dataframe)
    print("✓ Histogram generated")
    
    plot_scatter(dataframe)
    print("✓ Scatter plot generated")
    
    plot_boxplot(dataframe)
    print("✓ Box plot generated")
    
    plot_correlation_heatmap(dataframe)
    print("✓ Correlation heatmap generated")
    
    plot_scores_by_school_type(dataframe)
    print("✓ School type analysis generated")
    
    plot_scores_by_location_gender(dataframe)
    print("✓ Location & gender analysis generated")
    
    plot_scores_by_socioeconomic(dataframe)
    print("✓ Socioeconomic analysis generated")
    
    plot_attendance_vs_score(dataframe)
    print("✓ Attendance analysis generated")
    
    plot_teacher_quality_vs_score(dataframe)
    print("✓ Teacher quality analysis generated")
    
    plot_tutorials_and_materials_impact(dataframe)
    print("✓ Tutorial & materials impact generated")
    
    plot_parent_involvement_analysis(dataframe)
    print("✓ Parent involvement analysis generated")
    
    print("\nAll visualizations completed! Check the 'plots/' folder.")
