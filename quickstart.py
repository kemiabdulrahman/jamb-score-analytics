"""
JAMB Score Analytics - Quick Start Guide
Get everything running in 3 steps!
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command with feedback"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     JAMB SCORE ANALYTICS - QUICK START SETUP             ║
    ║     Full ML Pipeline with Streamlit Dashboard            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Install dependencies
    if not run_command(
        "pip install -r requirements.txt",
        "[STEP 1/3] Installing dependencies..."
    ):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    print("✅ Dependencies installed successfully")
    
    # Step 2: Train models
    if not run_command(
        "python train_models.py",
        "[STEP 2/3] Training ML models (this may take 2-3 minutes)..."
    ):
        print("❌ Failed to train models")
        sys.exit(1)
    
    print("✅ All models trained and saved")
    
    # Step 3: Run Streamlit app
    print(f"\n{'='*60}")
    print("[STEP 3/3] Launching Streamlit Dashboard...")
    print(f"{'='*60}")
    print("""
    🚀 The dashboard will open at: http://localhost:8501
    
    📊 Features Available:
    ├── 🏠 Home - Overview & statistics
    ├── 🎯 Score Predictor - Predict JAMB scores
    ├── 📈 Performance Classifier - Classify performance tiers
    ├── 🔍 Student Segmentation - K-Means clustering analysis
    ├── 🧬 Feature Analysis - Factor importance
    └── 📊 Model Comparison - Performance metrics
    
    💡 Tip: Close the terminal to stop the server (Ctrl+C)
    """)
    
    run_command("streamlit run app.py", "Starting dashboard...")

if __name__ == "__main__":
    main()
