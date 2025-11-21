#!/bin/bash
# Setup script for JAMB Score Analytics

echo "=========================================="
echo "JAMB Score Analytics - Setup Script"
echo "=========================================="

echo ""
echo "[1/3] Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "[2/3] Training ML models..."
python train_models.py

echo ""
echo "[3/3] Setup complete!"
echo ""
echo "To run the Streamlit app, execute:"
echo "  streamlit run app.py"
echo ""
