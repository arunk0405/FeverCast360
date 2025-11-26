#!/bin/bash

# FeverCast360 Quick Start Script
# This script sets up and runs the FeverCast360 application

echo "🌡️  FeverCast360 Quick Start"
echo "================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/requirements.installed" ]; then
    echo "📥 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/requirements.installed
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Check if Firebase credentials exist
if [ ! -f "newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json" ]; then
    echo ""
    echo "⚠️  WARNING: Firebase credentials not found!"
    echo "   Please place your Firebase service account key in the project root"
    echo "   File name: newp-9a65c-firebase-adminsdk-fbsvc-c607f614f3.json"
    echo ""
    echo "   See FIREBASE_SETUP.md for detailed instructions"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Create necessary directories
echo "📁 Checking project structure..."
mkdir -p models outputs/plots
echo "✅ Project structure ready"

echo ""
echo "🚀 Starting FeverCast360..."
echo "================================"
echo ""
echo "📍 Dashboard will open at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Run Streamlit
streamlit run app.py
