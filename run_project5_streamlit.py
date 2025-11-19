#!/usr/bin/env python3
"""
Runner script for Project 5 ML from Scratch Streamlit App
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    print("🚀 Starting ML from Scratch Streamlit App...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Check if required packages are installed
    required_packages = ['numpy', 'pandas', 'plotly']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Run the Streamlit app
    print("\n🎬 Launching Streamlit app...")
    print("📱 The app will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⚡ Features available:")
    print("  • 📈 Linear Regression with regularization")
    print("  • 🎯 Logistic Regression with decision boundaries")
    print("  • 🧠 Neural Networks with custom architectures")
    print("  • 🎪 K-Means Clustering visualization")
    print("  • 📊 PCA dimensionality reduction")
    print("  • 🔬 Model comparison tools")
    print("  • 📋 Algorithm theory and formulas")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "project5_ml_from_scratch_streamlit.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit app: {e}")

if __name__ == "__main__":
    main()