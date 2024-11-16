#!/bin/bash

# Update the system and install required tools
echo "Updating system and installing necessary tools..."
apt-get update && apt-get install -y python3-distutils

# Upgrade pip, setuptools, and wheel
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Pre-install numpy and pandas with binary wheels to avoid build issues
echo "Installing numpy and pandas..."
pip install --only-binary=:all: numpy>=1.24.0,<1.25.0 pandas==2.0.3

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Download and install Spacy model if not prepackaged
echo "Downloading Spacy model..."
python -m spacy download en_core_web_sm

echo "Setup complete!"
