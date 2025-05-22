#!/bin/bash

# Create directory structure
mkdir -p {src,models,scripts,config}
mkdir -p temp/nnUNet_raw
mkdir -p temp/nnUNet_preprocessed
mkdir -p temp/nnUNet_results

# Download the pre-trained nnUNet model
echo "Downloading pre-trained nnUNet model..."
wget -r -np -nH --cut-dirs=5 --reject "index.html?*" -P ./models https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_results/Dataset082_scrollmask2/nnUNetTrainerWorkshop__nnUNetPlans__2d/

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete! Now update the paths in config/config.py"
