# 2D TIF Volume Masking with nnUNetv2

This repository contains a pipeline for masking 2D TIF slices of a scroll using a trained nnUNetv2 model.

## Overview

The pipeline consists of three main steps:
1. **Pre-processing**: Downscale original TIF images by a factor of 4 and convert to NIfTI format
2. **Inference**: Run the trained nnUNetv2 model to create masks
3. **Post-processing**: Upscale the predicted masks and apply them to the original TIF images

## Requirements

- Python 3.12
- nnUNetv2
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:mvrcii/mask_2d_tif_volume.git
   cd mask_2d_tif_volume
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained nnUNet model:
   ```bash
   # Create models directory
   mkdir -p ./models
   
   # Download the model
   wget -r -np -nH --cut-dirs=5 --reject "index.html?*" -P ./models https://dl.ash2txt.org/community-uploads/bruniss/nnunet_models/nnUNet_results/Dataset082_scrollmask2/nnUNetTrainerWorkshop__nnUNetPlans__2d/
   ```

4. Update the configuration in `config/config.py` with your paths.

## Downloading TIF Files
### Using Python script
```bash
# Download TIF files
python scripts/download_tifs.py https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes/20231117161658/ ~/scrollprize/data/scroll4.volpkg/volumes/20231117161658/
```

## Usage

### Running the complete pipeline

```bash
python scripts/run_pipeline.py --input /path/to/tif/files --output /path/to/output --model ./models/nnUNetTrainerWorkshop__nnUNetPlans__2d
```

### Run individual steps

1. Preprocessing only:
   ```bash
   python src/preprocess.py
   ```

2. Inference only (requires preprocessing first):
   ```bash
   python src/inference.py
   ```

3. Postprocessing only (requires inference first):
   ```bash
   python src/postprocess.py
   ```

## Configuration

Edit `config/config.py` to set paths and parameters:

- `INPUT_DIR`: Directory containing the TIF files
- `OUTPUT_DIR`: Directory to save the masked TIF files
- `MODEL_DIR`: Path to the downloaded model (default: ./models/nnUNetTrainerWorkshop__nnUNetPlans__2d)
- `DOWNSCALE_FACTOR`: Factor to downscale the original images (default: 4)
- `THRESHOLD`: Threshold for binary segmentation (default: 0.5)

## License

[MIT License](LICENSE)
