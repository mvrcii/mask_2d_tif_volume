# config/config.py
import os

# Paths
INPUT_DIR = os.path.expanduser("~/scrollprize/data/scroll4.volpkg/volumes/dl.ash2txt.org/20231117161658")   # default volume tif files directory
OUTPUT_DIR = os.path.expanduser("~/scrollprize/data/scroll4.volpkg/volumes_masked/20231117161658") # default masked output directory path
TEMP_DIR = os.path.expanduser( "./tmp/nnunet_processing")

# nnUNet settings
MODEL_DIR = "./models/nnUNetTrainerV2__nnUNetPlans__2d"  # model directory path
DATASET_NAME = "Dataset082_scrollmask2"

# Processing parameters
DOWNSCALE_FACTOR = 4  # Scale down by 1/4 in each dimension
THRESHOLD = 0.5  # Threshold for binary segmentation
N_WORKERS = None # Number of parallel workers (None = auto-detect CPU count)

# nnUNet environment variables
os.environ["nnUNet_raw"] = os.path.join(TEMP_DIR, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(TEMP_DIR, "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(TEMP_DIR, "nnUNet_results")

# Create necessary directories
for d in [
    INPUT_DIR, OUTPUT_DIR, TEMP_DIR,
    os.environ["nnUNet_raw"],
    os.environ["nnUNet_preprocessed"],
    os.environ["nnUNet_results"],
]:
    os.makedirs(d, exist_ok=True)