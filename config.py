import os

# Default paths - can be overridden via command line arguments
INPUT_DIR = os.path.expanduser("~/scrollprize/data/scroll4.volpkg/volumes/20231117161658")
OUTPUT_DIR = os.path.expanduser("~/scrollprize/data/scroll4.volpkg/volumes_masked/20231117161658")
MODEL_DIR = "./models/nnUNetTrainerV2__nnUNetPlans__2d"

# Processing parameters
DOWNSCALE_FACTOR = 4
THRESHOLD = 0.5
N_WORKERS = None  # Auto-detect

# Temporary directory for processing
TEMP_DIR = "./tmp"

# Rich console configuration
CONSOLE_WIDTH = None  # Auto-detect terminal width
