import argparse
import os
import sys

# Add the repository root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import prepare_data_for_inference
from src.inference import run_inference
from src.postprocess import apply_masks
import config.config as cfg


def main():
    parser = argparse.ArgumentParser(description='Run the 2D TIF masking pipeline')
    parser.add_argument('--input', type=str, default=cfg.INPUT_DIR,
                        help=f'Input directory with TIF files (default: {cfg.INPUT_DIR})')
    parser.add_argument('--output', type=str, default=cfg.OUTPUT_DIR,
                        help=f'Output directory for masked TIF files (default: {cfg.OUTPUT_DIR})')
    parser.add_argument('--model', type=str, default=cfg.MODEL_DIR,
                        help=f'Path to the nnUNet model directory (default: {cfg.MODEL_DIR})')
    parser.add_argument('--downscale', type=int, default=cfg.DOWNSCALE_FACTOR,
                        help=f'Downscale factor (default: {cfg.DOWNSCALE_FACTOR})')
    parser.add_argument('--threshold', type=float, default=cfg.THRESHOLD,
                        help=f'Mask threshold (default: {cfg.THRESHOLD})')
    parser.add_argument('--workers', type=int, default=cfg.N_WORKERS,
                        help=f'Number of parallel workers for preprocessing (default: auto-detect)')

    args = parser.parse_args()

    # Update config with command line arguments (or use defaults)
    cfg.INPUT_DIR = args.input
    cfg.OUTPUT_DIR = args.output
    cfg.MODEL_DIR = args.model
    cfg.DOWNSCALE_FACTOR = args.downscale
    cfg.THRESHOLD = args.threshold
    cfg.N_WORKERS = args.workers

    # Create output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print("Starting 2D TIF masking pipeline...")
    print(f"Input directory: {cfg.INPUT_DIR}")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    print(f"Model directory: {cfg.MODEL_DIR}")
    print(f"Downscale factor: {cfg.DOWNSCALE_FACTOR}")
    print(f"Mask threshold: {cfg.THRESHOLD}")

    # Verify input directory exists and has TIF files
    if not os.path.exists(cfg.INPUT_DIR):
        print(f"Error: Input directory {cfg.INPUT_DIR} does not exist!")
        return 1

    import glob
    tif_files = glob.glob(os.path.join(cfg.INPUT_DIR, "*.tif"))
    if not tif_files:
        print(f"Error: No TIF files found in {cfg.INPUT_DIR}")
        return 1

    print(f"Found {len(tif_files)} TIF files to process")

    # Verify model directory exists
    if not os.path.exists(cfg.MODEL_DIR):
        print(f"Error: Model directory {cfg.MODEL_DIR} does not exist!")
        print("Run the setup script or download the model manually.")
        return 1

    # Run the pipeline
    try:
        print("\n=== Step 1: Preprocessing ===")
        tif_files, input_dir = prepare_data_for_inference(n_workers=cfg.N_WORKERS)

        print("\n=== Step 2: Inference ===")
        predictions_dir = run_inference(input_dir)

        print("\n=== Step 3: Postprocessing ===")
        apply_masks(tif_files, predictions_dir)

        print("\n=== Pipeline Complete! ===")
        print(f"Masked images saved to: {cfg.OUTPUT_DIR}")

        # Show final statistics
        import glob
        masked_files = glob.glob(os.path.join(cfg.OUTPUT_DIR, "*.tif"))
        print(f"Total masked files created: {len(masked_files)}")

        return 0

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
