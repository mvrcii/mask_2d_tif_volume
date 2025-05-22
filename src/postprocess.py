import glob
import os
import sys

import nibabel as nib
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import *


def apply_masks(tif_files, predictions_dir):
    """Apply predicted masks to original images"""

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect prediction files
    prediction_files = sorted(glob.glob(os.path.join(predictions_dir, "*.nii.gz")))

    # Check if the number of prediction files matches the number of TIF files
    if len(prediction_files) != len(tif_files):
        raise ValueError(
            f"Number of prediction files ({len(prediction_files)}) does not match number of TIF files ({len(tif_files)})")

    print("Applying masks to original images...")
    for i, (orig_path, pred_path) in enumerate(tqdm(zip(tif_files, prediction_files))):
        # Load original image
        original = imread(orig_path)

        # Load prediction (mask)
        pred_nii = nib.load(pred_path)
        prediction = pred_nii.get_fdata()
        prediction = np.squeeze(prediction)

        # Upscale prediction to match original size
        upscaled_mask = resize(prediction, original.shape,
                               order=0, preserve_range=True, anti_aliasing=False)

        # Convert to binary mask
        upscaled_mask = (upscaled_mask > THRESHOLD).astype(np.uint8)

        # Apply mask to original image
        masked_image = original * upscaled_mask

        # Save the result
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(orig_path))
        imsave(output_path, masked_image.astype(original.dtype))

    print(f"Masking complete! Saved {len(tif_files)} masked images to {OUTPUT_DIR}")


if __name__ == "__main__":
    from preprocess import prepare_data_for_inference
    from inference import run_inference

    tif_files, input_dir = prepare_data_for_inference()
    predictions_dir = run_inference(input_dir)
    apply_masks(tif_files, predictions_dir)
