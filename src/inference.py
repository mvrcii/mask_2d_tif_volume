import json
import os
import shutil
import sys

import nibabel as nib
import numpy as np
import torch

# Add the repository root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config.config as cfg


def check_model_expectations():
    """Check what the model expects as input format"""
    dataset_json = os.path.join(cfg.MODEL_DIR, "dataset.json")
    plans_json = os.path.join(cfg.MODEL_DIR, "plans.json")

    print("=== Model Expectations ===")
    if os.path.exists(dataset_json):
        with open(dataset_json) as f:
            dj = json.load(f)
        print("File ending:", dj.get("file_ending", "N/A"))
        print("Channels:", dj.get("channel_names", "N/A"))
        print("Modality:", dj.get("modality", "N/A"))
        print("Dataset name:", dj.get("name", "N/A"))
        if "image_reader_writer" in dj:
            print("Image reader:", dj["image_reader_writer"])
    if os.path.exists(plans_json):
        with open(plans_json) as f:
            pj = json.load(f)
        if "image_reader_writer" in pj:
            print("Plans reader:", pj["image_reader_writer"])
    print("===========================")


def postprocess_and_save(raw_results_dir, final_output_dir):
    """
    Post-process nnUNet .nii.gz outputs by thresholding
    and save final masks into final_output_dir.
    """
    os.makedirs(final_output_dir, exist_ok=True)
    files = [f for f in os.listdir(raw_results_dir) if f.endswith(".nii.gz")]
    print(f"Post-processing {len(files)} files with threshold {cfg.THRESHOLD}")
    for fn in files:
        raw_path = os.path.join(raw_results_dir, fn)
        nii = nib.load(raw_path)
        data = nii.get_fdata()
        # binary mask
        mask = (data >= cfg.THRESHOLD).astype(np.uint8)
        # save back to NIfTI (or change to .tif if desired)
        out_nii = nib.Nifti1Image(mask, affine=nii.affine)
        out_path = os.path.join(final_output_dir, fn)
        nib.save(out_nii, out_path)
    print(f"Final masks written to: {final_output_dir}")


def run_inference_on_nnunet_data():
    """Run nnUNet inference, stash raw output in tmp, postprocess and copy to final."""
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # input images
    imagesTs_dir = os.path.join(os.environ["nnUNet_raw"], cfg.DATASET_NAME, "imagesTs")
    if not os.path.isdir(imagesTs_dir):
        raise FileNotFoundError(f"{imagesTs_dir} not found – run preprocessing first")

    tifs = [f for f in os.listdir(imagesTs_dir) if f.lower().endswith(".tif")]
    if not tifs:
        raise ValueError(f"No TIFFs in {imagesTs_dir} – run preprocessing first")

    print(f"Found {len(tifs)} TIFFs, e.g.: {tifs[:3]}")

    # raw nnUNet results go here
    raw_results_dir = os.path.join(os.environ["nnUNet_results"], cfg.DATASET_NAME)
    os.makedirs(raw_results_dir, exist_ok=True)

    # check model
    if not os.path.isdir(cfg.MODEL_DIR):
        raise FileNotFoundError(f"Model folder not found: {cfg.MODEL_DIR}")
    check_model_expectations()
    fix_trainer_in_checkpoint()

    # init predictor
    print("Initializing nnUNet predictor…")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # load weights
    for ckpt in ("checkpoint_best.pth", "checkpoint_final.pth"):
        try:
            predictor.initialize_from_trained_model_folder(
                cfg.MODEL_DIR, use_folds=(0,), checkpoint_name=ckpt
            )
            print(f"Loaded {ckpt}")
            break
        except Exception:
            print(f"Failed to load {ckpt}")
    else:
        raise FileNotFoundError("No valid checkpoint found")

    # run inference
    print(f"Running inference → raw results at {raw_results_dir}")
    predictor.predict_from_files(
        imagesTs_dir,
        raw_results_dir,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0
    )
    print("Inference done.")

    # postprocess and copy out final masks
    final_output_dir = os.path.join(cfg.OUTPUT_DIR, "nnunet_predictions")
    postprocess_and_save(raw_results_dir, final_output_dir)
    return final_output_dir


def fix_trainer_in_checkpoint():
    """Ensure checkpoint uses standard nnUNetTrainer name"""
    checkpoint_names = ["checkpoint_best.pth", "checkpoint_final.pth"]
    ckpt_path = None
    # look in fold_0/
    for name in checkpoint_names:
        p = os.path.join(cfg.MODEL_DIR, "fold_0", name)
        if os.path.exists(p):
            ckpt_path = p
            break
    # fallback to model root
    if ckpt_path is None:
        for name in checkpoint_names:
            p = os.path.join(cfg.MODEL_DIR, name)
            if os.path.exists(p):
                ckpt_path = p
                break
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found in model folder")

    import torch
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    orig = checkpoint.get("trainer_name", "")
    if orig != "nnUNetTrainer":
        print(f"Fixing trainer_name '{orig}' → 'nnUNetTrainer'")
        backup = ckpt_path + ".backup"
        if not os.path.exists(backup):
            shutil.copy2(ckpt_path, backup)
            print("Backup created at", backup)
        checkpoint["trainer_name"] = "nnUNetTrainer"
        torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
        print("Checkpoint updated.")
    else:
        print("Trainer name OK:", orig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check model expectations, skip inference"
    )
    args = parser.parse_args()

    if args.check_only:
        check_model_expectations()
    else:
        out_dir = run_inference_on_nnunet_data()
        print("Final masks available in:", out_dir)
