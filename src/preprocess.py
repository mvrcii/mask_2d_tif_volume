import glob
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm

# Add the repository root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config.config as cfg


def process_tif_group(args):
    """Process a group of 4 TIFs: Z-mean then XY-downscale, preserving dtype."""
    group_idx, tif_group, output_path, downscale_factor = args

    # 1) Load exactly four slices
    images = [imread(p) for p in tif_group]
    orig_dtype = images[0].dtype

    # 2) Z-mean then cast back
    stacked = np.stack(images, axis=0)  # (4, H, W)
    z_reduced = np.mean(stacked, axis=0).astype(orig_dtype)  # (H, W)

    # 3) XY-downscale then cast back
    new_shape = (
        z_reduced.shape[0] // downscale_factor,
        z_reduced.shape[1] // downscale_factor
    )
    down_xy = resize(
        z_reduced,
        new_shape,
        preserve_range=True,
        anti_aliasing=True
    ).astype(orig_dtype)

    # 4) Save into nnUNet raw/imagesTs
    imsave(output_path, down_xy, check_contrast=False)

    return True, group_idx, len(tif_group), z_reduced.shape, down_xy.shape


def inspect_preprocessing_results(original_dir, processed_dir, num_samples=3):
    """Inspect original vs processed TIFFs."""
    print("=== Preprocessing Inspection ===")
    original_files = sorted(glob.glob(os.path.join(original_dir, "*.tif")))
    processed_files = sorted(glob.glob(os.path.join(processed_dir, "*.tif")))

    print(f"Original files:  {len(original_files)}")
    print(f"Processed files: {len(processed_files)}")
    print(f"Z-axis reduction ratio: {len(original_files) / len(processed_files):.1f}:1")

    for i in range(min(num_samples, len(processed_files))):
        print(f"\n--- Sample {i + 1} ---")
        proc = imread(processed_files[i])
        print(f"Processed: {os.path.basename(processed_files[i])}")
        print(f"  Shape: {proc.shape}, Dtype: {proc.dtype}, Range: {proc.min()}–{proc.max()}")
        # corresponding originals
        start, end = i * 4, min(i * 4 + 4, len(original_files))
        for orig_path in original_files[start:end]:
            orig = imread(orig_path)
            print(f"  {os.path.basename(orig_path)} → {orig.shape}")
    print("=== End Inspection ===")


def prepare_downscaled_tifs_3d(n_workers=None, inspect=True):
    """3D downsample: group of 4 → mean → XY downscale."""
    # write directly into nnUNet raw/imagesTs
    imagesTs_dir = os.path.join(os.environ["nnUNet_raw"], cfg.DATASET_NAME, "imagesTs")
    os.makedirs(imagesTs_dir, exist_ok=True)

    tif_files = sorted(glob.glob(os.path.join(cfg.INPUT_DIR, "*.tif")))
    print(f"Found {len(tif_files)} input TIFFs")
    if not tif_files:
        raise ValueError(f"No TIFFs found in {cfg.INPUT_DIR}")

    # group into fours
    tif_groups = [tif_files[i:i + 4] for i in range(0, len(tif_files), 4)]
    print(f"{len(tif_groups)} groups of 4 → {len(tif_groups)} output files")

    n_workers = n_workers or multiprocessing.cpu_count()
    print(f"Using {n_workers} workers; XY downscale ×{cfg.DOWNSCALE_FACTOR}")

    args_list = []
    for idx, group in enumerate(tif_groups):
        out_name = f"scroll_{idx:05d}_0000.tif"
        out_path = os.path.join(imagesTs_dir, out_name)
        args_list.append((idx, group, out_path, cfg.DOWNSCALE_FACTOR))

    succ = fail = 0
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for ok, *_ in tqdm(exe.map(process_tif_group, args_list),
                           total=len(args_list),
                           desc="3D downsampling"):
            if ok:
                succ += 1
            else:
                fail += 1

    print(f"Done: {succ} succeeded, {fail} failed")
    if inspect and succ:
        inspect_preprocessing_results(cfg.INPUT_DIR, imagesTs_dir)

    return imagesTs_dir


def process_single_2d(args):
    """2D-only: XY-downscale one slice, preserving dtype."""
    i, path, output_path, factor = args
    img = imread(path)
    dtype = img.dtype

    down = resize(
        img,
        (img.shape[0] // factor, img.shape[1] // factor),
        preserve_range=True,
        anti_aliasing=True
    ).astype(dtype)

    imsave(output_path, down, check_contrast=False)
    return True, i, None


def prepare_downscaled_tifs_2d_only(n_workers=None):
    """2D-only downsampling for comparison."""
    imagesTs_dir = os.path.join(os.environ["nnUNet_raw"], cfg.DATASET_NAME, "imagesTs_2d_only")
    os.makedirs(imagesTs_dir, exist_ok=True)

    tif_files = sorted(glob.glob(os.path.join(cfg.INPUT_DIR, "*.tif")))
    print(f"Found {len(tif_files)} input TIFFs")
    if not tif_files:
        raise ValueError(f"No TIFFs found in {cfg.INPUT_DIR}")

    n_workers = n_workers or multiprocessing.cpu_count()
    print(f"Using {n_workers} workers for 2D-only (×{cfg.DOWNSCALE_FACTOR})")

    args_list = [
        (i,
         fp,
         os.path.join(imagesTs_dir, f"2d_{i:05d}.tif"),
         cfg.DOWNSCALE_FACTOR)
        for i, fp in enumerate(tif_files)
    ]

    succ = fail = 0
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for ok, *_ in tqdm(exe.map(process_single_2d, args_list),
                           total=len(args_list),
                           desc="2D-only downsampling"):
            if ok:
                succ += 1
            else:
                fail += 1

    print(f"2D-only done: {succ} succeeded, {fail} failed")
    return imagesTs_dir


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=None,
                   help='parallel worker count')
    p.add_argument('--mode', choices=['3d', '2d', 'both'], default='3d')
    p.add_argument('--no-inspect', action='store_true')
    args = p.parse_args()

    if args.mode == '3d':
        out = prepare_downscaled_tifs_3d(n_workers=args.workers,
                                         inspect=not args.no_inspect)
        print("3D downscaled TIFFs in:", out)
    elif args.mode == '2d':
        out = prepare_downscaled_tifs_2d_only(n_workers=args.workers)
        print("2D-only downscaled TIFFs in:", out)
    else:
        print("=== 3D pass ===")
        d3 = prepare_downscaled_tifs_3d(n_workers=args.workers,
                                        inspect=not args.no_inspect)
        print("=== 2D pass ===")
        d2 = prepare_downscaled_tifs_2d_only(n_workers=args.workers)
        print("Outputs:", d3, d2)

    print("All downscaled TIFFs ready for inference.")
