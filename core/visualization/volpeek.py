import math
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from core.formats.volume_factory import create_volume_reader


def plot_overview(input_dir, output_path=None, step=1000, cmap='gray', downscale=1):
    import importlib

    input_dir = os.path.expanduser(input_dir)
    reader = create_volume_reader(input_dir)

    # -- OME-Zarr lower-res logic --
    ome_level_used = 0
    if reader.__class__.__name__ == "OMEZarrVolumeReader":
        # Try to use the coarsest (highest) available level
        ome_zarr_path = getattr(reader, 'ome_zarr_path', None)
        if ome_zarr_path and hasattr(os, "listdir"):
            try:
                levels = sorted(
                    [int(d) for d in os.listdir(ome_zarr_path) if d.isdigit()]
                )
                if len(levels) > 1:
                    ome_level_used = levels[-1]  # coarsest
                    # Re-open the reader at this coarser level
                    from core.formats.zarr_handler import OMEZarrVolumeReader
                    reader.close()
                    reader = OMEZarrVolumeReader(ome_zarr_path, level=ome_level_used)
                    print(f"[INFO] Using OME-Zarr level {ome_level_used} for fast plotting.")
            except Exception as e:
                print(f"[WARN] Could not switch OME-Zarr level: {e}")

    shape = reader.get_shape()  # (z, y, x)
    n_slices = shape[0]

    indices = list(range(0, n_slices, step))
    n_imgs = len(indices)
    nrows = 2
    ncols = math.ceil(n_imgs / nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), dpi=50, squeeze=False)
    fig.patch.set_facecolor('#282828')
    for ax in axes.flat:
        ax.axis('off')

    print(f"Plotting {n_imgs} slices from volume (z={n_slices}), stride={step}, grid: {nrows}x{ncols}")

    for i, z in enumerate(tqdm(indices, desc="Reading slices")):
        ax = axes[i // ncols, i % ncols]
        img = reader.read_slice(z)
        # Only downscale if requested, or if not using multiscale OME-Zarr
        if downscale > 1 and (img.shape[0] > 512 or img.shape[1] > 512):
            import cv2
            img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale), interpolation=cv2.INTER_AREA)
        img_disp = img.astype(np.float32)
        img_disp -= img_disp.min()
        if img_disp.max() > 0:
            img_disp /= img_disp.max()
        ax.imshow(img_disp, cmap=cmap)
        ax.text(0.97, 0.03, f"{z:05d}", fontsize=16, color='white',
                ha='right', va='bottom', transform=ax.transAxes)

    plt.tight_layout(pad=2.5, h_pad=2, w_pad=2)
    if output_path is None:
        # Default to input_dir basename (strip .zarr/.ome.zarr) + .png
        base = os.path.basename(os.path.normpath(input_dir))
        for suffix in [".ome.zarr", ".zarr"]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        output_path = f"{base}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Saved overview to {output_path}")

    reader.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot slice overview of a volume (tif-stack, zarr, or ome-zarr)")
    parser.add_argument("input_dir", help="Input directory (tif-stack, zarr, ome-zarr)")
    parser.add_argument("--output", default=None, help="Output image path (if not given, display interactively)")
    parser.add_argument("--step", type=int, default=1000, help="Step size (default: 1000)")
    parser.add_argument("--downscale", type=int, default=3,
                        help="Downscale factor for fast plotting (default: 3)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI (default: 300)")
    args = parser.parse_args()
    plot_overview(args.input_dir, args.output, step=args.step, downscale=args.downscale)
