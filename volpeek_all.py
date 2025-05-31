import os
from pathlib import Path

from core.visualization.volpeek import plot_overview


def find_volumes_in_dir(parent_dir):
    """Find all volume directories (TIF stack dir, .zarr, .ome.zarr) under parent_dir."""
    parent_dir = os.path.expanduser(parent_dir)
    parent_path = Path(parent_dir)
    volumes = []
    # Find .ome.zarr and .zarr
    volumes += [str(p) for p in parent_path.glob("*.ome.zarr")]
    volumes += [str(p) for p in parent_path.glob("*.zarr") if not p.name.endswith(".ome.zarr")]
    # Find TIF-stack directories (containing tif files, but not .zarr/.ome.zarr)
    for p in parent_path.iterdir():
        if p.is_dir() and not (p.name.endswith('.zarr') or p.name.endswith('.ome.zarr')):
            tif_files = list(p.glob("*.tif"))
            if tif_files:
                volumes.append(str(p))
    return sorted(volumes)


def plot_all_volumes_in_dir(parent_dir, output_dir, **plot_kwargs):
    vols = find_volumes_in_dir(parent_dir)
    output_dir = os.path.expanduser(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if not vols:
        print(f"No volumes found in {parent_dir}")
        return
    print(f"Found {len(vols)} volumes:")
    for v in vols:
        print("  -", v)
    for v in vols:
        # Use basename, strip known suffixes, add .png
        base = os.path.basename(os.path.normpath(v))
        for suffix in [".ome.zarr", ".zarr"]:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        out_png = os.path.join(output_dir, f"{base}.png")
        print(f"\n[INFO] Plotting overview for: {v}")
        try:
            plot_overview(v, output_path=out_png, **plot_kwargs)
        except Exception as e:
            print(f"[WARN] Failed to plot {v}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create overview plots for all volumes in a directory")
    parser.add_argument("dir", help="Directory containing multiple volumes (.zarr, .ome.zarr, or TIF stack dirs)")
    parser.add_argument("--output", required=True, help="Output directory for overview PNGs")
    parser.add_argument("--step", type=int, default=1000, help="Slice step (default: 1000)")
    parser.add_argument("--downscale", type=int, default=1, help="Downscale for plotting (default: 1 = no downscale)")
    args = parser.parse_args()
    plot_all_volumes_in_dir(args.dir, args.output, step=args.step, downscale=args.downscale)
