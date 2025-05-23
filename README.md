# 2D TIF Volume Masking with nnUNetv2

This repository contains a pipeline for masking 2D TIF slices of a scroll using a trained nnUNetv2 model.

## Requirements

- Python 3.12
- nnUNetv2
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:
```shell
git clone git@github.com:mvrcii/mask_2d_tif_volume.git
cd mask_2d_tif_volume
```

2. Setup the environment:
```shell
python setup.py
```

## Getting Started

3. Download the raw tif volume:
```shell
python download_volume.py --data-dir /home/marcel/scrollprize/data --source https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/volumes/20231117161658/ --target /home/marcel/scrollprize/data/scroll4.volpkg/volumes/20231117161658
```
   
4. Mask the tif volume:
```bash
python fast_masking.py --input-dir ~/scrollprize/data/scroll4.volpkg/volumes/20231117161658
```

5. Masked tif volume to uint8 ome-zarr:
```shell
python python scroll_to_zarr.py ~/scrollprize/data/scroll4.volpkg/volumes/20231117161658_masked ~/scrollprize/data/scroll4.volpkg/volumes/20231117161658_masked_uint8.zarr --obytes 1
```