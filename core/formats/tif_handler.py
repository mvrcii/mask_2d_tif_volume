from pathlib import Path
from typing import Union

import numpy as np
import tifffile

from .base import VolumeReader


class TifVolumeReader(VolumeReader):
    """Reader for TIF stack volumes (existing functionality)"""

    def __init__(self, tif_dir: Union[str, Path]):
        self.tif_dir = Path(tif_dir)
        self._scan_tif_files()

    def _scan_tif_files(self):
        """Scan directory for TIF files"""
        self.tif_files = sorted(list(self.tif_dir.glob("*.tif")))
        if not self.tif_files:
            raise ValueError(f"No TIF files found in {self.tif_dir}")

        # Read first file to get shape and dtype
        first_img = tifffile.imread(self.tif_files[0])
        self._height, self._width = first_img.shape
        self._dtype = first_img.dtype
        self._shape = (len(self.tif_files), self._height, self._width)

    def read_slice(self, z_index: int) -> np.ndarray:
        """Read a Z slice from TIF files"""
        if z_index >= len(self.tif_files):
            raise IndexError(f"Z index {z_index} out of range (max: {len(self.tif_files) - 1})")
        return tifffile.imread(self.tif_files[z_index])

    def get_shape(self) -> tuple:
        return self._shape

    def get_dtype(self) -> np.dtype:
        return self._dtype

    def get_z_indices(self) -> list:
        return list(range(len(self.tif_files)))

    def close(self):
        """TIF files don't need explicit closing"""
        pass

    def get_file_path(self, z_index: int) -> Path:
        """Get the file path for a specific Z index"""
        return self.tif_files[z_index]
