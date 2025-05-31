import warnings
warnings.filterwarnings("ignore",
                        message=".*NestedDirectoryStore is deprecated.*",
                        category=FutureWarning)

import json
from pathlib import Path
from typing import Union, Optional

import numpy as np
import zarr

from .base import VolumeReader, VolumeWriter


class ZarrVolumeReader(VolumeReader):
    """Reader for Zarr volumes"""

    def __init__(self, zarr_path: Union[str, Path]):
        self.zarr_path = Path(zarr_path)
        self.zarr_array = zarr.open(str(zarr_path), mode='r')
        self._metadata = self._load_metadata()

    def read_slice(self, z_index: int) -> np.ndarray:
        """Read a Z slice from zarr array"""
        return self.zarr_array[z_index]

    def get_shape(self) -> tuple:
        return self.zarr_array.shape

    def get_dtype(self) -> np.dtype:
        return self.zarr_array.dtype

    def get_z_indices(self) -> list:
        return list(range(self.zarr_array.shape[0]))

    def close(self):
        """Zarr arrays don't need explicit closing"""
        pass

    def _load_metadata(self) -> dict:
        """Load metadata from zarr attributes"""
        return dict(self.zarr_array.attrs)


class ZarrVolumeWriter(VolumeWriter):
    """Writer for Zarr volumes - following scroll_to_zarr.py pattern"""

    def __init__(self, output_path: Union[str, Path], shape: tuple, dtype: np.dtype,
                 chunks: Optional[tuple] = None, compression: str = 'blosc'):
        self.output_path = Path(output_path)
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks or (1, min(shape[1], 512), min(shape[2], 512))

        # Use zarr.NestedDirectoryStore pattern from scroll_to_zarr.py
        store = zarr.NestedDirectoryStore(str(output_path))
        self.zarr_array = zarr.open(
            store=store,
            shape=shape,
            chunks=self.chunks,
            dtype=dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=None,  # Following scroll_to_zarr.py pattern
            mode='w'
        )
        self._metadata = {}

    def write_slice(self, z_index: int, data: np.ndarray) -> None:
        """Write a Z slice to zarr array"""
        self.zarr_array[z_index] = data

    def set_metadata(self, metadata: dict) -> None:
        """Set metadata as zarr attributes"""
        self._metadata.update(metadata)
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, list)):
                self.zarr_array.attrs[key] = value

    def finalize(self) -> None:
        """Finalize zarr writing - no flush needed"""
        pass


class OMEZarrVolumeReader(VolumeReader):
    """Reader for OME-Zarr volumes - reads from level 0 by default"""

    def __init__(self, ome_zarr_path: Union[str, Path], level: int = 0):
        self.ome_zarr_path = Path(ome_zarr_path)
        self.level = level

        # OME-Zarr has hierarchical structure - read from specified level
        level_path = self.ome_zarr_path / str(level)
        if not level_path.exists():
            raise ValueError(f"OME-Zarr level {level} not found at {level_path}")

        # Use regular zarr reader for the specific level
        self.zarr_reader = ZarrVolumeReader(level_path)
        self._metadata = self._load_ome_metadata()

    def read_slice(self, z_index: int) -> np.ndarray:
        return self.zarr_reader.read_slice(z_index)

    def get_shape(self) -> tuple:
        return self.zarr_reader.get_shape()

    def get_dtype(self) -> np.dtype:
        return self.zarr_reader.get_dtype()

    def get_z_indices(self) -> list:
        return self.zarr_reader.get_z_indices()

    def close(self):
        self.zarr_reader.close()

    def _load_ome_metadata(self) -> dict:
        """Load OME-specific metadata"""
        metadata = self.zarr_reader._load_metadata()

        # Try to load OME .zattrs from root
        ome_attrs_file = self.ome_zarr_path / '.zattrs'
        if ome_attrs_file.exists():
            import json
            try:
                with open(ome_attrs_file, 'r') as f:
                    ome_attrs = json.load(f)
                metadata['ome_metadata'] = ome_attrs
            except Exception:
                pass

        metadata['ome_level'] = self.level
        return metadata


class OMEZarrVolumeWriter(VolumeWriter):
    """Writer for OME-Zarr volumes - following scroll_to_zarr.py pattern"""

    def __init__(self, output_path: Union[str, Path], shape: tuple, dtype: np.dtype,
                 chunks: Optional[tuple] = None, compression: str = 'blosc'):
        self.output_path = Path(output_path)
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks or (1, min(shape[1], 512), min(shape[2], 512))

        # Create OME-Zarr directory structure
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Create level 0 (full resolution)
        self.zarr_writer = ZarrVolumeWriter(
            self.output_path / "0", shape, dtype, chunks, compression
        )

        self._create_ome_metadata()

    def write_slice(self, z_index: int, data: np.ndarray) -> None:
        """Write to level 0"""
        self.zarr_writer.write_slice(z_index, data)

    def set_metadata(self, metadata: dict) -> None:
        """Set metadata for OME-Zarr"""
        self.zarr_writer.set_metadata(metadata)

    def finalize(self) -> None:
        """Finalize OME-Zarr with proper metadata"""
        self.zarr_writer.finalize()
        self._write_ome_metadata()

    def _create_ome_metadata(self):
        """Create OME-Zarr metadata structure"""
        self.ome_metadata = {
            "multiscales": [{
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"}
                ],
                "datasets": [{"path": "0"}],
                "name": "masked_volume",
                "version": "0.4"
            }]
        }

    def _write_ome_metadata(self):
        """Write OME-Zarr metadata files"""
        # Write .zattrs
        with open(self.output_path / '.zattrs', 'w') as f:
            json.dump(self.ome_metadata, f, indent=2)

        # Write .zgroup
        with open(self.output_path / '.zgroup', 'w') as f:
            json.dump({"zarr_format": 2}, f)
