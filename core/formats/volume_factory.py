from pathlib import Path
from typing import Union

from .base import VolumeReader
from .tif_handler import TifVolumeReader
from .zarr_handler import ZarrVolumeReader, OMEZarrVolumeReader


def create_volume_reader(input_path: Union[str, Path]) -> VolumeReader:
    """Factory function to create appropriate volume reader"""
    input_path = Path(input_path)

    if input_path.is_dir():
        # Check if it's an OME-Zarr directory (has .zattrs, .zgroup, and numbered subdirs)
        if ((input_path / '.zattrs').exists() and
                (input_path / '.zgroup').exists() and
                (input_path / '0').exists()):
            return OMEZarrVolumeReader(input_path, level=0)

        # Check if it's a regular zarr directory
        elif (input_path / '.zarray').exists() or (input_path / '.zgroup').exists():
            return ZarrVolumeReader(input_path)

        # Check if it contains TIF files
        elif list(input_path.glob("*.tif")):
            return TifVolumeReader(input_path)
        else:
            raise ValueError(f"Directory {input_path} doesn't contain TIF files, zarr, or OME-zarr data")

    elif input_path.suffix == '.zarr' or input_path.name.endswith('.zarr'):
        # Could be either zarr or OME-zarr - check for OME structure
        if ((input_path / '.zattrs').exists() and
                (input_path / '.zgroup').exists() and
                (input_path / '0').exists()):
            return OMEZarrVolumeReader(input_path, level=0)
        else:
            return ZarrVolumeReader(input_path)

    else:
        raise ValueError(f"Unsupported input format: {input_path}")


def detect_input_format(input_path: Union[str, Path]) -> str:
    """Detect the format of input volume"""
    input_path = Path(input_path)

    if input_path.is_dir():
        # Check for OME-Zarr first (more specific)
        if ((input_path / '.zattrs').exists() and
                (input_path / '.zgroup').exists() and
                (input_path / '0').exists()):
            return 'ome-zarr'

        # Check for regular zarr
        elif (input_path / '.zarray').exists() or (input_path / '.zgroup').exists():
            return 'zarr'

        # Check for TIF stack
        elif list(input_path.glob("*.tif")):
            return 'tif_stack'

    elif input_path.suffix == '.zarr' or input_path.name.endswith('.zarr'):
        # Could be either zarr or OME-zarr - check for OME structure
        if ((input_path / '.zattrs').exists() and
                (input_path / '.zgroup').exists() and
                (input_path / '0').exists()):
            return 'ome-zarr'
        else:
            return 'zarr'

    raise ValueError(f"Cannot detect format for: {input_path}")