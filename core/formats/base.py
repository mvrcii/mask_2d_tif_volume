from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple


class VolumeReader(ABC):
    """Abstract base class for reading volume data from different formats"""

    @abstractmethod
    def read_slice(self, z_index: int) -> np.ndarray:
        """Read a single Z slice from the volume"""
        pass

    @abstractmethod
    def get_shape(self) -> tuple:
        """Get volume shape (z, y, x)"""
        pass

    @abstractmethod
    def get_dtype(self) -> np.dtype:
        """Get volume data type"""
        pass

    @abstractmethod
    def get_z_indices(self) -> list:
        """Get list of available Z indices"""
        pass

    @abstractmethod
    def close(self):
        """Close any open resources"""
        pass


class VolumeWriter(ABC):
    """Abstract base class for writing volume data to different formats"""

    @abstractmethod
    def write_slice(self, z_index: int, data: np.ndarray) -> None:
        """Write a single Z slice to the volume"""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize the volume writing process"""
        pass

    @abstractmethod
    def set_metadata(self, metadata: dict) -> None:
        """Set volume metadata"""
        pass
