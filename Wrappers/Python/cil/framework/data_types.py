from enum import Enum

class DType(Enum):
    """Enum encompassing the array-api-compat backend and the numerical type."""
    
    # Numpy
    NUMPY_FLOAT32 = ('numpy', 'float32')
    NUMPY_FLOAT64 = ('numpy', 'float64')
    NUMPY_COMPLEX64 = ('numpy', 'complex64')
    
    # CuPy
    CUPY_FLOAT32 = ('cupy', 'float32')
    CUPY_FLOAT64 = ('cupy', 'float64')
    CUPY_COMPLEX64 = ('cupy', 'complex64')
    
    # PyTorch
    TORCH_FLOAT32 = ('torch', 'float32')
    TORCH_FLOAT64 = ('torch', 'float64')
    TORCH_COMPLEX64 = ('torch', 'complex64')

    def __init__(self, backend, numeric_type):
        # Unpack the tuple into accessible properties
        self.backend = backend
        self.numeric_type = numeric_type

    @property
    def xp(self):
        """Lazily imports and returns the array-api-compat namespace."""
        if self.backend == 'numpy':
            import array_api_compat.numpy as xp
        elif self.backend == 'cupy':
            import array_api_compat.cupy as xp
        elif self.backend == 'torch':
            import array_api_compat.torch as xp
        else:
            raise ImportError(f"Unsupported backend: {self.backend}")
        return xp

    @property
    def api_dtype(self):
        """Returns the actual resolved dtype object (e.g., xp.float32)."""
        return getattr(self.xp, self.numeric_type)