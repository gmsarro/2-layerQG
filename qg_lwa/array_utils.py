"""Utilities for coercing array orientations from NetCDF files.

The Fortran LWA codes may write arrays as (Y, T) or (X, Y, T) rather
than the expected (T, Y) or (T, Y, X).  These helpers detect the layout
and transpose when necessary.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.floating[Any]]


def ensure_TY(
    arr: NDArrayF,
    *,
    y_len: int,
    name: str = '',
) -> NDArrayF:
    """Return *arr* as (time, y).  Accepts (T, Y) or (Y, T)."""
    if arr.ndim != 2:
        raise ValueError('%s must be 2-D, got %d-D with shape %s' % (name, arr.ndim, arr.shape))
    s0, s1 = arr.shape
    if s1 == y_len:
        return arr
    if s0 == y_len:
        return arr.T
    raise ValueError('%s: no dim equals y_len=%d; shape=%s' % (name, y_len, arr.shape))


def ensure_TYX(
    arr: NDArrayF,
    *,
    y_len: int,
    x_len: int,
    name: str = '',
) -> NDArrayF:
    """Return *arr* as (time, y, x).  Handles common permutations."""
    if arr.ndim != 3:
        raise ValueError('%s must be 3-D, got %d-D with shape %s' % (name, arr.ndim, arr.shape))
    s0, s1, s2 = arr.shape
    if (s1, s2) == (y_len, x_len):
        return arr
    if (s0, s1) == (y_len, x_len):
        return np.transpose(arr, (2, 0, 1))
    if (s0, s1) == (x_len, y_len):
        return np.transpose(arr, (2, 1, 0))
    if (s1, s2) == (x_len, y_len):
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(
        '%s: cannot coerce shape %s to (T, %d, %d)' % (name, arr.shape, y_len, x_len)
    )
