"""
Compute eddy growth rate and mode structure for a 2-layer QG model.

Typical usage example:

```python
# Run script to compute growth rates and save outputs
python Calculate_Eddy_Growth.py --data-path /path/to/model.nc --output-file results.nc
```

The script performs the following steps:
    1. Loads 3D model data and computes zonal/time means of upper and lower wind fields
    2. Removes sponge layers at the meridional boundaries
    3. Sweeps wavenumbers and solves a generalized eigenvalue problem to compute growth rates
    4. Extracts the peak mode structure and saves results to NetCDF
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eig
import netCDF4 as nc

from build_matrices import build_matrices


# Constants
BETA: Final[float] = 0.2
RESOLUTION: Final[int] = 600
MAX_WAVENUMBER: Final[int] = 3
SPONGE_LAYER_MIN_INDEX: Final[int] = 11
SPONGE_LAYER_MAX_INDEX: Final[int] = -12


_LOG = logging.getLogger(__name__)


def compute_growth(
    data_path: Path,
    output_file: Path,
) -> None:
    with nc.Dataset(str(data_path), 'r') as nc_file:
        y: NDArray[np.floating] = nc_file.variables['y'][:]
        u1_mean: NDArray[np.floating] = np.mean(np.mean(nc_file.variables['u1'][:], axis=2), axis=0)
        u2_mean: NDArray[np.floating] = np.mean(np.mean(nc_file.variables['u2'][:], axis=2), axis=0)

    u1: NDArray[np.floating] = np.copy(u1_mean) * 0
    u2: NDArray[np.floating] = np.copy(u2_mean) * 0
    u1[SPONGE_LAYER_MIN_INDEX:SPONGE_LAYER_MAX_INDEX] = np.copy(
        u1_mean[SPONGE_LAYER_MIN_INDEX:SPONGE_LAYER_MAX_INDEX]
    )
    u2[SPONGE_LAYER_MIN_INDEX:SPONGE_LAYER_MAX_INDEX] = np.copy(
        u2_mean[SPONGE_LAYER_MIN_INDEX:SPONGE_LAYER_MAX_INDEX]
    )

    dy: float = float(y[1] - y[0])
    n_2: int = int(len(y) * 2)
    n: int = int(len(y))
    half_maxtrix: int = n - 2

    growth: NDArray[np.floating] = np.zeros(RESOLUTION)
    mean_growth: NDArray[np.floating] = np.zeros(RESOLUTION)
    kk: NDArray[np.floating] = np.zeros(RESOLUTION)

    range_k: NDArray[np.floating] = np.linspace(0, MAX_WAVENUMBER, RESOLUTION)

    loc: int = 0
    for rk in range_k:
        M, N = build_matrices(u1, u2, BETA, dy, n_2, float(rk), half_maxtrix, n)
        evals, V = eig(M, N)
        gr = evals.imag * rk
        growth[loc] = np.max(gr)
        mean_growth[loc] = np.mean(np.abs(gr))
        kk[loc] = rk
        _LOG.info('Processed wavenumber %s of %s', rk, MAX_WAVENUMBER)
        loc += 1

    peak_index: int = int(np.argmax(growth))
    rk_peak: float = float(kk[peak_index])
    M, N = build_matrices(u1, u2, BETA, dy, n_2, rk_peak, half_maxtrix, n)
    evals, V = eig(M, N)
    peak_mode_index: int = int(np.argmax(evals.imag * rk_peak))
    peak_mode_structure = V[:, peak_mode_index]

    peak_mode_structure_upper_img: NDArray[np.floating] = np.copy(y) * 0
    peak_mode_structure_lower_img: NDArray[np.floating] = np.copy(y) * 0
    peak_mode_structure_upper_real: NDArray[np.floating] = np.copy(y) * 0
    peak_mode_structure_lower_real: NDArray[np.floating] = np.copy(y) * 0

    peak_mode_structure_upper_img[1:-1] = peak_mode_structure.imag[:half_maxtrix]
    peak_mode_structure_lower_img[1:-1] = peak_mode_structure.imag[half_maxtrix:]
    peak_mode_structure_upper_real[1:-1] = peak_mode_structure.real[:half_maxtrix]
    peak_mode_structure_lower_real[1:-1] = peak_mode_structure.real[half_maxtrix:]

    with nc.Dataset(str(output_file), 'w', format='NETCDF4_CLASSIC') as nc_file:
        nc_file.createDimension('k_dim', len(kk))
        nc_file.createDimension('y_dim', len(y))
        nc_file.createVariable('k', 'f8', ('k_dim',))
        nc_file.createVariable('y', 'f8', ('y_dim',))
        nc_file.createVariable('largest_imaginary_eigenvalues', 'f8', ('k_dim',))
        nc_file.createVariable('mean_imaginary_eigenvalues', 'f8', ('k_dim',))
        nc_file.createVariable('optimal_mode_upper_img', 'f8', ('y_dim', ))
        nc_file.createVariable('optimal_mode_lower_img', 'f8', ('y_dim', ))
        nc_file.createVariable('optimal_mode_upper_real', 'f8', ('y_dim', ))
        nc_file.createVariable('optimal_mode_lower_real', 'f8', ('y_dim', ))

        nc_file['k'][:] = kk
        nc_file['y'][:] = y
        nc_file['largest_imaginary_eigenvalues'][:] = growth
        nc_file['mean_imaginary_eigenvalues'][:] = mean_growth
        nc_file['optimal_mode_upper_img'][:] = peak_mode_structure_upper_img
        nc_file['optimal_mode_lower_img'][:] = peak_mode_structure_lower_img
        nc_file['optimal_mode_upper_real'][:] = peak_mode_structure_upper_real
        nc_file['optimal_mode_lower_real'][:] = peak_mode_structure_lower_real


def main(data_path: Path, output_file: Path) -> None:
    _LOG.info('Starting eddy growth computation for %s', data_path)
    compute_growth(data_path=data_path, output_file=output_file)
    _LOG.info('Saved results to %s', output_file)


def cli(
    data_path: Path,
    output_file: Path,
) -> None:
    """CLI entry point for computing eddy growth and mode structure."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    main(data_path, output_file)


if __name__ == '__main__':
    # Lightweight CLI without Typer dependency when run as a script
    import argparse

    parser = argparse.ArgumentParser(description='Compute eddy growth for a 2-layer QG model')
    parser.add_argument('--data-path', type=Path, required=True, help='Path to model NetCDF file')
    parser.add_argument('--output-file', type=Path, required=True, help='Path to save NetCDF output')
    args = parser.parse_args()

    cli(args.data_path, args.output_file)
