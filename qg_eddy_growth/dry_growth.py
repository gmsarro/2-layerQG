"""Compute dry eddy growth rates and mode structure for a 2-layer QG model."""

import logging
import pathlib
import typing

import netCDF4
import numpy as np
import numpy.typing
import scipy.linalg  # type: ignore[import-untyped]
import typer

import qg_eddy_growth.matrices


_LOG = logging.getLogger(__name__)

app = typer.Typer(help='Compute dry eddy growth rates from a 2-layer QG model.')


def compute_growth(
    *,
    data_path: pathlib.Path,
    output_file: pathlib.Path,
    beta: float,
    resolution: int,
    max_wavenumber: float,
    sponge_min: int,
    sponge_max: int,
) -> None:
    with netCDF4.Dataset(str(data_path), 'r') as nc_file:
        y: numpy.typing.NDArray[np.floating[typing.Any]] = nc_file.variables['y'][:]
        u1_mean: numpy.typing.NDArray[np.floating[typing.Any]] = np.mean(np.mean(nc_file.variables['u1'][:], axis=2), axis=0)
        u2_mean: numpy.typing.NDArray[np.floating[typing.Any]] = np.mean(np.mean(nc_file.variables['u2'][:], axis=2), axis=0)

    u1: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros_like(u1_mean)
    u2: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros_like(u2_mean)
    u1[sponge_min:sponge_max] = np.copy(u1_mean[sponge_min:sponge_max])
    u2[sponge_min:sponge_max] = np.copy(u2_mean[sponge_min:sponge_max])

    dy: float = float(y[1] - y[0])
    n_2: int = int(len(y) * 2)
    n: int = int(len(y))
    half_matrix: int = n - 2

    growth: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros(resolution)
    mean_growth: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros(resolution)
    kk: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros(resolution)

    range_k: numpy.typing.NDArray[np.floating[typing.Any]] = np.linspace(0, max_wavenumber, resolution)

    loc: int = 0
    for rk in range_k:
        M, N = qg_eddy_growth.matrices.build_matrices(
            u1=u1,
            u2=u2,
            beta=beta,
            dy=dy,
            n_2=n_2,
            rk=float(rk),
            half_matrix=half_matrix,
            n=n,
        )
        evals, V = scipy.linalg.eig(M, N)
        gr = evals.imag * rk
        growth[loc] = np.max(gr)
        mean_growth[loc] = np.mean(np.abs(gr))
        kk[loc] = rk
        _LOG.info('Processed wavenumber %s of %s', rk, max_wavenumber)
        loc += 1

    peak_index: int = int(np.argmax(growth))
    rk_peak: float = float(kk[peak_index])
    M, N = qg_eddy_growth.matrices.build_matrices(
        u1=u1,
        u2=u2,
        beta=beta,
        dy=dy,
        n_2=n_2,
        rk=rk_peak,
        half_matrix=half_matrix,
        n=n,
    )
    evals, V = scipy.linalg.eig(M, N)
    peak_mode_index: int = int(np.argmax(evals.imag * rk_peak))
    peak_mode_structure = V[:, peak_mode_index]

    peak_mode_structure_upper_img: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros_like(y)
    peak_mode_structure_lower_img: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros_like(y)
    peak_mode_structure_upper_real: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros_like(y)
    peak_mode_structure_lower_real: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros_like(y)

    peak_mode_structure_upper_img[1:-1] = peak_mode_structure.imag[:half_matrix]
    peak_mode_structure_lower_img[1:-1] = peak_mode_structure.imag[half_matrix:]
    peak_mode_structure_upper_real[1:-1] = peak_mode_structure.real[:half_matrix]
    peak_mode_structure_lower_real[1:-1] = peak_mode_structure.real[half_matrix:]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(str(output_file), 'w', format='NETCDF4_CLASSIC') as nc_file:
        nc_file.createDimension('k_dim', len(kk))
        nc_file.createDimension('y_dim', len(y))
        nc_file.createVariable('k', 'f8', ('k_dim',))
        nc_file.createVariable('y', 'f8', ('y_dim',))
        nc_file.createVariable('largest_imaginary_eigenvalues', 'f8', ('k_dim',))
        nc_file.createVariable('mean_imaginary_eigenvalues', 'f8', ('k_dim',))
        nc_file.createVariable('optimal_mode_upper_img', 'f8', ('y_dim',))
        nc_file.createVariable('optimal_mode_lower_img', 'f8', ('y_dim',))
        nc_file.createVariable('optimal_mode_upper_real', 'f8', ('y_dim',))
        nc_file.createVariable('optimal_mode_lower_real', 'f8', ('y_dim',))

        nc_file['k'][:] = kk
        nc_file['y'][:] = y
        nc_file['largest_imaginary_eigenvalues'][:] = growth
        nc_file['mean_imaginary_eigenvalues'][:] = mean_growth
        nc_file['optimal_mode_upper_img'][:] = peak_mode_structure_upper_img
        nc_file['optimal_mode_lower_img'][:] = peak_mode_structure_lower_img
        nc_file['optimal_mode_upper_real'][:] = peak_mode_structure_upper_real
        nc_file['optimal_mode_lower_real'][:] = peak_mode_structure_lower_real

    _LOG.info('Saved results to %s', output_file)


@app.command()
def cli(
    *,
    data_path: typing.Annotated[pathlib.Path, typer.Option(
        help='Path to model NetCDF file (must contain u1, u2, y)',
    )],
    output_file: typing.Annotated[pathlib.Path, typer.Option(
        help='Path to save growth-rate NetCDF output',
    )],
    beta: typing.Annotated[float, typer.Option(
        help='Nondimensional beta parameter',
    )] = 0.2,
    resolution: typing.Annotated[int, typer.Option(
        help='Number of wavenumber points to sample',
    )] = 600,
    max_wavenumber: typing.Annotated[float, typer.Option(
        help='Maximum zonal wavenumber to sweep',
    )] = 3.0,
    sponge_min: typing.Annotated[int, typer.Option(
        help='Index of inner boundary of lower sponge layer',
    )] = 31,
    sponge_max: typing.Annotated[int, typer.Option(
        help='Index of inner boundary of upper sponge layer (negative = from end)',
    )] = -32,
) -> None:
    """Compute dry eddy growth rates and peak mode structure."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    compute_growth(
        data_path=data_path,
        output_file=output_file,
        beta=beta,
        resolution=resolution,
        max_wavenumber=max_wavenumber,
        sponge_min=sponge_min,
        sponge_max=sponge_max,
    )


if __name__ == '__main__':
    app()
