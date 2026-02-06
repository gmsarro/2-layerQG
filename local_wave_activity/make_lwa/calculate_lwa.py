"""Prepare input PV, run Fortran programs to compute reference PV and LWA, and move outputs.

The script performs the following steps:
  1. Reads PV, removes sponge layer values by capping to zonal extrema
  2. Writes temporary QGPV.nc in the working directory
  3. Compiles and runs Fortran codes rp2.f90 and rp4.f90
  4. Moves outputs to prefix with suffixes wac1_2.nc, waa1_2.nc, qref1_2.nc
"""

import logging
import pathlib
import shutil
import subprocess
import typing

import numpy as np
import xarray as xr
import typer


_LOG = logging.getLogger(__name__)


def _ensure_dir(
    *,
    path: pathlib.Path,
) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_qgpv(
    *,
    input_nc: pathlib.Path,
    work_dir: pathlib.Path,
) -> pathlib.Path:
    """Read PV, cap sponge values to zonal extrema, and write QGPV.nc."""
    out_path = work_dir / 'QGPV.nc'
    with xr.open_dataset(input_nc) as ds:
        ds_cropped = ds
        q1_mean = np.nanmean(ds_cropped['q1'], axis=2)
        location_max = np.argmax(q1_mean, axis=1)
        location_min = np.argmin(q1_mean, axis=1)
        value_max = np.max(q1_mean, axis=1)
        value_min = np.min(q1_mean, axis=1)
        for i in range(ds.sizes['time']):
            ds_cropped['q1'][i, location_max[i]:, :] = value_max[i]
            ds_cropped['q1'][i, :location_min[i], :] = value_min[i]
        ds_cropped_out = xr.Dataset({'q1': ds_cropped['q1']})
        ds_cropped_out.to_netcdf(out_path)
    out_path.chmod(0o744)
    _LOG.info('Prepared %s', out_path)
    return out_path


def run_fortran(
    *,
    work_dir: pathlib.Path,
) -> None:
    """Compile and run rp2 and rp4 Fortran codes."""
    work_dir_str = str(work_dir)
    subprocess.run(
        'gfortran -c -g -fcheck=all $NC_INC rp2.f90 -o rp2.o',
        shell=True, check=True, cwd=work_dir_str,
    )
    subprocess.run(
        'gfortran -o rp2 rp2.o $NC_LIB',
        shell=True, check=True, cwd=work_dir_str,
    )
    subprocess.run(
        './rp2',
        shell=True, check=True, cwd=work_dir_str,
    )
    subprocess.run(
        'gfortran -c -g -fcheck=all $NC_INC rp4.f90 -o rp4.o',
        shell=True, check=True, cwd=work_dir_str,
    )
    subprocess.run(
        'gfortran -o rp4 rp4.o $NC_LIB',
        shell=True, check=True, cwd=work_dir_str,
    )
    subprocess.run(
        './rp4',
        shell=True, check=True, cwd=work_dir_str,
    )
    _LOG.info('Fortran runs completed')


def move_outputs(
    *,
    work_dir: pathlib.Path,
    output_prefix: pathlib.Path,
) -> None:
    """Move Fortran output files to the target prefix location."""
    prefix_str = str(output_prefix)
    shutil.move(str(work_dir / 'wac1.nc'), prefix_str + 'wac1_2.nc')
    shutil.move(str(work_dir / 'waa1.nc'), prefix_str + 'waa1_2.nc')
    shutil.move(str(work_dir / 'qref1.nc'), prefix_str + 'qref1_2.nc')
    _LOG.info('Moved outputs to prefix %s*', prefix_str)


def cleanup_workdir(
    *,
    work_dir: pathlib.Path,
) -> None:
    """Remove build artifacts from the working directory."""
    for name in ('rp2.o', 'rp4.o', 'rp2', 'rp4', 'QGPV.nc'):
        artifact = work_dir / name
        artifact.unlink(missing_ok=True)
    _LOG.info('Cleaned working directory %s', work_dir)


def main(
    *,
    input_nc: pathlib.Path,
    work_dir: pathlib.Path,
    output_prefix: pathlib.Path,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    _ensure_dir(path=work_dir)
    _ensure_dir(path=output_prefix.parent)
    prepare_qgpv(input_nc=input_nc, work_dir=work_dir)
    run_fortran(work_dir=work_dir)
    move_outputs(work_dir=work_dir, output_prefix=output_prefix)
    cleanup_workdir(work_dir=work_dir)
    _LOG.info('Wrote outputs with prefix %s', output_prefix)


def cli(
    *,
    input_nc: typing.Annotated[pathlib.Path, typer.Option(help='Path to model NetCDF file with q1')],
    work_dir: typing.Annotated[pathlib.Path, typer.Option(help='Directory containing rp2.f90 and rp4.f90; will write QGPV.nc and build artifacts here')],
    output_prefix: typing.Annotated[pathlib.Path, typer.Option(help='Prefix for output files, e.g., /path/to/N128_L_C_E_U.')],
) -> None:
    print('Preparing QGPV and running Fortran LWA computations...')
    main(input_nc=input_nc, work_dir=work_dir, output_prefix=output_prefix)
    print('Done.')


if __name__ == '__main__':
    typer.run(cli)
