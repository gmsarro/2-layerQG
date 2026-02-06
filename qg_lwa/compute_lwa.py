"""Compute reference PV and local wave activity from model PV.

Uses an f2py-wrapped Fortran module (``lwa_2layer``) for the heavy
computation.  Dimensions and grid extents are read from the input file
and passed as arguments -- no hardcoded resolution.

Pipeline:
  1. Read PV, mask sponge-layer artefacts (cap to zonal extrema)
  2. Call Fortran ``calc_qref`` to compute the equivalent-latitude reference PV
  3. Call Fortran ``calc_lwa`` to compute cyclonic and anticyclonic LWA
  4. Write ``<stem>.qref1_2.nc``, ``<stem>.waa1_2.nc``, ``<stem>.wac1_2.nc``
"""

import importlib
import logging
import pathlib
import subprocess
import sys
import typing

import numpy as np
import xarray as xr
import typer


_LOG = logging.getLogger(__name__)
_FORTRAN_DIR = pathlib.Path(__file__).resolve().parent / 'fortran'

app = typer.Typer(help='Compute reference PV and LWA from model PV (any resolution).')


def _build_fortran() -> None:
    """Compile lwa_2layer.f90 via f2py (requires gfortran)."""
    src = _FORTRAN_DIR / 'lwa_2layer.f90'
    if not src.exists():
        raise FileNotFoundError(
            'Fortran source not found: %s' % src,
        )
    _LOG.info('Compiling Fortran module lwa_2layer ...')
    subprocess.run(
        [sys.executable, '-m', 'numpy.f2py',
         '-c', '--opt=-O2 -fbounds-check',
         '-m', 'lwa_2layer', str(src)],
        check=True,
        cwd=str(_FORTRAN_DIR),
    )
    _LOG.info('Fortran compilation successful')


def _get_fortran() -> typing.Any:
    """Import the lwa_2layer Fortran module, compiling on first use."""
    fortran_str = str(_FORTRAN_DIR)
    if fortran_str not in sys.path:
        sys.path.insert(0, fortran_str)
    try:
        return importlib.import_module('lwa_2layer')
    except ImportError:
        _build_fortran()
        return importlib.import_module('lwa_2layer')


def _find_dim(
    ds: xr.Dataset,
    *,
    name: str,
    fallbacks: tuple[str, ...] = (),
) -> str:
    """Find a dimension by name with fallback aliases."""
    for candidate in (name, *fallbacks):
        if candidate in ds.dims:
            return candidate
    raise ValueError(
        '%s (or %s) not found in dimensions %s' % (name, fallbacks, tuple(ds.dims)),
    )


def mask_sponge(
    *,
    q1: xr.DataArray,
    tdim: str,
    ydim: str,
    xdim: str,
) -> xr.DataArray:
    """Cap PV to zonal-mean extrema to remove sponge-layer artefacts."""
    q1_mean = q1.mean(xdim, skipna=True)
    loc_max = q1_mean.argmax(ydim, skipna=True)
    loc_min = q1_mean.argmin(ydim, skipna=True)
    val_max = q1_mean.max(ydim, skipna=True)
    val_min = q1_mean.min(ydim, skipna=True)
    yindex = np.arange(q1[ydim].size) * xr.ones_like(q1[ydim])
    q1 = q1.where(yindex < loc_max, val_max)  # type: ignore[operator]
    q1 = q1.where(yindex >= loc_min, val_min)  # type: ignore[operator]
    return q1


def compute_lwa_from_model_pv(
    *,
    input_nc: pathlib.Path,
    output_dir: pathlib.Path,
    lx: float,
    ly: float,
    max_time: typing.Optional[int] = None,
) -> None:
    """Full pipeline: read model PV, compute qref and LWA, write outputs."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    fortran = _get_fortran()
    output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(input_nc) as ds:
        tdim = _find_dim(ds, name='time', fallbacks=('t',))
        ydim = _find_dim(ds, name='y', fallbacks=('lat', 'latitude'))
        xdim = _find_dim(ds, name='x', fallbacks=('lon', 'longitude'))

        ds = ds.assign_coords({tdim: np.arange(ds[tdim].size)})
        q1 = ds['q1'].isel({tdim: slice(0, max_time)}).transpose(tdim, ydim, xdim)
    _LOG.info('Loaded q1 with shape %s', q1.shape)

    q1_masked = mask_sponge(q1=q1, tdim=tdim, ydim=ydim, xdim=xdim)
    q1_np = np.asfortranarray(q1_masked.values.transpose(2, 1, 0))
    _LOG.info('PV masked; calling Fortran calc_qref ...')

    qref = fortran.lwa_2layer.calc_qref(q1_np, wx=lx, wy=ly)
    _LOG.info('calc_qref done; calling calc_lwa ...')

    waa, wac = fortran.lwa_2layer.calc_lwa(q1_np, qref, wx=lx, wy=ly)
    _LOG.info('calc_lwa done')

    qref_da = xr.DataArray(
        qref.T,
        dims=(tdim, ydim),
        coords={tdim: q1.coords[tdim], ydim: q1.coords[ydim]},
        name='qref1',
    )
    waa_da = xr.DataArray(
        waa.transpose(2, 1, 0),
        dims=(tdim, ydim, xdim),
        coords={tdim: q1.coords[tdim], ydim: q1.coords[ydim], xdim: q1.coords[xdim]},
        name='waa1',
    )
    wac_da = xr.DataArray(
        wac.transpose(2, 1, 0),
        dims=(tdim, ydim, xdim),
        coords={tdim: q1.coords[tdim], ydim: q1.coords[ydim], xdim: q1.coords[xdim]},
        name='wac1',
    )

    stem = input_nc.stem
    if stem.endswith('.3d'):
        stem = stem[:-3]

    for da, suffix in [(qref_da, 'qref1_2'), (waa_da, 'waa1_2'), (wac_da, 'wac1_2')]:
        out_path = output_dir / ('%s.%s.nc' % (stem, suffix))
        out_path.unlink(missing_ok=True)
        da.to_dataset(name=da.name).to_netcdf(out_path)
        _LOG.info('Wrote %s', out_path)


@app.command()
def cli(
    *,
    input_nc: typing.Annotated[pathlib.Path, typer.Option(
        help='Path to model NetCDF file containing q1 (any resolution)',
    )],
    output_dir: typing.Annotated[pathlib.Path, typer.Option(
        help='Directory for output files (qref, waa, wac)',
    )],
    lx: typing.Annotated[float, typer.Option(
        help='Zonal domain extent (nondimensional)',
    )] = 72.0,
    ly: typing.Annotated[float, typer.Option(
        help='Meridional domain extent (nondimensional)',
    )] = 96.0,
    max_time: typing.Annotated[typing.Optional[int], typer.Option(
        help='Maximum timesteps to process (default: all)',
    )] = None,
) -> None:
    """Compute reference PV and (anti)cyclonic LWA from model PV."""
    compute_lwa_from_model_pv(
        input_nc=input_nc,
        output_dir=output_dir,
        lx=lx,
        ly=ly,
        max_time=max_time,
    )


if __name__ == '__main__':
    app()
