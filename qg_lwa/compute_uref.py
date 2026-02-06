"""Calculate U_REF and T_REF from Q_REF; all variables are nondimensionalized."""

import logging
import pathlib
import typing
from typing import Any

import netCDF4
import numpy as np
import numpy.typing as npt
import typer

import qg_lwa.array_utils

NDArrayF = npt.NDArray[np.floating[Any]]


_LOG = logging.getLogger(__name__)

app = typer.Typer(help='Compute reference wind and temperature from Q_ref.')


def solve_uref(
    *,
    qref: NDArrayF,
    um: NDArrayF,
    umb: NDArrayF,
    ys: NDArrayF,
    beta: float,
    Ld: float,
    maxerr: float = 1e-6,
    max_iterations: int = 10000,
    relax: float = 1.9,
) -> NDArrayF:
    """Solve SOR for U_REF given Q_REF gradient and boundary conditions.

    Parameters
    ----------
    qref : array (time, latitude)
        Reference PV.
    um : array (time, latitude)
        Zonal-mean U (upper) for initial guess and boundary.
    umb : array (time, latitude)
        Zonal-mean U (lower) used in forcing term.
    ys : array (latitude,)
        Latitude coordinate (monotonic, evenly spaced).
    beta : float
        Nondimensional beta parameter.
    Ld : float
        Deformation radius.
    maxerr : float
        Convergence tolerance.
    max_iterations : int
        Maximum SOR iterations.
    relax : float
        Relaxation factor.

    Returns
    -------
    array (time, latitude)
        Computed U_REF.
    """
    tn, yn = np.shape(qref)
    dy = ys[1] - ys[0]
    AC = np.array([1 / dy**2, -2 / dy**2, 1 / dy**2])
    qref_y = np.zeros((tn, yn))
    qref_y[:, 1:-1] = (qref[:, 2:] - qref[:, :-2]) / (2 * dy)
    uref = np.zeros((tn, yn)) + um[:, :]
    n_iterations = 0
    err = 1e5
    while n_iterations < max_iterations and err > maxerr:
        utemp = uref.copy()
        for y in range(1, yn - 1):
            RS = (
                AC[0] * uref[:, y - 1] + AC[1] * uref[:, y] + AC[2] * uref[:, y + 1]
            ) - beta + qref_y[:, y] - uref[:, y] / Ld**2 + umb[:, y] / Ld**2
            uref[:, y] = uref[:, y] - relax * RS / (AC[1] - 1 / Ld**2)
        err = float(np.max(np.abs(uref - utemp)))
        n_iterations += 1
    if n_iterations == max_iterations:
        _LOG.info('Not fully converged')
    else:
        _LOG.info('Converged at %s iterations', n_iterations)
    return uref


def integrate_tref(
    *,
    uref: NDArrayF,
    ys: NDArrayF,
    tm: NDArrayF,
) -> NDArrayF:
    """Integrate for T_REF from U_REF shear and adjust mean to match boundary template.

    Parameters
    ----------
    uref : array (time, latitude)
        Reference zonal wind.
    ys : array (latitude,)
        Latitude coordinate.
    tm : array (time, latitude)
        Zonal-mean temperature used for offset constraint.

    Returns
    -------
    array (time, latitude)
        Reference temperature.
    """
    tn, yn = np.shape(uref)
    dy = ys[1] - ys[0]
    tref = np.zeros((tn, yn))
    ushear = (uref[:, 1:] + uref[:, :-1]) * 0.5
    for y in range(yn - 1):
        tref[:, y + 1] = tref[:, 0] - np.sum(ushear[:, :y + 1] * dy, axis=1)
    for t in range(tn):
        offset = np.mean(tm[t, :] - tref[t, :])
        tref[t, :] += offset
    return tref


def compute_and_save(
    *,
    data_dir: pathlib.Path,
    output_directory: pathlib.Path,
    base_name: str,
    beta: float,
    Ld: float,
    max_time: int,
) -> None:
    """Compute U_REF and T_REF and save to NetCDF."""
    qref_path = data_dir / (base_name + '.qref1_2.nc')
    mean_path = data_dir / (base_name + '.nc')

    with netCDF4.Dataset(str(mean_path)) as ds_mean:
        um = ds_mean.variables['zu1'][:, :].data
        umb = ds_mean.variables['zu2'][:, :].data
        tm = ds_mean.variables['ztau'][:, :].data
        ys = ds_mean.variables['y'][:].data

    yn = len(ys)

    with netCDF4.Dataset(str(qref_path)) as ds_qref:
        qref_raw = ds_qref.variables['qref1'][:, :].data
    qref = qg_lwa.array_utils.ensure_TY(qref_raw, y_len=yn, name='qref1')

    tn = min(max_time, qref.shape[0], um.shape[0], umb.shape[0], tm.shape[0])
    qref = qref[:tn, :]
    um = um[:tn, :]
    umb = umb[:tn, :]
    tm = tm[:tn, :]

    uref = solve_uref(qref=qref, um=um, umb=umb, ys=ys, beta=beta, Ld=Ld)
    tref = integrate_tref(uref=uref, ys=ys, tm=tm)

    out_u = output_directory / (base_name + '.uref1_2.nc')
    out_t = output_directory / (base_name + '.tref1_2.nc')
    out_u.unlink(missing_ok=True)
    out_t.unlink(missing_ok=True)

    with netCDF4.Dataset(str(out_u), 'w') as ds_out:
        ds_out.createDimension('time', size=tn)
        ds_out.createDimension('latitude', size=yn)
        ds_out.createVariable('uref1', 'f4', dimensions=['time', 'latitude'])
        ds_out.createVariable('time', 'f4', dimensions=['time'])
        ds_out.createVariable('y', 'f4', dimensions=['latitude'])
        ds_out['uref1'][:, :] = uref[:, :]
        ds_out['y'][:] = ys[:]
        ds_out['time'][:] = np.arange(tn)
    _LOG.info('Saved %s', out_u)

    with netCDF4.Dataset(str(out_t), 'w') as ds_out:
        ds_out.createDimension('time', size=tn)
        ds_out.createDimension('latitude', size=yn)
        ds_out.createVariable('tref1', 'f4', dimensions=['time', 'latitude'])
        ds_out.createVariable('time', 'f4', dimensions=['time'])
        ds_out.createVariable('y', 'f4', dimensions=['latitude'])
        ds_out['tref1'][:, :] = tref[:, :]
        ds_out['y'][:] = ys[:]
        ds_out['time'][:] = np.arange(tn)
    _LOG.info('Saved %s', out_t)


@app.command()
def cli(
    *,
    data_dir: typing.Annotated[pathlib.Path, typer.Option(help='Input directory containing qref and mean fields')],
    output_directory: typing.Annotated[pathlib.Path, typer.Option(help='Output directory')],
    base_name: typing.Annotated[str, typer.Option(help='Base filename, e.g. N128_0.0_2.0_0.1_1.0')],
    beta: typing.Annotated[float, typer.Option(help='Nondimensional beta')] = 0.2,
    ld: typing.Annotated[float, typer.Option(help='Deformation radius')] = 1.0,
    max_time: typing.Annotated[int, typer.Option(help='Maximum number of timesteps to process')] = 10000,
) -> None:
    """Compute reference wind (U_ref) and temperature (T_ref) from Q_ref."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    compute_and_save(
        data_dir=data_dir,
        output_directory=output_directory,
        base_name=base_name,
        beta=beta,
        Ld=ld,
        max_time=max_time,
    )


if __name__ == '__main__':
    app()
