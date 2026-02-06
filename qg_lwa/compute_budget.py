"""Compute all terms of the LWA budget (excluding latent heating) and save to NetCDF."""

import logging
import pathlib
import typing

import netCDF4
import numpy as np
import typer

import qg_lwa.array_utils
import qg_lwa.budget


_LOG = logging.getLogger(__name__)

app = typer.Typer(help='Compute the full LWA budget (all terms except latent heating).')


def compute_budget(
    *,
    data_dir: pathlib.Path,
    output_directory: pathlib.Path,
    base_name: str,
    max_time: int,
    Ld: float,
) -> None:
    """Compute all LWA budget terms and save to a single NetCDF file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    with netCDF4.Dataset(str(data_dir / (base_name + '.3d.nc'))) as ds:
        qdat = ds.variables['q1'][:, :, :].data
        vdat = ds.variables['v1'][:, :, :].data
        udat = ds.variables['u1'][:, :, :].data
        tdat = ds.variables['tau'][:, :, :].data
        xs = ds.variables['x'][:].data
        ys = ds.variables['y'][:].data

    y_len = len(ys)
    x_len = len(xs)

    with netCDF4.Dataset(str(data_dir / (base_name + '.qref1_2.nc'))) as ds:
        Qref = qg_lwa.array_utils.ensure_TY(ds.variables['qref1'][:, :].data, y_len=y_len, name='qref1')
    with netCDF4.Dataset(str(data_dir / (base_name + '.uref1_2.nc'))) as ds:
        Uref = qg_lwa.array_utils.ensure_TY(ds.variables['uref1'][:, :].data, y_len=y_len, name='uref1')
    with netCDF4.Dataset(str(data_dir / (base_name + '.tref1_2.nc'))) as ds:
        Tref = qg_lwa.array_utils.ensure_TY(ds.variables['tref1'][:, :].data, y_len=y_len, name='tref1')
    with netCDF4.Dataset(str(data_dir / (base_name + '.wac1_2.nc'))) as ds:
        LWAC = qg_lwa.array_utils.ensure_TYX(ds.variables['wac1'][:, :, :].data, y_len=y_len, x_len=x_len, name='wac1')
    with netCDF4.Dataset(str(data_dir / (base_name + '.waa1_2.nc'))) as ds:
        LWAA = qg_lwa.array_utils.ensure_TYX(ds.variables['waa1'][:, :, :].data, y_len=y_len, x_len=x_len, name='waa1')

    LWA = LWAA + LWAC
    _LOG.info('Variables loaded')

    tn = min(
        max_time,
        qdat.shape[0], vdat.shape[0], udat.shape[0], tdat.shape[0],
        Qref.shape[0], Uref.shape[0], Tref.shape[0],
        LWAC.shape[0], LWAA.shape[0],
    )
    qdat = qdat[:tn]
    vdat = vdat[:tn]
    udat = udat[:tn]
    tdat = tdat[:tn]
    Qref = Qref[:tn]
    Uref = Uref[:tn]
    Tref = Tref[:tn]
    LWA = LWA[:tn]

    qe = qdat[:, :, :] - Qref[:, :, np.newaxis]
    ue = udat[:, :, :] - Uref[:, :, np.newaxis]
    ve = vdat[:, :, :]
    te = tdat[:, :, :] - Tref[:, :, np.newaxis]
    _LOG.info('Eddy variables calculated')

    dt = 1.0
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    LWAtend = qg_lwa.budget.lwatend(lwa=LWA, dt=dt)
    Urefadv = qg_lwa.budget.urefadv(lwa=LWA, uref=Uref, dx=dx, filt=False)
    ueqeadv = qg_lwa.budget.ueadv(q=qdat, qref=Qref, u=udat, uref=Uref, dx=dx, dy=dy, filt=False)
    EF_x = qg_lwa.budget.eddyflux_x(ue=ue, ve=ve, dx=dx, filt=False)
    EF_y = qg_lwa.budget.eddyflux_y(ue=ue, ve=ve, dy=dy, filt=False)
    EF_z = qg_lwa.budget.eddyflux_z(ve=ve, te=te, Ld=Ld, filt=False)
    EF = qg_lwa.budget.eddyflux(ve=ve, qe=qe, filt=False)

    RHS = Urefadv + ueqeadv + EF_x + EF_y + EF_z
    RES = LWAtend - RHS
    _LOG.info('Budget calculated')

    sname = 'LH1_' + base_name + '.nc'
    out_path = output_directory / sname
    out_path.unlink(missing_ok=True)

    with netCDF4.Dataset(str(out_path), 'w') as ds_out:
        ds_out.createDimension('time', size=tn)
        ds_out.createDimension('latitude', size=len(ys))
        ds_out.createDimension('longitude', size=len(xs))

        ds_out.createVariable('time', 'f4', dimensions=['time'])
        ds_out.createVariable('latitude', 'f4', dimensions=['latitude'])
        ds_out.createVariable('longitude', 'f4', dimensions=['longitude'])

        ds_out.createVariable('lwatend', 'f4', dimensions=['time', 'latitude', 'longitude'])
        ds_out.createVariable('urefadv', 'f4', dimensions=['time', 'latitude', 'longitude'])
        ds_out.createVariable('ueqeadv', 'f4', dimensions=['time', 'latitude', 'longitude'])
        ds_out.createVariable('ef_x', 'f4', dimensions=['time', 'latitude', 'longitude'])
        ds_out.createVariable('ef_y', 'f4', dimensions=['time', 'latitude', 'longitude'])
        ds_out.createVariable('ef_z', 'f4', dimensions=['time', 'latitude', 'longitude'])
        ds_out.createVariable('res', 'f4', dimensions=['time', 'latitude', 'longitude'])

        ds_out['longitude'][:] = xs[:]
        ds_out['latitude'][:] = ys[:]
        ds_out['time'][:] = np.arange(tn, dtype='f4')

        ds_out['lwatend'][:, :, :] = LWAtend[:, :, :]
        ds_out['urefadv'][:, :, :] = Urefadv[:, :, :]
        ds_out['ueqeadv'][:, :, :] = ueqeadv[:, :, :]
        ds_out['ef_x'][:, :, :] = EF_x[:, :, :]
        ds_out['ef_y'][:, :, :] = EF_y[:, :, :]
        ds_out['ef_z'][:, :, :] = EF_z[:, :, :]
        ds_out['res'][:, :, :] = RES[:, :, :]

    _LOG.info('Output saved to %s', out_path)


@app.command()
def cli(
    *,
    data_dir: typing.Annotated[pathlib.Path, typer.Option(help='Directory containing input NetCDF files')],
    output_directory: typing.Annotated[pathlib.Path, typer.Option(help='Directory to save output NetCDF files')],
    base_name: typing.Annotated[str, typer.Option(help='Base filename, e.g. N128_0.0_2.0_0.1_1.0')],
    max_time: typing.Annotated[int, typer.Option(help='Maximum number of timesteps to process')] = 10000,
    ld: typing.Annotated[float, typer.Option(help='Deformation radius')] = 1.0,
) -> None:
    """Compute the full LWA budget (all terms except latent heating)."""
    compute_budget(
        data_dir=data_dir,
        output_directory=output_directory,
        base_name=base_name,
        max_time=max_time,
        Ld=ld,
    )


if __name__ == '__main__':
    app()
