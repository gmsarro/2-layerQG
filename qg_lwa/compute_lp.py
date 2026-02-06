"""Compute the latent heating contribution of the LWA budget and save to NetCDF."""

import logging
import pathlib
import typing

import netCDF4
import numpy as np
import typer

import qg_lwa.array_utils
import qg_lwa.budget


_LOG = logging.getLogger(__name__)

app = typer.Typer(help='Compute the latent heating term of the LWA budget.')


def compute_lp(
    *,
    data_dir: pathlib.Path,
    output_directory: pathlib.Path,
    base_name: str,
    latent_heating: float,
    max_time: int,
) -> None:
    """Compute the latent heating LWA budget term and save to NetCDF."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    with netCDF4.Dataset(str(data_dir / (base_name + '.3d.nc'))) as ds:
        qdat = ds.variables['q1'][:, :, :].data
        pdat = ds.variables['P'][:, :, :].data
        xs = ds.variables['x'][:].data
        ys = ds.variables['y'][:].data

    y_len = len(ys)

    with netCDF4.Dataset(str(data_dir / (base_name + '.qref1_2.nc'))) as ds:
        Qref = qg_lwa.array_utils.ensure_TY(ds.variables['qref1'][:, :].data, y_len=y_len, name='qref1')
    _LOG.info('Variables loaded')

    tn = min(max_time, qdat.shape[0], pdat.shape[0], Qref.shape[0])
    qdat = qdat[:tn]
    pdat = pdat[:tn]
    Qref = Qref[:tn]

    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    LP = qg_lwa.budget.LH(p=pdat, q=qdat, qref=Qref, L=latent_heating, dx=dx, dy=dy, filt=False)
    _LOG.info('Budget calculated')

    sname = 'LP_' + base_name + '.nc'
    out_path = output_directory / sname
    out_path.unlink(missing_ok=True)

    with netCDF4.Dataset(str(out_path), 'w') as ds_out:
        ds_out.createDimension('time', size=tn)
        ds_out.createDimension('latitude', size=len(ys))
        ds_out.createDimension('longitude', size=len(xs))

        ds_out.createVariable('time', 'f4', dimensions=['time'])
        ds_out.createVariable('latitude', 'f4', dimensions=['latitude'])
        ds_out.createVariable('longitude', 'f4', dimensions=['longitude'])
        ds_out.createVariable('LH', 'f4', dimensions=['time', 'latitude', 'longitude'])

        ds_out['longitude'][:] = xs[:]
        ds_out['latitude'][:] = ys[:]
        ds_out['time'][:] = np.arange(tn, dtype='f4')
        ds_out['LH'][:, :, :] = LP[:, :, :]

    _LOG.info('Output saved to %s', out_path)


@app.command()
def cli(
    *,
    data_dir: typing.Annotated[pathlib.Path, typer.Option(help='Directory containing input NetCDF files')],
    output_directory: typing.Annotated[pathlib.Path, typer.Option(help='Directory to save output NetCDF files')],
    base_name: typing.Annotated[str, typer.Option(help='Base filename, e.g. N128_0.0_2.0_0.1_1.0')],
    latent_heating: typing.Annotated[float, typer.Option(help='Latent heating parameter L')],
    max_time: typing.Annotated[int, typer.Option(help='Maximum number of timesteps to process')] = 10000,
) -> None:
    """Compute the latent heating contribution of the LWA budget."""
    compute_lp(
        data_dir=data_dir,
        output_directory=output_directory,
        base_name=base_name,
        latent_heating=latent_heating,
        max_time=max_time,
    )


if __name__ == '__main__':
    app()
