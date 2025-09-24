"""
Compute the latent heating contribution of the LWA budget and save to NetCDF.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

import numpy as np
import netCDF4
import typer
from typing_extensions import Annotated

from lwabudget import LH


_LOG = logging.getLogger(__name__)


def compute_lp(
    *,
    load_dir: Path,
    save_dir: Path,
) -> None:
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
	)
	loaddir = str(load_dir) if str(load_dir).endswith('/') else str(load_dir) + '/'
	savedir = str(save_dir) if str(save_dir).endswith('/') else str(save_dir) + '/'

	Llist = np.arange(0.0)
	Ulist = np.array([1.0], dtype=float)
	max_lenghth: Final[int] = 10000

	sname = 'LP_%s_2.0_0.1_%s.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))
	with netCDF4.Dataset(loaddir+'N128_%s_2.0_0.1_%s.3d.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))) as read:
		qdat = read.variables['q1'][:,:,:].data
		pdat = read.variables['P'][:,:,:].data
		xs = read.variables['x'][:].data
		ys = read.variables['y'][:].data
	with netCDF4.Dataset(loaddir+'N128_%s_2.0_0.1_%s.qref1_2.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))) as read:
		Qref = read.variables['qref1'][:,:].data
	_LOG.info('variables loaded')

	L = float(Llist[0])

	times = np.linspace(0, max_lenghth, max_lenghth, endpoint=False)[:]
	dt = times[1]-times[0]
	dx = xs[1]-xs[0]
	dy = ys[1]-ys[0]

	LP = LH(pdat, qdat, Qref, L, dx, dy, filt=False)
	_LOG.info('budget calculated')

	with netCDF4.Dataset(savedir+sname,'w') as write:
		write.createDimension('time', size=len(times))
		write.createDimension('latitude', size=len(ys))
		write.createDimension('longitude', size=len(xs))

		time = write.createVariable('time','f4',dimensions=['time'])
		latitude = write.createVariable('latitude','f4',dimensions=['latitude'])
		longitude = write.createVariable('longitude','f4',dimensions=['longitude'])

		term1 = write.createVariable('LH','f4',dimensions=['time','latitude','longitude'])

		longitude[:]=xs[:]
		latitude[:]=ys[:]
		time[:]=times

		term1[:,:,:]=LP[:,:,:]
	_LOG.info('output saved; done')


def cli(
    *,
    load_dir: Annotated[Path, typer.Option(help='Directory containing input NetCDF files')],
    save_dir: Annotated[Path, typer.Option(help='Directory to save output NetCDF files')],
) -> None:
	print('Computing latent heating contribution of the LWA budget...')
	compute_lp(load_dir=load_dir, save_dir=save_dir)
	print('Done.')


if __name__ == '__main__':
	typer.run(cli)
