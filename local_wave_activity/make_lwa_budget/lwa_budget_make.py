"""
Compute all terms of the LWA budget (excluding latent heating contribution) and save to NetCDF.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import netCDF4
import typer
from typing_extensions import Annotated
from lwabudget import lwatend, urefadv, ueadv, eddyflux_x, eddyflux_y, eddyflux_z, eddyflux


_LOG = logging.getLogger(__name__)


def compute_budget(
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
	Ld: Final[float] = 1.0

	sname = 'LH1_%s_2.0_0.1_%s.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))

	with netCDF4.Dataset(loaddir+'N128_%s_2.0_0.1_%s.3d.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))) as read:
		qdat = read.variables['q1'][:,:,:].data
		vdat = read.variables['v1'][:,:,:].data
		udat = read.variables['u1'][:,:,:].data
		tdat = read.variables['tau'][:,:,:].data
		xs = read.variables['x'][:].data
		ys = read.variables['y'][:].data

	with netCDF4.Dataset(loaddir+'N128_%s_2.0_0.1_%s.qref1_2.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))) as read:
		Qref = read.variables['qref1'][:,:].data
	with netCDF4.Dataset(loaddir+'N128_%s_2.0_0.1_%s.uref1_2.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))) as read:
		Uref = read.variables['uref1'][:,:].data
	with netCDF4.Dataset(loaddir+'N128_%s_2.0_0.1_%s.tref1_2.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))) as read:
		Tref = read.variables['tref1'][:,:].data
	with netCDF4.Dataset(loaddir+'N128_%s_2.0_0.1_%s.wac1_2.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))) as read:
		LWAC = read.variables['wac1'][:,:,:].data
	with netCDF4.Dataset(loaddir+'N128_%s_2.0_0.1_%s.waa1_2.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))) as read:
		LWAA = read.variables['waa1'][:,:,:].data
	LWA = LWAA[:,:,:] + LWAC[:,:,:]
	_LOG.info('variables loaded')

	qe = qdat[:,:,:]-Qref[:,:,np.newaxis]
	ue = udat[:,:,:]-Uref[:,:,np.newaxis]
	ve = vdat[:,:,:]
	te = tdat[:,:,:]-Tref[:,:,np.newaxis]
	_LOG.info('eddy variables calculated')

	times = np.linspace(0, max_lenghth, max_lenghth, endpoint=False)[:]
	dt = times[1]-times[0]
	dx = xs[1]-xs[0]
	dy = ys[1]-ys[0]

	LWAtend = lwatend(lwa=LWA, dt=dt)
	Urefadv = urefadv(lwa=LWA, uref=Uref, dx=dx, filt=False)
	ueqeadv = ueadv(q=qdat, qref=Qref, u=udat, uref=Uref, dx=dx, dy=dy, filt=False)
	EF_x = eddyflux_x(ue=ue, ve=ve, dx=dx, filt=False)
	EF_y = eddyflux_y(ue=ue, ve=ve, dy=dy, filt=False)
	EF_z = eddyflux_z(ve=ve, te=te, Ld=Ld, filt=False)
	EF = eddyflux(ve=ve, qe=qe, filt=False)

	RHS = Urefadv + ueqeadv + EF_x + EF_y + EF_z
	RES = LWAtend - RHS

	_LOG.info('budget calculated')

	import os
	os_system_rm: str = 'rm -f %s%s'%(savedir, sname)
	os.system(os_system_rm)
	with netCDF4.Dataset(savedir+sname,'w') as write:
		write.createDimension('time', size=len(times))
		write.createDimension('latitude', size=len(ys))
		write.createDimension('longitude', size=len(xs))

		time = write.createVariable('time','f4',dimensions=['time'])
		latitude = write.createVariable('latitude','f4',dimensions=['latitude'])
		longitude = write.createVariable('longitude','f4',dimensions=['longitude'])

		term1 = write.createVariable('lwatend','f4',dimensions=['time','latitude','longitude'])
		term2 = write.createVariable('urefadv','f4',dimensions=['time','latitude','longitude'])
		term3 = write.createVariable('ueqeadv','f4',dimensions=['time','latitude','longitude'])
		term4 = write.createVariable('ef_x','f4',dimensions=['time','latitude','longitude'])
		term5 = write.createVariable('ef_y','f4',dimensions=['time','latitude','longitude'])
		term6 = write.createVariable('ef_z','f4',dimensions=['time','latitude','longitude'])
		term7 = write.createVariable('res','f4',dimensions=['time','latitude','longitude'])

		longitude[:]=xs[:]
		latitude[:]=ys[:]
		time[:]=times

		term1[:,:,:]=LWAtend[:,:,:]
		term2[:,:,:]=Urefadv[:,:,:]
		term3[:,:,:]=ueqeadv[:,:,:]
		term4[:,:,:]=EF_x[:,:,:]
		term5[:,:,:]=EF_y[:,:,:]
		term6[:,:,:]=EF_z[:,:,:]
		term7[:,:,:]=RES[:,:,:]

	_LOG.info('output saved; done')


def cli(
    *,
    load_dir: Annotated[Path, typer.Option(help='Directory containing input NetCDF files')],
    save_dir: Annotated[Path, typer.Option(help='Directory to save output NetCDF files')],
) -> None:
	print('Computing LWA budget terms...')
	compute_budget(load_dir=load_dir, save_dir=save_dir)
	print('Done.')


if __name__ == '__main__':
	typer.run(cli)
