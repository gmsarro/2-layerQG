"""
Calculate U_REF and T_REF from Q_REF; all variables are nondimensionalized.
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


_LOG = logging.getLogger(__name__)


def solve_uref(
	*,
	qref: np.ndarray,
	um: np.ndarray,
	umb: np.ndarray,
	ys: np.ndarray,
	beta: float,
	Ld: float,
	maxerr: float = 1e-6,
	maxIT: int = 10000,
	relax: float = 1.9,
) -> np.ndarray:
	"""Solve SOR for U_REF given Q_REF gradient and boundary conditions.

	:param qref: Reference PV (time, latitude)
	:param um: Zonal-mean U (upper) for initial guess and boundary
	:param umb: Zonal-mean U (lower) used in forcing term
	:param ys: Latitude coordinate (monotonic, evenly spaced)
	:param beta: Nondimensional beta parameter
	:param Ld: Deformation radius
	:param maxerr: Convergence tolerance
	:param maxIT: Maximum iterations
	:param relax: Relaxation factor for SOR
	:return: Computed U_REF (time, latitude)
	"""
	tn, yn = np.shape(qref)
	dy = ys[1] - ys[0]
	AC = np.array([1/dy**2, -2/dy**2, 1/dy**2])
	qref_y = np.zeros((tn, yn))
	qref_y[:,1:-1] = (qref[:,2:] - qref[:,:-2])/(2*dy)
	uref = np.zeros((tn, yn)) + um[:,:]
	nIT = 0
	err = 1e5
	while nIT < maxIT and err > maxerr:
		utemp = np.zeros((tn, yn))
		utemp[:,:] = uref[:,:]
		for y in range(1, yn-1):
			RS = (
				AC[0]*uref[:,y-1] + AC[1]*uref[:,y] + AC[2]*uref[:,y+1]
			) - beta + qref_y[:,y] - uref[:,y]/Ld**2 + umb[:,y]/Ld**2
			uref[:,y] = uref[:,y] - relax*RS/(AC[1] - 1/Ld**2)
		err = np.max(np.abs(uref[:,:] - utemp[:,:]))
		nIT += 1
	if nIT == maxIT:
		_LOG.info('Not fully converged')
	else:
		_LOG.info('Converged at %s iterations', nIT)
	return uref


def integrate_tref(
	*,
	uref: np.ndarray,
	ym: np.ndarray,
	tm: np.ndarray,
) -> np.ndarray:
	"""Integrate for T_REF from U_REF shear and adjust mean to match boundary template.

	:param uref: Reference zonal wind (time, latitude)
	:param ym: Latitude coordinate (used for spacing)
	:param tm: Zonal-mean temperature (time, latitude) used for offset constraint
	:return: Reference temperature (time, latitude)
	"""
	tn, yn = np.shape(uref)
	dy = ym[1] - ym[0]
	tref = np.zeros((tn, yn))
	ushear = (uref[:,1:] + uref[:,:-1]) * 0.5
	for y in range(yn-1):
		tref[:,y+1] = tref[:,0] - np.sum(ushear[:,:y+1]*dy, axis=1)
	for t in range(tn):
		offset = np.mean(tm[t,:] - tref[t,:])
		tref[t,:] += offset
	return tref


def compute_and_save(
	*,
	data_dir: Path,
	save_dir: Path,
	Llist: np.ndarray,
	Ulist: np.ndarray,
	beta: float,
	Ld: float,
) -> None:
	data_dir_s = str(data_dir) if str(data_dir).endswith('/') else str(data_dir) + '/'
	save_dir_s = str(save_dir) if str(save_dir).endswith('/') else str(save_dir) + '/'
	sname_u = 'N128_%s_2.0_0.1_%s.uref1_2.nc' % (str(np.round(Llist[0],2)), str(np.round(Ulist[0],2)))
	sname_t = 'N128_%s_2.0_0.1_%s.tref1_2.nc' % (str(np.round(Llist[0],2)), str(np.round(Ulist[0],2)))

	with netCDF4.Dataset(data_dir_s+'N128_%s_2.0_0.1_%s.qref1_2.nc' % (str(np.round(Llist[0],2)), str(np.round(Ulist[0],2)))) as readq:
		qref = readq.variables['qref1'][:,:].data
		tn, yn = np.shape(qref)
	with netCDF4.Dataset(data_dir_s+'N128_%s_2.0_0.1_%s.nc' % (str(np.round(Llist[0],2)), str(np.round(Ulist[0],2)))) as readu:
		um = readu.variables['zu1'][:,:].data
		umb = readu.variables['zu2'][:,:].data
		tm = readu.variables['ztau'][:,:].data
		ys = readu.variables['y'][:].data

	uref = solve_uref(qref=qref, um=um, umb=umb, ys=ys, beta=beta, Ld=Ld)
	tref = integrate_tref(uref=uref, ym=ys, tm=tm)

	os.system('rm -f %s' % (save_dir_s + sname_u))
	with netCDF4.Dataset(save_dir_s + sname_u,'w') as write:
		write.createDimension('time', size=tn)
		write.createDimension('latitude', size=yn)
		var_u = write.createVariable('uref1','f4', dimensions=['time','latitude'])
		time = write.createVariable('time','f4', dimensions=['time'])
		latitude = write.createVariable('y','f4', dimensions=['latitude'])
		var_u[:,:] = uref[:,:]
		latitude[:] = ys[:]
		time[:] = np.arange(tn)
	_LOG.info('Saved %s', save_dir_s + sname_u)

	os.system('rm -f %s' % (save_dir_s + sname_t))
	with netCDF4.Dataset(save_dir_s + sname_t,'w') as write:
		write.createDimension('time', size=tn)
		write.createDimension('latitude', size=yn)
		var_t = write.createVariable('tref1','f4', dimensions=['time','latitude'])
		time = write.createVariable('time','f4', dimensions=['time'])
		latitude = write.createVariable('y','f4', dimensions=['latitude'])
		var_t[:,:] = tref[:,:]
		latitude[:] = ys[:]
		time[:] = np.arange(tn)
	_LOG.info('Saved %s', save_dir_s + sname_t)


def cli(
	*,
	data_dir: Annotated[Path, typer.Option(help='Input directory containing qref and mean fields')],
	save_dir: Annotated[Path, typer.Option(help='Output directory')],
	beta: Annotated[float, typer.Option(help='Nondimensional beta')] = 0.2,
	Ld: Annotated[float, typer.Option(help='Deformation radius')] = 1.0,
) -> None:
	print('Computing U_REF and T_REF...')
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
	)
	Llist = np.arange(0.0)
	Ulist = np.array([1.0], dtype=float)
	compute_and_save(data_dir=data_dir, save_dir=save_dir, Llist=Llist, Ulist=Ulist, beta=beta, Ld=Ld)
	print('Done.')


if __name__ == '__main__':
	typer.run(cli)

