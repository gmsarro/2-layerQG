"""
Compute the latent heating contribution of the LWA budget and save to NetCDF.

Typical usage example:

```python
python LP_budget_make.py --load-dir /path/to/data --save-dir /path/to/output
```
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
from netCDF4 import Dataset
from lwabudget import LH


_LOG = logging.getLogger(__name__)


def compute_lp(load_dir: Path, save_dir: Path) -> None:
	loaddir = str(load_dir) if str(load_dir).endswith('/') else str(load_dir) + '/'
	savedir = str(save_dir) if str(save_dir).endswith('/') else str(save_dir) + '/'

	Llist = np.arange(0.0)
	Ulist = np.array([1.0], dtype=float)
	max_lenghth: Final[int] = 10000

	sname = 'LP_%s_2.0_0.1_%s.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2)))
	read = Dataset(loaddir+'N128_%s_2.0_0.1_%s.3d.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2))))
	qdat = read.variables['q1'][:,:,:].data
	pdat = read.variables['P'][:,:,:].data
	xs = read.variables['x'][:].data
	ys = read.variables['y'][:].data
	read.close()
	read = Dataset(loaddir+'N128_%s_2.0_0.1_%s.qref1_2.nc'%(str(np.round(Llist[0],2)),str(np.round(Ulist[0],2))))
	Qref = read.variables['qref1'][:,:].data
	read.close()
	_LOG.info('variables loaded')

	L = float(Llist[0])

	times = np.linspace(0, max_lenghth, max_lenghth, endpoint=False)[:]
	dt = times[1]-times[0]
	dx = xs[1]-xs[0]
	dy = ys[1]-ys[0]

	LP = LH(pdat, qdat, Qref, L, dx, dy, filt=False)
	_LOG.info('budget calculated')

	import os
	os.system('rm -f %s%s'%(savedir, sname))
	write = Dataset(savedir+sname,'w')
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

	write.close()
	_LOG.info('output saved; done')


def cli(load_dir: Path, save_dir: Path) -> None:
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
	)
	compute_lp(load_dir, save_dir)


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Compute latent heating contribution of LWA budget')
	parser.add_argument('--load-dir', type=Path, required=True)
	parser.add_argument('--save-dir', type=Path, required=True)
	args = parser.parse_args()
	cli(args.load_dir, args.save_dir)
