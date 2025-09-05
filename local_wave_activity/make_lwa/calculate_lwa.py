"""
Prepare input PV, run Fortran programs to compute reference PV and LWA, and move outputs.

Typical usage example:

```python
python Calculate_LWA.py \
  --input-nc /path/to/N128_L_C_E_U.3d.nc \
  --work-dir /path/to/fortran_workspace \
  --output-prefix /path/to/output/N128_L_C_E_U.
```

The script performs the following steps:
  1. Reads PV, removes sponge layer values by capping to zonal extrema
  2. Writes temporary `QGPV.nc` in the working directory
  3. Compiles and runs Fortran codes `rp2.f90` and `rp4.f90`
  4. Moves outputs to prefix with suffixes `wac1_2.nc`, `waa1_2.nc`, `qref1_2.nc`
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Final

import numpy as np
import xarray as xr


_LOG = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def prepare_qgpv(input_nc: Path, work_dir: Path) -> None:
	out_name: Final[str] = 'QGPV.nc'
	ds = xr.open_dataset(input_nc)
	ds_cropped = ds
	q1_mean = np.nanmean(ds_cropped['q1'], axis=2)
	location_max = np.argmax(q1_mean, axis=1)
	location_min = np.argmin(q1_mean, axis=1)
	value_max = np.max(q1_mean, axis=1)
	value_min = np.min(q1_mean, axis=1)
	for i in range(ds.sizes['time']):
		ds_cropped['q1'][i, location_max[i]:, :] = value_max[i]
		ds_cropped['q1'][i, :location_min[i], :] = value_min[i]
	# Save for Fortran
	work_dir_str = str(work_dir)
	out_path = os.path.join(work_dir_str, out_name)
	ds_cropped = xr.Dataset({'q1': ds_cropped['q1']})
	ds_cropped.to_netcdf(out_path)
	ds.close()
	ds_cropped.close()
	os.chmod(out_path, 0o744)
	_LOG.info('Prepared %s', out_path)


def run_fortran(work_dir: Path) -> None:
	cwd = os.getcwd()
	try:
		os.chdir(work_dir)
		os.system('gfortran -c -g -fcheck=all $NC_INC rp2.f90 -o rp2.o')
		os.system('gfortran -o rp2 rp2.o $NC_LIB')
		os.system('./rp2')
		os.system('gfortran -c -g -fcheck=all $NC_INC rp4.f90 -o rp4.o')
		os.system('gfortran -o rp4 rp4.o $NC_LIB')
		os.system('./rp4')
	finally:
		os.chdir(cwd)
	_LOG.info('Fortran runs completed')


def move_outputs(work_dir: Path, output_prefix: Path) -> None:
	work_dir_str = str(work_dir)
	prefix = str(output_prefix)
	os.system('mv %s/wac1.nc %swac1_2.nc' % (work_dir_str, prefix))
	os.system('mv %s/waa1.nc %swaa1_2.nc' % (work_dir_str, prefix))
	os.system('mv %s/qref1.nc %sqref1_2.nc' % (work_dir_str, prefix))
	_LOG.info('Moved outputs to prefix %s*', prefix)


def cleanup_workdir(work_dir: Path) -> None:
	for name in ('rp2.o','rp4.o','rp2','rp4','QGPV.nc'):
		path = work_dir / name
		if path.exists():
			os.remove(path)
	_LOG.info('Cleaned working directory %s', work_dir)


def main(input_nc: Path, work_dir: Path, output_prefix: Path) -> None:
	_ensure_dir(work_dir)
	_ensure_dir(output_prefix.parent)
	prepare_qgpv(input_nc, work_dir)
	run_fortran(work_dir)
	move_outputs(work_dir, output_prefix)
	cleanup_workdir(work_dir)


def cli(input_nc: Path, work_dir: Path, output_prefix: Path) -> None:
	"""CLI entry point to compute LWA and reference PV via Fortran helpers."""
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S',
	)
	main(input_nc, work_dir, output_prefix)


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Run Fortran LWA computations')
	parser.add_argument('--input-nc', type=Path, required=True, help='Path to model NetCDF file with q1')
	parser.add_argument('--work-dir', type=Path, required=True, help='Directory containing rp2.f90 and rp4.f90; will write QGPV.nc and build artifacts here')
	parser.add_argument('--output-prefix', type=Path, required=True, help='Prefix for output files, e.g., /path/to/N128_L_C_E_U.')
	args = parser.parse_args()
	cli(args.input_nc, args.work_dir, args.output_prefix)

