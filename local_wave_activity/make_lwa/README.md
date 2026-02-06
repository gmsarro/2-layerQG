# Make LWA

Preprocess PV to remove sponge, then compile and run Fortran codes to compute reference PV and (anti)cyclonic LWA.

## Usage

```bash
python calculate_lwa.py \
  --input-nc /path/to/N128_L_C_E_U.3d.nc \
  --work-dir /path/to/fortran_workspace \
  --output-prefix /path/to/output/N128_L_C_E_U.
```

Notes:
- Requires `gfortran` and NetCDF-Fortran libraries. Environment variables `$NC_INC` and `$NC_LIB` should be set appropriately.
- `rp2.f90` and `rp4.f90` must be present in `--work-dir`.
