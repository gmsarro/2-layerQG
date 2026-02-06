# Compute Local Wave Activity and the eddy growth rate in 2-Layer QG Models

This repository contains scripts associated with the methods section of Sarro et al. (2025). The scripts are organized into two main categories:

1. **Local Wave Activity and Budget Calculation for a 2-Layer QG Model (with Latent Heating)**
2. **Eddy Growth Rates Calculation for a 2-Layer QG Model**

These scripts are primarily designed for the [Lutsko and Hell (2021) model](https://github.com/nicklutsko/moist_QG_channel/tree/main), but they can be adapted to other 2-layer QG models. For more complex datasets, such as reanalysis, consider using the [falwa package](https://github.com/csyhuang/hn2016_falwa) for the LWA budget calculation.

## Quickstart

### Prerequisites
- Python >= 3.10
- Fortran toolchain for LWA computation: `gfortran`, NetCDF-Fortran (`$NC_INC` and `$NC_LIB` should be set)

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run: LWA pipeline (three steps)

All three budget scripts accept a `--base-name` argument (e.g. `N128_0.0_2.0_0.1_1.0`) and derive input/output filenames from it.

```bash
# 1) Compute reference PV and (anti)cyclonic LWA via Fortran helpers
python local_wave_activity/make_lwa/calculate_lwa.py \
  --input-nc /path/to/N128_0.0_2.0_0.1_1.0.3d.nc \
  --work-dir /path/to/fortran_workspace \
  --output-prefix /path/to/output/N128_0.0_2.0_0.1_1.0.

# 2) Compute reference wind and temperature from qref
python local_wave_activity/make_lwa_budget/uref_make.py \
  --data-dir /path/to/output \
  --output-directory /path/to/output \
  --base-name N128_0.0_2.0_0.1_1.0

# 3a) Full LWA budget (all terms except latent heating)
python local_wave_activity/make_lwa_budget/lwa_budget_make.py \
  --data-dir /path/to/output \
  --output-directory /path/to/output \
  --base-name N128_0.0_2.0_0.1_1.0

# 3b) Latent heating contribution only
python local_wave_activity/make_lwa_budget/lp_budget_make.py \
  --data-dir /path/to/output \
  --output-directory /path/to/output \
  --base-name N128_0.0_2.0_0.1_1.0 \
  --latent-heating 0.0
```

### Run: Eddy growth (dry)
```bash
python eddy_growth/dry_eddy_growth/calculate_eddy_growth.py \
  --data-path /path/to/model.nc \
  --output-file /path/to/results.nc
```

### Run: Eddy growth (moist)
```python
import moist_growth_matrix

kk, growth, q1p, q2p, P = moist_growth_matrix.moist_matrix(L=0.2, U1=1.0, U2=0.5)
```

## Repository Structure

```plaintext
2-layerQG/
├── local_wave_activity/
│   ├── make_lwa/              # Fortran-based LWA and reference PV computation
│   └── make_lwa_budget/       # Reference wind, reference temperature, and LWA budget
├── eddy_growth/
│   ├── dry_eddy_growth/       # Dry eddy growth rate eigenvalue problem
│   └── moist_eddy_growth/     # Moist eddy growth rate (3x3 matrix)
├── requirements.txt
└── README.md
```

## Data Requirements
- Expected file pattern for LWA workflow: `<base_name>.3d.nc`, `<base_name>.nc`, `<base_name>.qref1_2.nc`, etc.
- Required variables (names used by scripts):
  - `q1`, `u1`, `v1`, `tau`, `P` (3D fields where applicable)
  - `zu1`, `zu2`, `ztau` (zonal means)
  - `x`, `y` coordinates (evenly spaced)
- Units/assumptions: nondimensional fields consistent with model settings; periodic in x; evenly spaced y.

## CLI Reference

- `local_wave_activity/make_lwa/calculate_lwa.py`
  - `--input-nc PATH`: model file with `q1`
  - `--work-dir PATH`: directory containing `rp2.f90` and `rp4.f90` and for build artifacts
  - `--output-prefix PATH`: prefix for output files, e.g., `/path/to/N128_0.0_2.0_0.1_1.0.`

- `local_wave_activity/make_lwa_budget/uref_make.py`
  - `--data-dir PATH`: directory containing `qref` and mean fields
  - `--output-directory PATH`: where to write `uref1_2.nc` and `tref1_2.nc`
  - `--base-name TEXT`: base filename (e.g. `N128_0.0_2.0_0.1_1.0`)
  - `--beta FLOAT` (default 0.2), `--ld FLOAT` (default 1.0), `--max-time INT` (default 10000)

- `local_wave_activity/make_lwa_budget/lwa_budget_make.py`
  - `--data-dir PATH`, `--output-directory PATH`, `--base-name TEXT`
  - `--max-time INT` (default 10000), `--ld FLOAT` (default 1.0)

- `local_wave_activity/make_lwa_budget/lp_budget_make.py`
  - `--data-dir PATH`, `--output-directory PATH`, `--base-name TEXT`
  - `--latent-heating FLOAT` (required): the L parameter
  - `--max-time INT` (default 10000)

- `eddy_growth/dry_eddy_growth/calculate_eddy_growth.py`
  - `--data-path PATH`: model NetCDF
  - `--output-file PATH`: output NetCDF

## Fortran Notes
- Ensure `gfortran` and NetCDF-Fortran are installed.
- Environment variables must be set before running the LWA step, e.g.:
```bash
export NC_INC="-I/usr/local/include"
export NC_LIB="-L/usr/local/lib -lnetcdff -lnetcdf"
```
- Place `rp2.f90` and `rp4.f90` in `--work-dir`.

## Execution Order

To compute the LWA budget, execute the scripts in the following order:

1. `local_wave_activity/make_lwa/calculate_lwa.py`
2. `local_wave_activity/make_lwa_budget/uref_make.py`
3. `local_wave_activity/make_lwa_budget/lp_budget_make.py`
   -or-
   `local_wave_activity/make_lwa_budget/lwa_budget_make.py`

## Contact Information

For any issues or questions, please contact me at [gmsarro@uchicago.edu](mailto:gmsarro@uchicago.edu).
