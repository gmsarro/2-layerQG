# Compute Local Wave Activity and the eddy growth rate in 2-Layer QG Models

This repository contains scripts associated with the methods section of Sarro et al. (2025). The scripts are organized into two main categories:

1. **Local Wave Activity and Budget Calculation for a 2-Layer QG Model (with Latent Heating)**
2. **Eddy Growth Rates Calculation for a 2-Layer QG Model**

These scripts are primarily designed for the [Lutsko and Hell (2021) model](https://github.com/nicklutsko/moist_QG_channel/tree/main), but they can be adapted to other 2-layer QG models. For more complex datasets, such as reanalysis, consider using the [falwa package](https://github.com/csyhuang/hn2016_falwa) for the LWA budget calculation.

## Quickstart

### Prerequisites
- Python ≥ 3.10
- Optional: Bazel (if using the provided BUILD files)
- Fortran toolchain for LWA computation: `gfortran`, NetCDF-Fortran (`$NC_INC` and `$NC_LIB` should be set)

### Install (Python only)
```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy netCDF4 xarray
```

### Run: LWA pipeline (three steps)
```bash
# 1) Compute reference PV and (anti)cyclonic LWA via Fortran helpers
python local_wave_activity/make_lwa/calculate_lwa.py \
  --input-nc /path/to/N128_L_C_E_U.3d.nc \
  --work-dir /path/to/fortran_workspace \
  --output-prefix /path/to/output/N128_L_C_E_U.

# 2) Compute reference wind and temperature from qref
python local_wave_activity/make_lwa_budget/uref_make.py \
  --data-dir /path/to/output \
  --save-dir /path/to/output

# 3) Choose either latent heating-only term or full budget
python local_wave_activity/make_lwa_budget/lp_budget_make.py \
  --load-dir /path/to/output \
  --save-dir /path/to/output
# or
python local_wave_activity/make_lwa_budget/lwa_budget_make.py \
  --load-dir /path/to/output \
  --save-dir /path/to/output
```

### Run: Eddy growth (dry)
```bash
python eddy_growth/dry_eddy_growth/calculate_eddy_growth.py \
  --data-path /path/to/model.nc \
  --output-file /path/to/results.nc
```

## Repository Structure

```plaintext
2-LayerQG/
├── local_wave_activity/
│   ├── make_lwa/              # Scripts for local wave activity calculation and reference PV
│   ├── make_lwa_budget/       # Scripts for reference wind, reference temperature, and LWA budget calculation
│
└── eddy_growth/               # Scripts for eddy growth rates calculation
```

## Data Requirements
- Expected file pattern for LWA workflow: `N128_<L>_2.0_0.1_<U>*.nc`
- Required variables (names used by scripts):
  - `q1`, `u1`, `v1`, `tau` (3D fields where applicable)
  - `x`, `y` coordinates (evenly spaced)
- Units/assumptions: nondimensional fields consistent with model settings; periodic in x; evenly spaced y.

## CLI Reference

- `local_wave_activity/make_lwa/calculate_lwa.py`
  - `--input-nc PATH`: model file with `q1`
  - `--work-dir PATH`: directory containing `rp2.f90` and `rp4.f90` and for build artifacts
  - `--output-prefix PATH`: output prefix, e.g., `/out/N128_L_C_E_U.`

- `local_wave_activity/make_lwa_budget/uref_make.py`
  - `--data-dir PATH`: directory containing `qref` and mean fields
  - `--save-dir PATH`: where to write `uref1_2.nc` and `tref1_2.nc`
  - `--beta FLOAT` (default 0.2), `--Ld FLOAT` (default 1.0)

- `local_wave_activity/make_lwa_budget/lp_budget_make.py`
  - `--load-dir PATH`, `--save-dir PATH`

- `local_wave_activity/make_lwa_budget/lwa_budget_make.py`
  - `--load-dir PATH`, `--save-dir PATH`

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

## Reproducibility
- Use the three-step LWA workflow above in order to reproduce LWA budgets.
- The dry eddy growth script reproduces growth-rate spectra and mode structures; provide the model `u1`, `u2`, `y` fields in the input file.

## Scripts Overview

### 1. Local Wave Activity and Budget Calculation

This section includes scripts for calculating the local wave activity (LWA) and its budget within a 2-layer quasi-geostrophic (QG) model, considering latent heating effects. These scripts correspond to the results presented in "The Local Wave Activity Budget in the 2-Layer Model" of Sarro et al. (2025).

**Subdirectories:**

- **make_lwa/**: Contains scripts for calculating local wave activity and reference potential vorticity (PV).
- **make_lwa_budget/**: Includes scripts for calculating reference wind, reference temperature, and the LWA budget.

**Execution Order:**

To compute the LWA budget, execute the scripts in the following order:

1. `2-LayerQG/local_wave_activity/make_lwa/calculate_lwa.py`
2. `2-LayerQG/local_wave_activity/make_lwa_budget/uref_make.py`
3. `2-LayerQG/local_wave_activity/make_lwa_budget/lp_budget_make.py` 
   -or- 
   `2-LayerQG/local_wave_activity/make_lwa_budget/lwa_budget_make.py`

### 2. Eddy Growth Rates Calculation

This section includes scripts for calculating eddy growth rates for the 2-layer QG model, only using dry and by including moisture assuming saturation and a temperature-dependent saturation. The script also computes the spatial structure of the waves in the most unstable mode. These scripts correspond to the analysis discussed in "Fastest Growth Rate Calculations" section of Sarro et al. (2025).

## Contact Information

For any issues or questions, please contact me at [gmsarro@uchicago.edu](mailto:gmsarro@uchicago.edu).
