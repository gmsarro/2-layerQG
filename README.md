# Two-Layer QG Tools

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typed-mypy--strict-blue.svg)](https://mypy-lang.org/)

Compute **Local Wave Activity (LWA) budgets** and **eddy growth rates** for 2-layer quasi-geostrophic channel models. Associated with the methods described in [Sarro et al. (2026)](#citation).

## Installation

```bash
pip install git+https://github.com/gmsarro/2-layerQG.git
```

Or from a local clone:

```bash
git clone https://github.com/gmsarro/2-layerQG.git
cd 2-layerQG
pip install -e .
```

### What you get

Two importable Python packages and five CLI commands:

| Package | Import | CLI Commands |
|---------|--------|--------------|
| **qg_lwa** | `import qg_lwa` | `qg-compute-lwa`, `qg-compute-uref`, `qg-compute-budget`, `qg-compute-lp` |
| **qg_eddy_growth** | `import qg_eddy_growth` | `qg-compute-growth` |

## Quick Start

### LWA Budget Pipeline

The LWA budget is computed in four sequential steps. All scripts are
**resolution-independent** -- grid dimensions are read from the input NetCDF
file, and the domain extent (`--lx`, `--ly`) is configurable.

```bash
# Step 1: Compute reference PV and LWA (Fortran compiled automatically on first run)
qg-compute-lwa \
  --input-nc /path/to/model.3d.nc \
  --output-directory /path/to/output \
  --lx 48.0 --ly 72.0

# Step 2: Compute reference wind (U_ref) and temperature (T_ref)
qg-compute-uref \
  --data-dir /path/to/output \
  --output-directory /path/to/output \
  --base-name model \
  --beta 0.2 --ld 1.0

# Step 3: Compute full LWA budget (all terms except latent heating)
qg-compute-budget \
  --data-dir /path/to/output \
  --output-directory /path/to/output \
  --base-name model --ld 1.0

# Step 4: Compute latent heating contribution
qg-compute-lp \
  --data-dir /path/to/output \
  --output-directory /path/to/output \
  --base-name model \
  --latent-heating 0.15
```

### Eddy Growth Rates

**Dry growth** (CLI):

```bash
qg-compute-growth \
  --data-path /path/to/model.3d.nc \
  --output-file /path/to/growth_rates.nc \
  --beta 0.2 --resolution 600 --max-wavenumber 3.0 \
  --sponge-min 31 --sponge-max -32
```

**Moist growth** (Python API):

```python
import qg_eddy_growth.moist_growth

kk, growth, q1_prime, q2_prime, P = qg_eddy_growth.moist_growth.moist_matrix(
    L=0.2, U1=1.0, U2=0.5,
)
```

## Python API

Both packages expose their functions for direct use in scripts and notebooks:

```python
import qg_lwa.budget
import qg_eddy_growth.matrices

# Compute LWA time tendency
tendency = qg_lwa.budget.lwatend(lwa=lwa_field, dt=1.0)

# Build eigenvalue matrices for dry eddy growth
M, N = qg_eddy_growth.matrices.build_matrices(
    u1=u1, u2=u2, beta=0.2, dy=dy,
    n_2=n_2, rk=1.0, half_matrix=hm, n=n,
)
```

## Configurable Parameters

All physical and numerical parameters are exposed as CLI options:

| Parameter | Flag | Default | Scripts |
|-----------|------|---------|---------|
| Domain extent (zonal) | `--lx` | 72.0 | `qg-compute-lwa` |
| Domain extent (meridional) | `--ly` | 96.0 | `qg-compute-lwa` |
| Beta | `--beta` | 0.2 | `qg-compute-uref`, `qg-compute-growth` |
| Deformation radius | `--ld` | 1.0 | `qg-compute-uref`, `qg-compute-budget` |
| Latent heating (L) | `--latent-heating` | *required* | `qg-compute-lp` |
| Max timesteps | `--max-time` | all | all LWA scripts |
| Wavenumber resolution | `--resolution` | 600 | `qg-compute-growth` |
| Max wavenumber | `--max-wavenumber` | 3.0 | `qg-compute-growth` |
| Sponge layer bounds | `--sponge-min/max` | 31 / -32 | `qg-compute-growth` |

## Repository Structure

```
2-layerQG/
├── pyproject.toml              # Package configuration (pip install .)
├── qg_lwa/                     # Local Wave Activity package
│   ├── __init__.py
│   ├── budget.py               #   LWA budget functions
│   ├── array_utils.py          #   NetCDF array orientation helpers
│   ├── compute_lwa.py          #   CLI: f2py-based LWA computation (any resolution)
│   ├── compute_uref.py         #   CLI: Reference wind / temperature
│   ├── compute_budget.py       #   CLI: Full LWA budget
│   ├── compute_lp.py           #   CLI: Latent heating budget term
│   ├── py.typed                #   PEP 561 type marker
│   └── fortran/
│       └── lwa_2layer.f90      #   Fortran module (auto-compiled via f2py)
├── qg_eddy_growth/             # Eddy Growth Rate package
│   ├── __init__.py
│   ├── matrices.py             #   Coefficient matrix construction
│   ├── dry_growth.py           #   CLI: Dry eddy growth eigenvalue solver
│   ├── moist_growth.py         #   Moist eddy growth (3x3 matrix)
│   └── py.typed                #   PEP 561 type marker
├── notebooks/                  # Example Jupyter notebooks
│   ├── plot_eddy_growth.ipynb
│   └── plot_moist_growth.ipynb
├── README.md
├── LICENSE
└── .gitignore
```

## Requirements

### Python

- Python >= 3.9
- Dependencies are installed automatically via `pip install .`

### Fortran (LWA step only)

The `qg-compute-lwa` step uses an f2py-wrapped Fortran module that is
**compiled automatically** on first use. This requires:

- `gfortran` (any recent version)
- **No NetCDF-Fortran needed** -- the Fortran module is purely computational; all I/O is handled by Python

## Model Compatibility

These tools are designed for the
[Lutsko and Hell (2021) moist QG channel model](https://github.com/nicklutsko/moist_QG_channel)
but can be adapted to any 2-layer QG model that writes NetCDF output with the
variables listed below. For reanalysis data, consider
the [falwa package](https://github.com/csyhuang/hn2016_falwa).

### Expected NetCDF Variables

| Variable | Description | Dimensions |
|----------|-------------|------------|
| `q1` | Upper-layer QGPV | (time, y, x) |
| `u1`, `v1` | Upper-layer winds | (time, y, x) |
| `u2` | Lower-layer zonal wind | (time, y, x) |
| `tau` | Temperature | (time, y, x) |
| `P` | Precipitation | (time, y, x) |
| `zu1`, `zu2`, `ztau` | Zonal means | (time, y) |
| `x`, `y` | Coordinates | (x,), (y,) |

## Citation

If you use this code, please cite:

> Sarro, G., J. Kang, S. Smith, A. Chaudhri, and N. Nakamura, 2026: Non-monotonic response of blocking dynamics with increased latent heating in an Idealized 2-Layer QG model. *In Revision, Journal of the Atmospheric Sciences*.

## License

[Apache License 2.0](LICENSE)

## Contact

Giorgio M. Sarro -- [gmsarro@uchicago.edu](mailto:gmsarro@uchicago.edu)
