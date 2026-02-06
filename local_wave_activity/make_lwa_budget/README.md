# LWA Budget Scripts

Compute the LWA budget and latent heating contribution, saving NetCDF outputs.

## Usage

```bash
python uref_make.py \
  --data-dir /path/to/data \
  --output-directory /path/to/output \
  --base-name N128_0.0_2.0_0.1_1.0

python lwa_budget_make.py \
  --data-dir /path/to/data \
  --output-directory /path/to/output \
  --base-name N128_0.0_2.0_0.1_1.0

python lp_budget_make.py \
  --data-dir /path/to/data \
  --output-directory /path/to/output \
  --base-name N128_0.0_2.0_0.1_1.0 \
  --latent-heating 0.15
```
