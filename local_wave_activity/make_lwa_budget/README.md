# LWA Budget Scripts

Compute the LWA budget and latent heating contribution, saving NetCDF outputs.

Usage examples:
```bash
python lwa_budget_make.py --load-dir /path/to/data --save-dir /path/to/output
python lp_budget_make.py --load-dir /path/to/data --save-dir /path/to/output
```

Bazel:
```bash
bazel run //local_wave_activity/make_lwa_budget:lwa_budget_make -- \
  --load-dir /path/to/data --save-dir /path/to/output
bazel run //local_wave_activity/make_lwa_budget:lp_budget_make -- \
  --load-dir /path/to/data --save-dir /path/to/output
``` 
