# LWA Budget Scripts

Compute the LWA budget and latent heating contribution, saving NetCDF outputs.

Usage examples:
```bash
python LWA_budget_make.py --load-dir /path/to/data --save-dir /path/to/output
python LP_budget_make.py --load-dir /path/to/data --save-dir /path/to/output
```

Bazel:
```bash
bazel run //Local_Wave_Activity/Make_LWA_budget:lwa_budget_make -- \
  --load-dir /path/to/data --save-dir /path/to/output
bazel run //Local_Wave_Activity/Make_LWA_budget:lp_budget_make -- \
  --load-dir /path/to/data --save-dir /path/to/output
``` 