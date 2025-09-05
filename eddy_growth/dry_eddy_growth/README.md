# Dry Eddy Growth (2-layer QG)

Compute eddy growth rates and mode structures by solving a generalized eigenvalue problem.

Typical usage example:
```bash
python Calculate_Eddy_Growth.py --data-path /path/to/model.nc --output-file /path/to/results.nc
```

Bazel targets:
```bash
bazel run //Eddy_Growth/dry_eddy_growth:calculate_eddy_growth -- \
  --data-path /path/to/model.nc \
  --output-file /path/to/results.nc
``` 