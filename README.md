# Compute Local Wave Activity and the eddy growth rate in 2-Layer QG Models

This repository contains scripts associated with the methods section of Sarro et al. (2024). The scripts are organized into two main categories:

1. **Local Wave Activity and Budget Calculation for a 2-Layer QG Model (with Latent Heating)**
2. **Eddy Growth Rates Calculation for a 2-Layer QG Model**

These scripts are primarily designed for the [Lutsko and Hell (2021) model](https://github.com/nicklutsko/moist_QG_channel/tree/main), but they can be adapted to other 2-layer QG models. For more complex datasets, such as reanalysis, consider using the [falwa package](https://github.com/csyhuang/hn2016_falwa) for the LWA budget calculation.

## Repository Structure

```plaintext
2-LayerQG/
├── Local_Wave_Activity/
│   ├── Make_LWA/              # Scripts for local wave activity calculation and reference PV
│   ├── Make_LWA_budget/       # Scripts for reference wind, reference temperature, and LWA budget calculation
│
└── Eddy_Growth/               # Scripts for eddy growth rates calculation
```

## Scripts Overview

### 1. Local Wave Activity and Budget Calculation

This section includes scripts for calculating the local wave activity (LWA) and its budget within a 2-layer quasi-geostrophic (QG) model, considering latent heating effects. These scripts correspond to the results presented in "The Local Wave Activity Budget in the 2-Layer Model" of Sarro et al. (2025).

**Subdirectories:**

- **Make_LWA/**: Contains scripts for calculating local wave activity and reference potential vorticity (PV).
- **Make_LWA_budget/**: Includes scripts for calculating reference wind, reference temperature, and the LWA budget.

**Execution Order:**

To compute the LWA budget, execute the scripts in the following order:

1. `2-LayerQG/Local_Wave_Activity/Make_LWA/Calculate_LWA.py`
2. `2-LayerQG/Local_Wave_Activity/Make_LWA_budget/UREF_make.py`
3. `2-LayerQG/Local_Wave_Activity/Make_LWA_budget/LP_budget_make.py` 
   -or- 
   `2-LayerQG/Local_Wave_Activity/Make_LWA_budget/LWA_budget_make.py`

### 2. Eddy Growth Rates Calculation

This section includes scripts for calculating eddy growth rates for the 2-layer QG model, only using dry and by including moisture assuming saturation and a temperature-dependent saturation. The script also computes the spatial structure of the waves in the most unstable mode. These scripts correspond to the analysis discussed in "Fastest Growth Rate Calculations" section of Sarro et al. (2025).

## Contact Information

For any issues or questions, please contact me at [gmsarro@uchicago.edu](mailto:gmsarro@uchicago.edu).
