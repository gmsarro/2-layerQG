"""Local Wave Activity budget computation for 2-layer quasi-geostrophic models.

Provides functions and CLI tools for computing reference PV, LWA,
reference wind/temperature, and all terms of the LWA budget
(including latent heating).

Submodules
----------
budget        -- LWA budget helper functions (lwatend, urefadv, ueadv, etc.)
array_utils   -- NetCDF array orientation coercion utilities
compute_lwa   -- CLI: Fortran-based LWA and reference PV computation
compute_uref  -- CLI: Reference wind (U_ref) and temperature (T_ref)
compute_budget -- CLI: Full LWA budget (all terms except latent heating)
compute_lp    -- CLI: Latent heating budget term
"""

__version__ = "1.0.0"
