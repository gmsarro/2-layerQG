"""Eddy growth rate computation for 2-layer quasi-geostrophic models.

Provides tools for computing dry and moist eddy growth rates via
generalized eigenvalue problems.

Submodules
----------
matrices     -- Coefficient matrix construction for the dry eigenproblem
dry_growth   -- CLI: Dry eddy growth rate solver (wavenumber sweep)
moist_growth -- Moist eddy growth rate (3x3 complex matrix)
"""

__version__ = "1.0.0"
