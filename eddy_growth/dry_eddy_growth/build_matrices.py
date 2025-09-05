"""
Build coefficient matrices for the generalized eigenvalue problem of a 2-layer QG model.

Typical usage example:

    # Build matrices for given inputs
    M, N = build_matrices(u1, u2, beta, dy, n_2, rk, half_matrix, n)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def build_matrices(
    u1: NDArray[np.floating],
    u2: NDArray[np.floating],
    beta: float,
    dy: float,
    n_2: int,
    rk: float,
    half_maxtrix: int,
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Construct coefficient matrices M and N for the 2-layer QG eigenproblem.

    :param u1: Upper-layer mean zonal wind array along meridional grid
    :param u2: Lower-layer mean zonal wind array along meridional grid
    :param beta: Planetary vorticity gradient (nondimensional)
    :param dy: Meridional grid spacing
    :param n_2: Total system size (2 * number_of_meridional_points)
    :param rk: Zonal wavenumber (nondimensional)
    :param half_maxtrix: Half matrix size (n - 2) used for block partitioning
    :param n: Number of meridional grid points
    :return: Tuple (M, N) of matrices for the generalized eigenvalue problem M v = c N v
    """
    M: NDArray[np.floating] = np.zeros((n_2-4, n_2-4))
    N: NDArray[np.floating] = np.zeros((n_2-4, n_2-4))
    
    for j in range(n-4):
        M[j,j] = -u1[j+1]*(rk**2*dy**2+2.+dy**2)
        M[j,j] += (beta*dy**2-(u1[j+2]+u1[j]-2.*u1[j+1]))
        M[j,j] += (u1[j+1]-u2[j+1])*dy**2
        M[j+half_maxtrix,j+half_maxtrix] = -u2[j+1]*(rk**2*dy**2+2.+dy**2)
        M[j+half_maxtrix,j+half_maxtrix] += (beta*dy**2-(u2[j+2]+u2[j]-2.*u2[j+1]))
        M[j+half_maxtrix,j+half_maxtrix] += -(u1[j+1]-u2[j+1])*dy**2
        M[j,j+half_maxtrix]=u1[j+1]*dy**2
        M[j+half_maxtrix,j]=u2[j+1]*dy**2
        M[j,j+1] = u1[j+1]
        M[j+1,j] = u1[j+2]
        M[j+half_maxtrix,j+half_maxtrix+1] = u2[j+1]
        M[j+half_maxtrix+1,j+half_maxtrix] = u2[j+2]
        
        N[j,j+1] = 1.
        N[j+1,j] = 1.
        N[j+half_maxtrix,j+half_maxtrix+1] = 1.
        N[j+half_maxtrix+1,j+half_maxtrix] = 1.
        N[j,j] = -(rk**2*dy**2+2.+dy**2)
        N[j+half_maxtrix,j+half_maxtrix] = -(rk**2*dy**2+2.+dy**2)
        N[j,j+half_maxtrix]=dy**2
        N[j+half_maxtrix,j]=dy**2
        
    jo = n-3    
    M[jo,jo] = -u1[jo+1]*(rk**2*dy**2+2.+dy**2)
    M[jo,jo] += (beta*dy**2-(u1[jo+2]+u1[jo]-2.*u1[jo+1]))
    M[jo,jo] += (u1[jo+1]-u2[jo+1])*dy**2
    M[jo+half_maxtrix,jo+half_maxtrix] = -u2[jo+1]*(rk**2*dy**2+2.+dy**2)
    M[jo+half_maxtrix,jo+half_maxtrix] += (beta*dy**2-(u2[jo+2]+u2[jo]-2.*u2[jo+1]))
    M[jo+half_maxtrix,jo+half_maxtrix] += -(u1[jo+1]-u2[jo+1])*dy**2
    M[jo,jo+half_maxtrix]=u1[jo+1]*dy**2
    M[jo+half_maxtrix,jo]=u2[jo+1]*dy**2
    
    N[jo,jo] = -(rk**2*dy**2+2.+dy**2)
    N[jo+half_maxtrix,jo+half_maxtrix] = -(rk**2*dy**2+2.+dy**2)
    N[jo,jo+half_maxtrix]=dy**2
    N[jo+half_maxtrix,jo]=dy**2

    return M, N

