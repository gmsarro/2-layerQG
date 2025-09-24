"""
Compute fastest growing moist modes in a 2-layer QG model, and return growth and PV structure.
"""

from __future__ import annotations

import logging
from typing import Final, Tuple

import numpy as np
import numpy.typing
import scipy.linalg


_LOG = logging.getLogger(__name__)


def laplacian(
    *,
    f: numpy.typing.NDArray[np.floating],
    dx: float,
) -> numpy.typing.NDArray[np.floating]:
    return (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / dx**2


def gradient(
    *,
    f: numpy.typing.NDArray[np.floating],
    dx: float,
) -> numpy.typing.NDArray[np.floating]:
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)


def moist_matrix(
    *,
    L: float,
    U1: float,
    U2: float,
) -> Tuple[
    numpy.typing.NDArray[np.floating],
    numpy.typing.NDArray[np.floating],
    numpy.typing.NDArray[np.floating],
    numpy.typing.NDArray[np.floating],
    numpy.typing.NDArray[np.floating],
]:
    delta: Final[float] = 1.0
    el: Final[float] = 0.0
    ep: Final[float] = 0.5
    dk: Final[float] = 4.0/400.0
    D: float = U1 - U2
    B: Final[float] = 0.2
    C: Final[float] = 2.0
    R: Final[float] = 0.0
    G: float = B - (D + C * L) / (delta * (1.0 - L))
    E: float = 0.5 * L / (1.0 - L)

    aa: numpy.typing.NDArray[np.complexfloating] = np.zeros((3,3), dtype=complex)
    bb: numpy.typing.NDArray[np.complexfloating] = np.zeros((3,3), dtype=complex)
    growth: numpy.typing.NDArray[np.floating] = np.zeros(400)
    kk: numpy.typing.NDArray[np.floating] = np.zeros(400)

    m: int = 1
    while m < 400:
        rk: float = dk * m
        aa[0,0] = -U1*(rk*rk+el*el+1./(2.-delta))+(B+D/(2.-delta))+ep*(0.+1.j)*L*C/rk
        aa[0,1] = U1/(2.-delta)-ep*(0.+1.j)*L*C/rk
        aa[0,2] = -ep*(0.+1.j)*L/rk
        bb[0,0] = -(rk*rk+el*el+1./(2.-delta))
        bb[0,1] = 1./(2.-delta)
        bb[0,2] = 0.
        aa[1,0] = U2/delta-ep*(0.+1.j)*L*C/rk
        aa[1,1] = -U2*(rk*rk+el*el+1./delta)+(B-D/delta)+(0.+1.j)*R*(rk*rk+el*el)/rk+ep*(0.+1.j)*L*C/rk
        aa[1,2] = ep*(0.+1.j)*L/rk
        bb[1,0] = 1./delta
        bb[1,1] = -(rk*rk+el*el+1./delta)
        bb[1,2] = 0.
        aa[2,0] = U2+(0.+1.j)*(1.-L)*C*ep/rk
        aa[2,1] = -U2-(C+D)-(0.+1.j)*(1.-L)*C*ep/rk
        aa[2,2] = U2-(0.+1.j)*(1.-L)*ep/rk
        bb[2,0] = 1.
        bb[2,1] = -1.
        bb[2,2] = 1.
    
        evals, V = scipy.linalg.eig(aa, bb)
        gr = evals.imag*rk
        growth[m] = np.max(gr)
        kk[m] = rk
        m = m+1

    peak_index: int = int(np.argmax(growth))
    rk = float(kk[peak_index])
    aa[0,0] = -U1*(rk*rk+el*el+1./(2.-delta))+(B+D/(2.-delta))+ep*(0.+1.j)*L*C/rk
    aa[0,1] = U1/(2.-delta)-ep*(0.+1.j)*L*C/rk
    aa[0,2] = -ep*(0.+1.j)*L/rk
    bb[0,0] = -(rk*rk+el*el+1./(2.-delta))
    bb[0,1] = 1./(2.-delta)
    bb[0,2] = 0.
    aa[1,0] = U2/delta-ep*(0.+1.j)*L*C/rk
    aa[1,1] = -U2*(rk*rk+el*el+1./delta)+(B-D/delta)+(0.+1.j)*R*(rk*rk+el*el)/rk+ep*(0.+1.j)*L*C/rk
    aa[1,2] = ep*(0.+1.j)*L/rk
    bb[1,0] = 1./delta
    bb[1,1] = -(rk*rk+el*el+1./delta)
    bb[1,2] = 0.
    aa[2,0] = U2+(0.+1.j)*(1.-L)*C*ep/rk
    aa[2,1] = -U2-(C+D)-(0.+1.j)*(1.-L)*C*ep/rk
    aa[2,2] = U2-(0.+1.j)*(1.-L)*ep/rk
    bb[2,0] = 1.
    bb[2,1] = -1.
    bb[2,2] = 1.
    evals, V = scipy.linalg.eig(aa, bb)
    gr = evals.imag*rk
    peak_mode_index: int = int(np.argmax(evals.imag * rk))
    
    psi_1_prime = V[0, peak_mode_index]
    psi_2_prime = V[1, peak_mode_index]
    P_prime = 0.5*V[2, peak_mode_index]*L

    x: numpy.typing.NDArray[np.floating] = np.arange(0,6*np.pi,.1)
    dx = .1
    psi_1 = (np.outer(psi_1_prime.real, np.cos(rk * x)) - np.outer(psi_1_prime.imag, np.sin(rk * x)))[0,:]
    psi_2 = (np.outer(psi_2_prime.real, np.cos(rk * x)) - np.outer(psi_2_prime.imag, np.sin(rk * x)))[0,:]
    P = (np.outer(P_prime.real, np.cos(rk * x)) - np.outer(P_prime.imag, np.sin(rk * x)))[0,:]
    
    q_1_prime = laplacian(f=psi_1, dx=dx) - (psi_1 - psi_2)
    q_2_prime = laplacian(f=psi_2, dx=dx) + (psi_1 - psi_2)
        
    return kk, growth, q_1_prime, q_2_prime, P