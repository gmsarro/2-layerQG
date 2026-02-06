"""LWA budget helper functions.

Each function computes one term of the local wave activity budget
for a 2-layer quasi-geostrophic channel model.  All arrays follow
the convention (time, latitude, longitude) for 3-D fields and
(time, latitude) for 2-D fields.
"""

from typing import Any

import numpy as np
import numpy.typing


def lwatend(
    *,
    lwa: numpy.typing.NDArray[np.floating[Any]],
    dt: float,
) -> numpy.typing.NDArray[np.floating[Any]]:
    """Compute time tendency of LWA using leapfrog (forward/backward at ends).

    Parameters
    ----------
    lwa : array (time, latitude, longitude)
        Local wave activity field.
    dt : float
        Time step between snapshots.

    Returns
    -------
    array
        Time derivative of LWA with same shape as *lwa*.
    """
    result: numpy.typing.NDArray[np.floating[Any]] = np.zeros(np.shape(lwa))
    result[1:-1, :, :] = (lwa[2:, :, :] - lwa[:-2, :, :]) / (2 * dt)
    result[0, :, :] = (lwa[1, :, :] - lwa[0, :, :]) / dt
    result[-1, :, :] = (lwa[-1, :, :] - lwa[-2, :, :]) / dt
    return result


def urefadv(
    *,
    lwa: numpy.typing.NDArray[np.floating[Any]],
    uref: numpy.typing.NDArray[np.floating[Any]],
    dx: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[Any]]:
    """Compute advection of LWA by the reference wind U_ref.

    Parameters
    ----------
    lwa : array (time, latitude, longitude)
        LWA field.
    uref : array (time, latitude)
        Reference zonal wind.
    dx : float
        Zonal grid spacing.
    filt : bool
        If True, apply 1-2-1 temporal filter.

    Returns
    -------
    array
        Tendency due to U_ref advection.
    """
    lwagradx: numpy.typing.NDArray[np.floating[Any]] = (
        np.roll(lwa, -1, axis=2) - np.roll(lwa, 1, axis=2)
    ) / (2 * dx)
    out: numpy.typing.NDArray[np.floating[Any]] = -lwagradx * uref[:, :, np.newaxis]
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def ueadv(
    *,
    q: numpy.typing.NDArray[np.floating[Any]],
    qref: numpy.typing.NDArray[np.floating[Any]],
    u: numpy.typing.NDArray[np.floating[Any]],
    uref: numpy.typing.NDArray[np.floating[Any]],
    dx: float,
    dy: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[Any]]:
    """Compute advection of PV anomaly by the eddy wind.

    Parameters
    ----------
    q : array (time, latitude, longitude)
        Full PV field.
    qref : array (time, latitude)
        Reference PV.
    u : array (time, latitude, longitude)
        Full zonal wind.
    uref : array (time, latitude)
        Reference zonal wind.
    dx : float
        Zonal grid spacing.
    dy : float
        Meridional grid spacing.
    filt : bool
        If True, apply 1-2-1 temporal filter.

    Returns
    -------
    array
        Tendency from eddy advection.
    """
    tn, yn, xn = np.shape(q)
    Iuq: numpy.typing.NDArray[np.floating[Any]] = np.zeros(np.shape(q))
    for t in range(tn):
        for y1 in range(yn):
            q_e = q[t, :, :] - qref[t, y1]
            u_e = u[t, :, :] - uref[t, y1]
            for x in range(xn):
                for y2 in range(yn):
                    if y2 < y1 and q_e[y2, x] > 0:
                        Iuq[t, y1, x] += u_e[y2, x] * q_e[y2, x] * dy
                    if y2 >= y1 and q_e[y2, x] <= 0:
                        Iuq[t, y1, x] += u_e[y2, x] * q_e[y2, x] * (-dy)
    out: numpy.typing.NDArray[np.floating[Any]] = (
        np.roll(Iuq, -1, axis=2) - np.roll(Iuq, 1, axis=2)
    ) / (2 * dx)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return -out


def eddyflux_x(
    *,
    ue: numpy.typing.NDArray[np.floating[Any]],
    ve: numpy.typing.NDArray[np.floating[Any]],
    dx: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[Any]]:
    """Compute zonal EP flux convergence, -1/2 d/dx (v^2 - u^2).

    Parameters
    ----------
    ue : array (time, latitude, longitude)
        Eddy zonal wind.
    ve : array (time, latitude, longitude)
        Eddy meridional wind.
    dx : float
        Zonal grid spacing.
    filt : bool
        If True, apply 1-2-1 temporal filter.

    Returns
    -------
    array
        Zonal eddy flux convergence.
    """
    v2_u2: numpy.typing.NDArray[np.floating[Any]] = 0.5 * (ve[:, :, :] ** 2 - ue[:, :, :] ** 2)
    out: numpy.typing.NDArray[np.floating[Any]] = -(
        np.roll(v2_u2, -1, axis=2) - np.roll(v2_u2, 1, axis=2)
    ) / (2 * dx)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def eddyflux_y(
    *,
    ue: numpy.typing.NDArray[np.floating[Any]],
    ve: numpy.typing.NDArray[np.floating[Any]],
    dy: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[Any]]:
    """Compute meridional EP flux convergence, d/dy (uv), with zero at boundaries.

    Parameters
    ----------
    ue : array (time, latitude, longitude)
        Eddy zonal wind.
    ve : array (time, latitude, longitude)
        Eddy meridional wind.
    dy : float
        Meridional grid spacing.
    filt : bool
        If True, apply 1-2-1 temporal filter.

    Returns
    -------
    array
        Meridional eddy flux convergence.
    """
    uv: numpy.typing.NDArray[np.floating[Any]] = np.pad(
        ue[:, :, :] * ve[:, :, :], ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=0,
    )
    out: numpy.typing.NDArray[np.floating[Any]] = (uv[:, 2:, :] - uv[:, :-2, :]) / (2 * dy)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def eddyflux_z(
    *,
    ve: numpy.typing.NDArray[np.floating[Any]],
    te: numpy.typing.NDArray[np.floating[Any]],
    Ld: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[Any]]:
    """Compute vertical eddy heat flux convergence, vT / Ld^2.

    Parameters
    ----------
    ve : array (time, latitude, longitude)
        Eddy meridional wind.
    te : array (time, latitude, longitude)
        Eddy temperature.
    Ld : float
        Deformation radius.
    filt : bool
        If True, apply 1-2-1 temporal filter.

    Returns
    -------
    array
        Vertical eddy flux convergence.
    """
    out: numpy.typing.NDArray[np.floating[Any]] = ve * te / (Ld ** 2)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def eddyflux(
    *,
    ve: numpy.typing.NDArray[np.floating[Any]],
    qe: numpy.typing.NDArray[np.floating[Any]],
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[Any]]:
    """Compute EP flux term -v'q'.

    Parameters
    ----------
    ve : array (time, latitude, longitude)
        Eddy meridional wind.
    qe : array (time, latitude, longitude)
        Eddy PV anomaly.
    filt : bool
        If True, apply 1-2-1 temporal filter.

    Returns
    -------
    array
        EP flux term.
    """
    out: numpy.typing.NDArray[np.floating[Any]] = -ve * qe
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def LH(
    *,
    p: numpy.typing.NDArray[np.floating[Any]],
    q: numpy.typing.NDArray[np.floating[Any]],
    qref: numpy.typing.NDArray[np.floating[Any]],
    L: float,
    dx: float,
    dy: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[Any]]:
    """Compute latent heating contribution to the LWA budget.

    Parameters
    ----------
    p : array (time, latitude, longitude)
        Precipitation field.
    q : array (time, latitude, longitude)
        Full PV field.
    qref : array (time, latitude)
        Reference PV.
    L : float
        Latent heating parameter.
    dx : float
        Zonal grid spacing (unused; kept for interface consistency).
    dy : float
        Meridional grid spacing.
    filt : bool
        If True, apply 1-2-1 temporal filter.

    Returns
    -------
    array
        Latent heating budget term.
    """
    tn, yn, xn = np.shape(p)
    out: numpy.typing.NDArray[np.floating[Any]] = np.zeros(np.shape(p))
    for t in range(tn):
        for y1 in range(yn):
            q_e = q[t, :, :] - qref[t, y1]
            for x in range(xn):
                for y2 in range(yn):
                    if y2 < y1 and q_e[y2, x] > 0:
                        out[t, y1, x] += L * p[t, y2, x] * dy
                    if y2 >= y1 and q_e[y2, x] <= 0:
                        out[t, y1, x] += L * p[t, y2, x] * (-dy)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return -out
