"""LWA budget helper functions."""

import numpy as np
import numpy.typing


def lwatend(
    *,
    lwa: numpy.typing.NDArray[np.floating],
    dt: float,
) -> numpy.typing.NDArray[np.floating]:
    """Compute time tendency of LWA using leapfrog (forward/backward at ends).

    :param lwa: Local wave activity array with dimensions (time, latitude, longitude)
    :param dt: Time step
    :return: Time derivative of LWA with same shape as input
    """
    result: numpy.typing.NDArray[np.floating] = np.zeros(np.shape(lwa))
    result[1:-1, :, :] = (lwa[2:, :, :] - lwa[:-2, :, :]) / (2 * dt)
    result[0, :, :] = (lwa[1, :, :] - lwa[0, :, :]) / dt
    result[-1, :, :] = (lwa[-1, :, :] - lwa[-2, :, :]) / dt
    return result


def urefadv(
    *,
    lwa: numpy.typing.NDArray[np.floating],
    uref: numpy.typing.NDArray[np.floating],
    dx: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating]:
    """Compute advection of LWA by reference wind Uref.

    :param lwa: LWA field (time, latitude, longitude)
    :param uref: Reference zonal wind (time, latitude)
    :param dx: Zonal grid spacing
    :param filt: If True, apply 1-2-1 temporal filter to the result
    :return: Tendency due to Uref advection
    """
    lwagradx: numpy.typing.NDArray[np.floating] = (
        np.roll(lwa, -1, axis=2) - np.roll(lwa, 1, axis=2)
    ) / (2 * dx)
    out: numpy.typing.NDArray[np.floating] = -lwagradx * uref[:, :, np.newaxis]
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def ueadv(
    *,
    q: numpy.typing.NDArray[np.floating],
    qref: numpy.typing.NDArray[np.floating],
    u: numpy.typing.NDArray[np.floating],
    uref: numpy.typing.NDArray[np.floating],
    dx: float,
    dy: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating]:
    """Compute advection of PV anomaly by eddy wind contribution.

    :param q: Full PV (time, latitude, longitude)
    :param qref: Reference PV (time, latitude)
    :param u: Full zonal wind (time, latitude, longitude)
    :param uref: Reference zonal wind (time, latitude)
    :param dx: Zonal grid spacing
    :param dy: Meridional grid spacing
    :param filt: If True, apply 1-2-1 temporal filter
    :return: Tendency term from eddy advection
    """
    tn, yn, xn = np.shape(q)
    Iuq: numpy.typing.NDArray[np.floating] = np.zeros(np.shape(q))
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
    out: numpy.typing.NDArray[np.floating] = (
        np.roll(Iuq, -1, axis=2) - np.roll(Iuq, 1, axis=2)
    ) / (2 * dx)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return -out


def eddyflux_x(
    *,
    ue: numpy.typing.NDArray[np.floating],
    ve: numpy.typing.NDArray[np.floating],
    dx: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating]:
    """Compute zonal EP flux convergence, -1/2 d/dx (v^2 - u^2).

    :param ue: Eddy zonal wind (time, latitude, longitude)
    :param ve: Eddy meridional wind (time, latitude, longitude)
    :param dx: Zonal grid spacing
    :param filt: If True, apply 1-2-1 temporal filter
    :return: Zonal eddy flux convergence term
    """
    v2_u2: numpy.typing.NDArray[np.floating] = 0.5 * (ve[:, :, :] ** 2 - ue[:, :, :] ** 2)
    out: numpy.typing.NDArray[np.floating] = -(
        np.roll(v2_u2, -1, axis=2) - np.roll(v2_u2, 1, axis=2)
    ) / (2 * dx)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def eddyflux_y(
    *,
    ue: numpy.typing.NDArray[np.floating],
    ve: numpy.typing.NDArray[np.floating],
    dy: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating]:
    """Compute meridional EP flux convergence, d/dy (uv), with zero at boundaries.

    :param ue: Eddy zonal wind
    :param ve: Eddy meridional wind
    :param dy: Meridional grid spacing
    :param filt: If True, apply 1-2-1 temporal filter
    :return: Meridional eddy flux convergence term
    """
    uv: numpy.typing.NDArray[np.floating] = np.pad(
        ue[:, :, :] * ve[:, :, :], ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=0,
    )
    out: numpy.typing.NDArray[np.floating] = (uv[:, 2:, :] - uv[:, :-2, :]) / (2 * dy)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def eddyflux_z(
    *,
    ve: numpy.typing.NDArray[np.floating],
    te: numpy.typing.NDArray[np.floating],
    Ld: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating]:
    """Compute vertical eddy heat flux convergence, vT/Ld**2.

    :param ve: Eddy meridional wind
    :param te: Eddy temperature
    :param Ld: Deformation radius
    :param filt: If True, apply 1-2-1 temporal filter
    :return: Vertical eddy flux convergence term
    """
    out: numpy.typing.NDArray[np.floating] = ve * te / (Ld ** 2)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def eddyflux(
    *,
    ve: numpy.typing.NDArray[np.floating],
    qe: numpy.typing.NDArray[np.floating],
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating]:
    """Compute EP flux term -ve*qe.

    :param ve: Eddy meridional wind
    :param qe: Eddy PV (or tracer) anomaly
    :param filt: If True, apply 1-2-1 temporal filter
    :return: EP flux term
    """
    out: numpy.typing.NDArray[np.floating] = -ve * qe
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def LH(
    *,
    p: numpy.typing.NDArray[np.floating],
    q: numpy.typing.NDArray[np.floating],
    qref: numpy.typing.NDArray[np.floating],
    L: float,
    dx: float,
    dy: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating]:
    """Compute latent heating contribution integrated meridionally with sign selection.

    :param p: Precipitation field (time, latitude, longitude)
    :param q: Full PV (time, latitude, longitude)
    :param qref: Reference PV (time, latitude)
    :param L: Latent heating parameter
    :param dx: Zonal grid spacing (unused but kept for consistency)
    :param dy: Meridional grid spacing
    :param filt: If True, apply 1-2-1 temporal filter
    :return: Latent heating term
    """
    tn, yn, xn = np.shape(p)
    out: numpy.typing.NDArray[np.floating] = np.zeros(np.shape(p))
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
