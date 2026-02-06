"""LWA budget helper functions."""

import typing

import numpy as np
import numpy.typing


def lwatend(
    *,
    lwa: numpy.typing.NDArray[np.floating[typing.Any]],
    dt: float,
) -> numpy.typing.NDArray[np.floating[typing.Any]]:
    result: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros(np.shape(lwa))
    result[1:-1, :, :] = (lwa[2:, :, :] - lwa[:-2, :, :]) / (2 * dt)
    result[0, :, :] = (lwa[1, :, :] - lwa[0, :, :]) / dt
    result[-1, :, :] = (lwa[-1, :, :] - lwa[-2, :, :]) / dt
    return result


def urefadv(
    *,
    lwa: numpy.typing.NDArray[np.floating[typing.Any]],
    uref: numpy.typing.NDArray[np.floating[typing.Any]],
    dx: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[typing.Any]]:
    lwagradx: numpy.typing.NDArray[np.floating[typing.Any]] = (
        np.roll(lwa, -1, axis=2) - np.roll(lwa, 1, axis=2)
    ) / (2 * dx)
    out: numpy.typing.NDArray[np.floating[typing.Any]] = -lwagradx * uref[:, :, np.newaxis]
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def ueadv(
    *,
    q: numpy.typing.NDArray[np.floating[typing.Any]],
    qref: numpy.typing.NDArray[np.floating[typing.Any]],
    u: numpy.typing.NDArray[np.floating[typing.Any]],
    uref: numpy.typing.NDArray[np.floating[typing.Any]],
    dx: float,
    dy: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[typing.Any]]:
    tn, yn, xn = np.shape(q)
    Iuq: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros(np.shape(q))
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
    out: numpy.typing.NDArray[np.floating[typing.Any]] = (
        np.roll(Iuq, -1, axis=2) - np.roll(Iuq, 1, axis=2)
    ) / (2 * dx)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return -out


def eddyflux_x(
    *,
    ue: numpy.typing.NDArray[np.floating[typing.Any]],
    ve: numpy.typing.NDArray[np.floating[typing.Any]],
    dx: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[typing.Any]]:
    v2_u2: numpy.typing.NDArray[np.floating[typing.Any]] = 0.5 * (ve[:, :, :] ** 2 - ue[:, :, :] ** 2)
    out: numpy.typing.NDArray[np.floating[typing.Any]] = -(
        np.roll(v2_u2, -1, axis=2) - np.roll(v2_u2, 1, axis=2)
    ) / (2 * dx)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def eddyflux_y(
    *,
    ue: numpy.typing.NDArray[np.floating[typing.Any]],
    ve: numpy.typing.NDArray[np.floating[typing.Any]],
    dy: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[typing.Any]]:
    uv: numpy.typing.NDArray[np.floating[typing.Any]] = np.pad(
        ue[:, :, :] * ve[:, :, :], ((0, 0), (1, 1), (0, 0)), mode='constant', constant_values=0,
    )
    out: numpy.typing.NDArray[np.floating[typing.Any]] = (uv[:, 2:, :] - uv[:, :-2, :]) / (2 * dy)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def eddyflux_z(
    *,
    ve: numpy.typing.NDArray[np.floating[typing.Any]],
    te: numpy.typing.NDArray[np.floating[typing.Any]],
    Ld: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[typing.Any]]:
    out: numpy.typing.NDArray[np.floating[typing.Any]] = ve * te / (Ld ** 2)
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def eddyflux(
    *,
    ve: numpy.typing.NDArray[np.floating[typing.Any]],
    qe: numpy.typing.NDArray[np.floating[typing.Any]],
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[typing.Any]]:
    out: numpy.typing.NDArray[np.floating[typing.Any]] = -ve * qe
    if filt:
        out[1:-1, :] = out[:-2, :] * 0.25 + out[1:-1, :] * 0.5 + out[2:, :] * 0.25
    return out


def LH(
    *,
    p: numpy.typing.NDArray[np.floating[typing.Any]],
    q: numpy.typing.NDArray[np.floating[typing.Any]],
    qref: numpy.typing.NDArray[np.floating[typing.Any]],
    L: float,
    dx: float,
    dy: float,
    filt: bool = True,
) -> numpy.typing.NDArray[np.floating[typing.Any]]:
    tn, yn, xn = np.shape(p)
    out: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros(np.shape(p))
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
