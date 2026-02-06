"""Build coefficient matrices for the 2-layer QG generalized eigenvalue problem."""

import typing

import numpy as np
import numpy.typing


def build_matrices(
    *,
    u1: numpy.typing.NDArray[np.floating[typing.Any]],
    u2: numpy.typing.NDArray[np.floating[typing.Any]],
    beta: float,
    dy: float,
    n_2: int,
    rk: float,
    half_matrix: int,
    n: int,
) -> tuple[numpy.typing.NDArray[np.floating[typing.Any]], numpy.typing.NDArray[np.floating[typing.Any]]]:
    M: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros((n_2 - 4, n_2 - 4))
    N: numpy.typing.NDArray[np.floating[typing.Any]] = np.zeros((n_2 - 4, n_2 - 4))

    for j in range(n - 4):
        M[j, j] = -u1[j + 1] * (rk**2 * dy**2 + 2.0 + dy**2)
        M[j, j] += (beta * dy**2 - (u1[j + 2] + u1[j] - 2.0 * u1[j + 1]))
        M[j, j] += (u1[j + 1] - u2[j + 1]) * dy**2
        M[j + half_matrix, j + half_matrix] = -u2[j + 1] * (rk**2 * dy**2 + 2.0 + dy**2)
        M[j + half_matrix, j + half_matrix] += (beta * dy**2 - (u2[j + 2] + u2[j] - 2.0 * u2[j + 1]))
        M[j + half_matrix, j + half_matrix] += -(u1[j + 1] - u2[j + 1]) * dy**2
        M[j, j + half_matrix] = u1[j + 1] * dy**2
        M[j + half_matrix, j] = u2[j + 1] * dy**2
        M[j, j + 1] = u1[j + 1]
        M[j + 1, j] = u1[j + 2]
        M[j + half_matrix, j + half_matrix + 1] = u2[j + 1]
        M[j + half_matrix + 1, j + half_matrix] = u2[j + 2]

        N[j, j + 1] = 1.0
        N[j + 1, j] = 1.0
        N[j + half_matrix, j + half_matrix + 1] = 1.0
        N[j + half_matrix + 1, j + half_matrix] = 1.0
        N[j, j] = -(rk**2 * dy**2 + 2.0 + dy**2)
        N[j + half_matrix, j + half_matrix] = -(rk**2 * dy**2 + 2.0 + dy**2)
        N[j, j + half_matrix] = dy**2
        N[j + half_matrix, j] = dy**2

    jo = n - 3
    M[jo, jo] = -u1[jo + 1] * (rk**2 * dy**2 + 2.0 + dy**2)
    M[jo, jo] += (beta * dy**2 - (u1[jo + 2] + u1[jo] - 2.0 * u1[jo + 1]))
    M[jo, jo] += (u1[jo + 1] - u2[jo + 1]) * dy**2
    M[jo + half_matrix, jo + half_matrix] = -u2[jo + 1] * (rk**2 * dy**2 + 2.0 + dy**2)
    M[jo + half_matrix, jo + half_matrix] += (beta * dy**2 - (u2[jo + 2] + u2[jo] - 2.0 * u2[jo + 1]))
    M[jo + half_matrix, jo + half_matrix] += -(u1[jo + 1] - u2[jo + 1]) * dy**2
    M[jo, jo + half_matrix] = u1[jo + 1] * dy**2
    M[jo + half_matrix, jo] = u2[jo + 1] * dy**2

    N[jo, jo] = -(rk**2 * dy**2 + 2.0 + dy**2)
    N[jo + half_matrix, jo + half_matrix] = -(rk**2 * dy**2 + 2.0 + dy**2)
    N[jo, jo + half_matrix] = dy**2
    N[jo + half_matrix, jo] = dy**2

    return M, N
