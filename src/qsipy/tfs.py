"""tfs.py
Provides analytical functions related to the phase resolution during an interferometry
experiment using twin-Fock states (i.e. |n,n>) as an input state.
The interferometer considered is a generic Mach-Zehnder with a phase difference ϕ
between both arms.
The detectors may have a finite quantum efficiency η.

In order to minimise sub-functions calls, we kept most of the functions in an expanded,
analytical form.
"""

import numpy as np
import numpy.typing as npt
from .trigonometry import arccsc, arcsec

# - OBSERVABLE: Var. of the difference of numbers of particles between both arms
# -------------------------------------------------------------------------------

# |- DETECTORS: perfects
# -----------------------


def ev_variance_difference_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the variance of the difference of number of
    particles detected at both output arms of the interferometer.
    The detectors are perfect, meaning that their quantum efficiency is equal to 1.
    Since the expectation value of the difference itself is zero, the variance is
    actually equal to the expectation value of the square of the difference.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(N_output1 - N_output2) ** 2>
    """
    return 0.25 * N * (1 + N / 2) * np.sin(phi) ** 2


def ev_difference_quarted_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the power four of the difference of number of
    particles detected at both output arms of the interferometer.
    The detectors are perfect, meaning that their quantum efficiency is equal to 1.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(N_output1 - N_output2) ** 4>
    """
    return (
        0.25
        * N
        * (1 + N / 2)
        * np.sin(phi) ** 2
        * (1 + 1.5 * (-1 + N / 4 + N**2 / 8) * np.sin(phi) ** 2)
    )


def ev_fourth_moment_difference_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the variance of the variance (4th moment) of
    the difference of number of particles detected at both output arms of the
    interferometer.
    The detectors are perfect, meaning that their quantum efficiency is equal to 1.
    This function is equal to:

    (ev_difference_quarted_perfect_qe - ev_difference_squared_perfect_qe ** 2)

    It corresponds to the square of the noise on the variance of the difference of
    number of particles detected.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <Var(Var(N_output1 - N_output2))>
    """
    return (
        (1 / 16)
        * N
        * (1 + N / 2)
        * np.sin(phi) ** 2
        * (4 + (-6 + N / 2 + N**2 / 4) * np.sin(phi) ** 2)
    )


def noise_difference_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
) -> float | npt.NDArray[np.float_]:
    """Returns the noise on the variance of the difference of number of particles
    detected at both output arms of the interferometer.
    The detectors are perfect, meaning that their quantum efficiency is equal to 1.
    This function is equal to:

    sqrt(ev_fourth_moment_difference_perfect_qe)

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.

    Returns
    -------
    float | npt.NDArray[np.float_]
        RMS noise on the variance of the difference of number of particles detected.
    """
    return np.sqrt(ev_fourth_moment_difference_perfect_qe(phi, N))


def phase_resolution_difference_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
) -> float | npt.NDArray[np.float_]:
    """Returns the resolution of the phase estimation during an interferometry
    experiment using twin-Fock states, perfect detectors and considering the variance
    of the difference of particles at the output as the observable of interest.
    The detectors are perfect, meaning that their quantum efficiency is equal to 1.
    Notice that this resolution depends on the phase difference phi.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Phase resolution.
    """
    return (
        (np.sqrt(2) / 2)
        * (1 / np.sqrt((N / 2) * (1 + N / 2)))
        * (1 / np.cos(phi))
        * np.sqrt(1 + 0.25 * (-6 + N / 2 + N**2 / 4) * np.sin(phi) ** 2)
    )


# |- DETECTORS: finite quantum efficiency
# ---------------------------------------

# ||- general shape


def ev_variance_difference_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the variance of the difference of number of
    particles detected at both output arms of the interferometer.
    The detectors have a finite quantum efficiency eta.
    Since the expectation value of the difference itself is zero, the variance is
    actually equal to the expectation value of the square of the difference.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(N_output1 - N_output2) ** 2>
    """
    return eta**2 * (N / 4) * (1 + N / 2) * np.sin(phi) ** 2 + eta * (1 - eta) * N / 4


def ev_difference_quarted_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the power four of the difference of number of
    particles detected at both output arms of the interferometer.
    The detectors have a finite quantum efficiency eta.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(N_output1 - N_output2) ** 4>
    """
    return (N * eta / 1024) * (
        64
        - 320 * eta
        + 256 * N * eta
        + 384 * eta**2
        - 384 * N * eta**2
        + 96 * N**2 * eta**2
        - 144 * eta**3
        + 156 * N * eta**3
        - 60 * N**2 * eta**3
        + 9 * N**3 * eta**3
        - 4
        * (2 + N)
        * eta
        * (16 + 24 * (-2 + N) * eta + 3 * (8 - 6 * N + N**2) * eta**2)
        * np.cos(2 * phi)
        + 3 * (-16 - 4 * N + 4 * N**2 + N**3) * eta**3 * np.cos(4 * phi)
    )


def ev_fourth_moment_difference_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the variance of the variance (4th moment) of
    the difference of number of particles detected at both output arms of the
    interferometer.
    The detectors have a finite quantum efficiency eta.
    This function is equal to:

    (ev_difference_quarted_finite_qe - ev_difference_squared_finite_qe ** 2)

    It corresponds to the square of the noise on the variance of the difference of
    number of particles detected.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <Var(Var(N_output1 - N_output2))>
    """
    return (N * eta / 1024) * (
        64
        - 320 * eta
        + 192 * N * eta
        + 384 * eta**2
        - 320 * N * eta**2
        + 64 * N**2 * eta**2
        - 144 * eta**3
        + 132 * N * eta**3
        - 52 * N**2 * eta**3
        + 3 * N**3 * eta**3
        - 4
        * (2 + N)
        * eta
        * (16 + 16 * (-3 + N) * eta + (24 - 14 * N + N**2) * eta**2)
        * np.cos(2 * phi)
        + (-48 - 20 * N + 4 * N**2 + N**3) * eta**3 * np.cos(4 * phi)
    )


def noise_difference_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the noise on the variance of the difference of number of particles
    detected at both output arms of the interferometer.
    The detectors have a finite quantum efficiency eta.
    This function is equal to:

    sqrt(ev_fourth_moment_difference_finite_qe)

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        RMS noise on the variance of the difference of number of particles detected.
    """
    return np.sqrt(ev_fourth_moment_difference_finite_qe(phi, N, eta))


def phase_resolution_difference_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the resolution of the phase estimation during an interferometry
    experiment using twin-Fock states, detectors with finite quantum efficiency eta and
    considering the variance of the difference of particles at the output as the
    observable of interest.
    Notice that this resolution depends on the phase difference phi.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Phase resolution.
    """
    p0 = (
        64
        - 320 * eta
        + 192 * N * eta
        + 384 * eta**2
        - 320 * N * eta**2
        + 64 * N**2 * eta**2
        - 144 * eta**3
        + 132 * N * eta**3
        - 52 * N**2 * eta**3
        + 3 * N**3 * eta**3
    )

    p1 = (
        -4
        * (N + 2)
        * eta
        * (16 + 16 * (-3 + N) * eta + (24 - 14 * N + N**2) * eta**2)
    )

    p2 = (-48 - 20 * N + 4 * N**2 + N**3) * eta**3

    return np.sqrt(N * eta * (p0 + p1 * np.cos(2 * phi) + p2 * np.cos(4 * phi))) / (
        8 * N * (N + 2) * eta**2 * np.sin(phi) * np.cos(phi)
    )


# ||- behaviour at the optimal phase


def derivative_phase_resolution_difference_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the derivative (with respect to phi) of the resolution of the phase
    estimation during an interferometry experiment using twin-Fock states, detectors
    with finite quantum efficiency and considering the variance of the difference of
    particles at the output as the observable of interest. The detectors have a finite
    quantum efficiency eta.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Derivative of the phase resolution.
    """
    n = N / 2
    return (
        -((1 + n) * eta * (4 + 6 * (-2 + eta) * eta + n * eta * (8 + (-7 + n) * eta)))
        - (-1 + eta)
        * (-1 + 2 * (-3 + 2 * n) * (-1 + eta) * eta)
        * (1 / np.sin(phi) ** 2)
        + (
            1
            + eta
            * (-3 + n * (8 + 3 * (-4 + eta) * eta + n * eta * (8 + (-6 + n) * eta)))
        )
        * (1 / np.cos(phi) ** 2)
    ) / (
        (1 + n)
        * eta
        * np.sqrt(
            n
            * eta
            * (
                8
                + 8 * (-5 + 6 * n) * eta
                + 16 * (-1 + n) * (-3 + 2 * n) * eta**2
                + (-18 + n * (33 + n * (-26 + 3 * n))) * eta**3
                + (1 + n)
                * eta
                * (
                    -4
                    * (4 + 6 * (-2 + eta) * eta + n * eta * (8 + (-7 + n) * eta))
                    * np.cos(2 * phi)
                    + (-6 + n + n**2) * eta**2 * np.cos(4 * phi)
                )
            )
        )
    )


def optimal_phi_difference(
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the optimal phase to estimate (minimizing the resolution) during an
    interferometry experiment using twin-Fock states, detectors with finite quantum
    efficiency and considering the variance of the difference of particles at the
    output as the observable of interest. The detectors have a finite quantum efficiency
    eta.

    Parameters
    ----------
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Optimal phase to estimate experimentally.
    """
    n = N / 2
    return arccsc(
        np.sqrt(
            (
                1
                + (-7 + 4 * n) * eta
                + (12 - 8 * n) * eta**2
                + (-6 + 4 * n) * eta**3
                + np.sqrt(
                    (-1 + eta)
                    * (
                        -1
                        + (9 - 12 * n) * eta
                        - 4 * (6 - 19 * n + 10 * n**2) * eta**2
                        + (18 - 135 * n + 134 * n**2 - 33 * n**3) * eta**3
                        + 2 * n * (45 - 72 * n + 31 * n**2 - 2 * n**3) * eta**4
                        + 2 * n * (-9 + 24 * n - 15 * n**2 + 2 * n**3) * eta**5
                    )
                )
            )
            / ((-1 + eta) * (-1 + (6 - 4 * n) * eta + (-6 + 4 * n) * eta**2))
        )
    )


def resolution_at_optimal_phi_difference(
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the resolution at the optimal phase to estimate during an interferometry
    experiment using twin-Fock states, detectors with finite quantum efficiency eta and
    considering the variance of the difference of particles at the output as the
    observable of interest.
    This function is therefore just:

    phase_resolution_difference_finite_qe(optimal_phi_difference(n, eta), n, eta)

    Parameters
    ----------
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Optimal resolution.
    """
    n = N / 2
    return (
        1
        / (4 * n * (n + 1) * eta**2)
        * np.sqrt(
            n
            * eta
            * (
                8
                + 8 * (-5 + 6 * n) * eta
                + 16 * (-1 + n) * (-3 + 2 * n) * eta**2
                + (-18 + n * (33 + n * (-26 + 3 * n))) * eta**3
                + (1 + n)
                * eta
                * (
                    4
                    * (4 + 6 * (-2 + eta) * eta + n * eta * (8 + (-7 + n) * eta))
                    * np.cos(
                        2
                        * arcsec(
                            np.sqrt(
                                (
                                    1
                                    + eta
                                    * (
                                        -7
                                        + 4 * n * (-1 + eta) ** 2
                                        - 6 * (-2 + eta) * eta
                                    )
                                    + np.sqrt(
                                        (-1 + eta)
                                        * (-1 + 2 * (-3 + 2 * n) * (-1 + eta) * eta)
                                        * (
                                            1
                                            + eta
                                            * (
                                                -3
                                                + n
                                                * (
                                                    8
                                                    + 3 * (-4 + eta) * eta
                                                    + n * eta * (8 + (-6 + n) * eta)
                                                )
                                            )
                                        )
                                    )
                                )
                                / (
                                    (-1 + eta)
                                    * (-1 + 2 * (-3 + 2 * n) * (-1 + eta) * eta)
                                )
                            )
                        )
                    )
                    + (-6 + n + n**2)
                    * eta**2
                    * np.cos(
                        4
                        * arcsec(
                            np.sqrt(
                                (
                                    1
                                    + eta
                                    * (
                                        -7
                                        + 4 * n * (-1 + eta) ** 2
                                        - 6 * (-2 + eta) * eta
                                    )
                                    + np.sqrt(
                                        (-1 + eta)
                                        * (-1 + 2 * (-3 + 2 * n) * (-1 + eta) * eta)
                                        * (
                                            1
                                            + eta
                                            * (
                                                -3
                                                + n
                                                * (
                                                    8
                                                    + 3 * (-4 + eta) * eta
                                                    + n * eta * (8 + (-6 + n) * eta)
                                                )
                                            )
                                        )
                                    )
                                )
                                / (
                                    (-1 + eta)
                                    * (-1 + 2 * (-3 + 2 * n) * (-1 + eta) * eta)
                                )
                            )
                        )
                    )
                )
            )
        )
        / np.sin(
            2
            * arcsec(
                np.sqrt(
                    (
                        1
                        + eta * (-7 + 4 * n * (-1 + eta) ** 2 - 6 * (-2 + eta) * eta)
                        + np.sqrt(
                            (-1 + eta)
                            * (-1 + 2 * (-3 + 2 * n) * (-1 + eta) * eta)
                            * (
                                1
                                + eta
                                * (
                                    -3
                                    + n
                                    * (
                                        8
                                        + 3 * (-4 + eta) * eta
                                        + n * eta * (8 + (-6 + n) * eta)
                                    )
                                )
                            )
                        )
                    )
                    / ((-1 + eta) * (-1 + 2 * (-3 + 2 * n) * (-1 + eta) * eta))
                )
            )
        )
    )


def asymptotic_limit_resolution_at_optimal_phi_difference(
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the asymptotic limit (as the number of particles goes to infinity) of the ratio between:
        - the resolution at the optimal phase and considering the variance of the difference of
          particles at the output as the observable of interest
        - the SQL 1/sqrt(eta x N).

    It only depends on the quantum efficiency of the detectors.

    Parameters
    ----------
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Optimal resolution to SQL ratio in the asymptotic limit of n.
    """
    return np.sqrt(3) * np.sqrt(1 - eta)
