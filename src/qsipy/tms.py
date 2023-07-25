"""tms.py
Provides analytical functions related to the phase uncertainty during an interferometry
experiment using two-mode squeezed vacuum states as an input state. We denote "nu" the
average population per mode of the TSM. The total average number of particles N is 
therefore N = 2*nu.
The interferometer considered is a generic Mach-Zehnder with a phase difference ϕ
between both arms.
The detectors may have a finite quantum efficiency η.

In order to minimise functions calls, we keep most of the functions in an expanded
analytical form.

abbreviations:
--------------
    - ev: Expectation Value
    - vhd: Variance of the Half Difference of the number of particles at the output
        (i.e. our observable)

general notations:
------------------
    - N: the total number of particles in the interferometer. It corresponds to the
        main experimental resource.
    - phi: the phase difference between both arms of the interferometer
    - eta: quantum efficiency of the detectors (between 0 and 1). More generally it may
        characterize any loss process occurring after the last beam splitter of the
        interferometer.
"""

import numpy as np
import numpy.typing as npt
from .trigonometry import arccsc, arcsec

# - OBSERVABLE: Var. of the difference of numbers of particles between both arms
# -------------------------------------------------------------------------------

# |- DETECTORS: perfects
# -----------------------


def ev_vhd_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the variance of the half difference of number of
    particles detected at both output ports of the interferometer.
    The detectors are perfect, meaning that their quantum efficiency is equal to 1.
    Since the expectation value of the difference itself is zero, the variance is
    actually equal to the expectation value of the square of the difference.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the two-mode squeezed vacuum
        state with N/2 particles per mode on average.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(1/4) * (N_output1 - N_output2) ** 2>
    """
    return (N / 2) * (1 + N / 2) * np.sin(phi) ** 2


def ev_vhd_squared_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the power four of the half difference of number
    of particles detected at both output ports of the interferometer.
    The detectors are perfect, meaning that their quantum efficiency is equal to 1.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the two-mode squeezed vacuum
        state with N/2 particles per mode on average.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(1/16) * (N_output1 - N_output2) ** 4>
    """
    return (
        (N / 2)
        * (1 + N / 2)
        * np.sin(phi) ** 2
        * (1 + 9 * N / 2 * (1 + N / 2) * np.sin(2 * phi) ** 2)
    )


def fluctuations_vhd_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
) -> float | npt.NDArray[np.float_]:
    """Returns the quantum fluctuations of the variance of the half difference of number
    of particles detected at both output ports of the interferometer.
    The detectors are perfect, meaning that their quantum efficiency is equal to 1.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the two-mode squeezed vacuum
        state with N/2 particles per mode on average.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <Sqrt[Var( Var{ (1/2) * (N_output1 - N_output2) } )]>
    """
    return np.sqrt(ev_vhd_squared_perfect_qe(phi, N) - ev_vhd_perfect_qe(phi, N) ** 2)


def phase_uncertainty_vhd_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the phase uncertainty during an interferometry experiment using
        - two-mode squeezed vacuum state with N/2 particles per mode on average;
        - perfect detectors (quantum efficiency equal to 1);
        - considering the variance of the half difference of particles detected at the
            output as the observable of interest;

    Notice that this resolution depends on the phase difference phi.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the two-mode squeezed vacuum
        state with N/2 particles per mode on average.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Phase uncertainty.
    """
    return np.sqrt(1 + 4 * N * (1 + N / 2) * np.sin(phi) ** 2) / (
        2 * np.cos(phi) * np.sqrt(N / 2 * (1 + N / 2))
    )


# |- DETECTORS: finite quantum efficiency
# ---------------------------------------

# ||- general shape


def ev_variance_difference_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: float | npt.NDArray[np.float_],
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
    N : float | npt.NDArray[np.float]
        Average total number of particles (twice the average number of particles per
        input mode).
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(N_output1 - N_output2) ** 2>
    """
    return (
        eta**2 * np.sin(phi) ** 2 * (N / 2) * (1 + N / 2)
        + 0.25 * eta * (1 - eta) * N / 2
    )


def ev_difference_quarted_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: float | npt.NDArray[np.float_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the power four of the difference of number of
    particles detected at both output arms of the interferometer.
    The detectors have a finite quantum efficiency eta.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : float | npt.NDArray[np.float]
        Average total number of particles (twice the average number of particles per
        input mode).
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(N_output1 - N_output2) ** 4>
    """
    nu = N / 2
    return (eta * nu / 8) * (
        1
        + 3 * eta
        + 16 * eta * nu
        + 12 * eta**2 * nu
        + 3 * eta**3 * nu
        + 36 * eta**2 * nu**2
        + 18 * eta**3 * nu**2
        + 27 * eta**3 * nu**3
        - 4
        * eta
        * (1 + nu)
        * (1 + 9 * eta * nu + 9 * eta**2 * nu**2)
        * np.cos(2 * phi)
        + 9 * eta**3 * nu * (1 + nu) ** 2 * np.cos(4 * phi)
    )


def ev_fourth_moment_difference_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: float | npt.NDArray[np.float_],
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
    N : float | npt.NDArray[np.float]
        Average total number of particles (twice the average number of particles per
        input mode).
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <Var(Var(N_output1 - N_output2))>
    """
    nu = N / 2
    return (eta * nu / 8) * (
        1
        + 3 * eta
        + 14 * eta * nu
        + 12 * eta**2 * nu
        + 2 * eta**3 * nu
        + 32 * eta**2 * nu**2
        + 16 * eta**3 * nu**2
        + 24 * eta**3 * nu**3
        - 4
        * eta
        * (1 + nu)
        * (1 + 8 * eta * nu + 8 * eta**2 * nu**2)
        * np.cos(2 * phi)
        + 8 * eta**3 * nu * (1 + nu) ** 2 * np.cos(4 * phi)
    )


def noise_difference_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: float | npt.NDArray[np.float_],
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
    N : float | npt.NDArray[np.float]
        Average total number of particles (twice the average number of particles per
        input mode).
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
    N: float | npt.NDArray[np.float_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the resolution of the phase estimation during an interferometry
    experiment using two-mode squeezed states, detectors with finite quantum efficiency
    eta and considering the variance of the difference of particles at the output as the
    observable of interest.
    Notice that this resolution depends on the phase difference phi.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : float | npt.NDArray[np.float]
        Average total number of particles (twice the average number of particles per
        input mode).
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Phase resolution.
    """
    nu = N / 2
    p0 = (
        1
        + 3 * eta
        + 14 * eta * nu
        + 12 * eta**2 * nu
        + 2 * eta**3 * nu
        + 32 * eta**2 * nu**2
        + 16 * eta**3 * nu**2
        + 24 * eta**3 * nu**3
    )

    p1 = -4 * eta * (1 + nu) * (1 + 8 * eta * nu + 8 * eta**2 * nu**2)

    p2 = 8 * eta**3 * nu * (1 + nu) ** 2

    return np.sqrt(eta * nu * (p0 + p1 * np.cos(2 * phi) + p2 * np.cos(4 * phi))) / (
        2 * np.sqrt(2) * eta**2 * nu * (1 + nu) * np.sin(2 * phi)
    )


# ||- behaviour at the optimal phase


def optimal_phi_difference(
    N: float | npt.NDArray[np.float_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the optimal phase to estimate (minimizing the resolution) during an
    interferometry experiment using two-mode squeezed states, detectors with finite
    quantum efficiency and considering the variance of the difference of particles at
    the output as the observable of interest. The detectors have a finite quantum
    efficiency eta.

    Parameters
    ----------
    N : float | npt.NDArray[np.float]
        Average total number of particles (twice the average number of particles per
        input mode).
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Optimal phase to estimate experimentally.
    """
    nu = N / 2
    return arccsc(
        np.sqrt(
            1
            / ((-1 + eta) * (-1 - 10 * eta * nu + 10 * eta**2 * nu))
            * (
                1
                - 20 * eta**2 * nu
                + 10 * eta**3 * nu
                + eta * (-1 + 10 * nu)
                + np.sqrt(
                    (-1 + eta)
                    * (
                        -1
                        - 7 * eta * (1 + 4 * nu)
                        - 4 * eta**2 * nu * (26 + 61 * nu)
                        + 20 * eta**5 * nu**2 * (5 + 32 * nu + 32 * nu**2)
                        - 4 * eta**3 * nu * (-15 + 81 * nu + 176 * nu**2)
                        + eta**4 * (340 * nu**2 - 640 * nu**4)
                    )
                )
            )
        )
    )


def resolution_at_optimal_phi_difference(
    N: float | npt.NDArray[np.float_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the resolution at the optimal phase to estimate during an interferometry
    experiment using two-mode squeezed states, detectors with finite quantum efficiency
    eta and considering the variance of the difference of particles at the output as the
    observable of interest.
    This function is therefore just:

    phase_resolution_difference_finite_qe(optimal_phi_difference(nu, eta), nu, eta)

    Parameters
    ----------
    N : float | npt.NDArray[np.float]
        Average total number of particles (twice the average number of particles per
        input mode).
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Optimal resolution.
    """
    nu = N / 2
    return (
        1
        / (2 * np.sqrt(2) * eta**2 * nu * (1 + nu))
        * np.sqrt(
            eta
            * nu
            * (
                1
                + eta
                * (
                    3
                    + 2
                    * nu
                    * (7 + eta * (6 + eta + 8 * (2 + eta) * nu + 12 * eta * nu**2))
                )
                + 4
                * eta
                * (1 + nu)
                * (1 + 8 * eta * nu * (1 + eta * nu))
                * np.cos(
                    2
                    * arcsec(
                        np.sqrt(
                            (
                                1
                                + eta * (-1 + 10 * (-1 + eta) ** 2 * nu)
                                + np.sqrt(
                                    (-1 + eta)
                                    * (-1 + 10 * (-1 + eta) * eta * nu)
                                    * (
                                        1
                                        + eta
                                        * (
                                            7
                                            + 2
                                            * nu
                                            * (
                                                9
                                                + eta
                                                * (
                                                    22
                                                    + 32 * nu
                                                    + eta * (5 + 32 * nu * (1 + nu))
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (1 - eta + 10 * (-1 + eta) ** 2 * eta * nu)
                        )
                    )
                )
                + 8
                * eta**3
                * nu
                * (1 + nu) ** 2
                * np.cos(
                    4
                    * arcsec(
                        np.sqrt(
                            (
                                1
                                + eta * (-1 + 10 * (-1 + eta) ** 2 * nu)
                                + np.sqrt(
                                    (-1 + eta)
                                    * (-1 + 10 * (-1 + eta) * eta * nu)
                                    * (
                                        1
                                        + eta
                                        * (
                                            7
                                            + 2
                                            * nu
                                            * (
                                                9
                                                + eta
                                                * (
                                                    22
                                                    + 32 * nu
                                                    + eta * (5 + 32 * nu * (1 + nu))
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                            / (1 - eta + 10 * (-1 + eta) ** 2 * eta * nu)
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
                        + eta * (-1 + 10 * (-1 + eta) ** 2 * nu)
                        + np.sqrt(
                            (-1 + eta)
                            * (-1 + 10 * (-1 + eta) * eta * nu)
                            * (
                                1
                                + eta
                                * (
                                    7
                                    + 2
                                    * nu
                                    * (
                                        9
                                        + eta
                                        * (
                                            22
                                            + 32 * nu
                                            + eta * (5 + 32 * nu * (1 + nu))
                                        )
                                    )
                                )
                            )
                        )
                    )
                    / (1 - eta + 10 * (-1 + eta) ** 2 * eta * nu)
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
    return ((2 / 5) ** (1 / 4) * np.sqrt(5 + 2 * np.sqrt(10))) * np.sqrt(1 - eta)
