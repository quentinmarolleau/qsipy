"""tfs.py
Provides analytical functions related to the phase uncertainty during an interferometry
experiment using twin-Fock states (i.e. |n,n>) as an input state.
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

# - OBSERVABLE: Variance of the difference of numbers of particles between both arms
# ----------------------------------------------------------------------------------

# |- DETECTORS: perfects
# ----------------------


def ev_vhd_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
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
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(1/4) * (N_output1 - N_output2) ** 2>
    """
    return 0.25 * N * (1 + N / 2) * np.sin(phi) ** 2


def ev_vhd_squared_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the power four of the half difference of number
    of particles detected at both output ports of the interferometer.
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
        Expectation value: <(1/16) * (N_output1 - N_output2) ** 4>
    """
    return (
        0.25
        * N
        * (1 + N / 2)
        * np.sin(phi) ** 2
        * (1 + 1.5 * (-1 + N / 4 + N**2 / 8) * np.sin(phi) ** 2)
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
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <Sqrt[Var( Var{ (1/2) * (N_output1 - N_output2) } )]>
    """
    return np.sqrt(ev_vhd_squared_perfect_qe(phi, N) - ev_vhd_perfect_qe(phi, N) ** 2)


def phase_uncertainty_vhd_perfect_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
) -> float | npt.NDArray[np.float_]:
    """Returns the phase uncertainty during an interferometry experiment using
        - twin-Fock states at the input;
        - perfect detectors (quantum efficiency equal to 1);
        - considering the variance of the half difference of particles detected at the
            output as the observable of interest;

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
        Phase uncertainty.
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


def ev_vhd_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the variance of the half difference of number of
    particles detected at both output ports of the interferometer.
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
        Expectation value: <(1/4) * (N_output1 - N_output2) ** 2>
    """
    return eta**2 * (N / 4) * (1 + N / 2) * np.sin(phi) ** 2 + eta * (1 - eta) * N / 4


def ev_vhd_squared_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the power four of the half difference of number
    of particles detected at both output ports of the interferometer.
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
        Expectation value: <(1/16) * (N_output1 - N_output2) ** 4>
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


def fluctuations_vhd_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the quantum fluctuations of the variance of the half difference of number
    of particles detected at both output ports of the interferometer.
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
        Expectation value: <Sqrt[Var( Var{ (1/2) * (N_output1 - N_output2) } )]>
    """
    return np.sqrt(
        ev_vhd_squared_finite_qe(phi, N, eta) - ev_vhd_finite_qe(phi, N, eta) ** 2
    )


def phase_uncertainty_vhd_finite_qe(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the phase uncertainty during an interferometry experiment using
        - twin-Fock states at the input;
        - detectors with finite quantum efficiency eta;
        - considering the variance of the half difference of particles detected at the
            output as the observable of interest;

    Notice that this resolution depends on the phase difference phi. Here (eta < 1),
    meaning that when (phi = 0), the phase uncertainty goes to infinity.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be strictly between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Phase uncertainty.
    """
    return np.divide(
        np.sqrt(
            N
            * eta
            * (
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
                * (N + 2)
                * eta
                * (16 + 16 * (-3 + N) * eta + (24 - 14 * N + N**2) * eta**2)
                * np.cos(2 * phi)
                + (-48 - 20 * N + 4 * N**2 + N**3) * eta**3 * np.cos(4 * phi)
            )
        ),
        4 * N * (N + 2) * eta**2 * np.abs(np.sin(2 * phi)),
        # we initialize the output with np.inf, and replace with the proper values where
        # phi!=0. There may be a cleaner way than doing "phi + N + eta" to initialize
        # an array with the correct shape, but the issue is that phi, N, eta can either
        # be scalars, or same-shape arrays....
        out=np.full_like(phi + N + eta, np.inf, dtype=np.float64),
        where=(phi != 0),
    )


# ||- agnostic functions


def ev_vhd(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_] = 1,
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the variance of the half difference of number of
    particles detected at both output ports of the interferometer.
    Since the expectation value of the difference itself is zero, the variance is
    actually equal to the expectation value of the square of the difference.

    This function only calls either "ev_vhd_perfect_qe" or "ev_vhd_finite_qe", depending
    on the value of eta that it is set as argument.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_], optional
        Quantum efficiency of the detector, must be between 0 and 1, by default 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(1/4) * (N_output1 - N_output2) ** 2>
    """
    return np.where(
        eta == 1,
        ev_vhd_perfect_qe(phi, N),
        ev_vhd_finite_qe(phi, N, eta),
    )


def ev_vhd_squared(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_] = 1,
) -> float | npt.NDArray[np.float_]:
    """Returns the expectation value of the power four of the half difference of number
    of particles detected at both output ports of the interferometer.

    This function only calls either "ev_vhd_squared_perfect_qe" or
    "ev_vhd_squared_finite_qe", depending on the value of eta that it is set as
    argument.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_], optional
        Quantum efficiency of the detector, must be between 0 and 1, by default 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <(1/16) * (N_output1 - N_output2) ** 4>
    """
    return np.where(
        eta == 1,
        ev_vhd_squared_perfect_qe(phi, N),
        ev_vhd_squared_finite_qe(phi, N, eta),
    )


def fluctuations_vhd(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_] = 1,
) -> float | npt.NDArray[np.float_]:
    """Returns the quantum fluctuations of the variance of the half difference of number
    of particles detected at both output ports of the interferometer.

    This function only calls either "fluctuations_vhd_perfect_qe" or
    "fluctuations_vhd_finite_qe", depending on the value of eta that it is set as
    argument.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_], optional
        Quantum efficiency of the detector, must be between 0 and 1, by default 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Expectation value: <Sqrt[Var( Var{ (1/2) * (N_output1 - N_output2) } )]>
    """
    return np.where(
        eta == 1,
        fluctuations_vhd_perfect_qe(phi, N),
        fluctuations_vhd_finite_qe(phi, N, eta),
    )


def phase_uncertainty_vhd(
    phi: float | npt.NDArray[np.float_],
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_] = 1,
) -> float | npt.NDArray[np.float_]:
    """Returns the phase uncertainty during an interferometry experiment using
        - twin-Fock states at the input;
        - considering the variance of the half difference of particles detected at the
            output as the observable of interest;

    This function only calls either "phase_uncertainty_vhd_perfect_qe" or
    "phase_uncertainty_vhd_finite_qe", depending on the value of eta that it is set as
    argument.

    Parameters
    ----------
    phi : float | npt.NDArray[np.float_]
        Phase difference between both arms of the interferometer.
    N : int | npt.NDArray[np.int_]
        Total number of particles. The input state being the twin-Fock |N/2,N/2>.
    eta : float | npt.NDArray[np.float_], optional
        Quantum efficiency of the detector, must be between 0 and 1, by default 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Phase uncertainty.
    """
    return np.where(
        eta == 1,
        phase_uncertainty_vhd_perfect_qe(phi, N),
        phase_uncertainty_vhd_finite_qe(phi, N, eta),
    )


# ||- behaviour at the optimal phase


def optimal_phi_vhd(
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_] = 1,
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
    eta : float | npt.NDArray[np.float_], optional
        Quantum efficiency of the detector, must be between 0 and 1, by default 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Optimal phase to estimate experimentally.
    """
    n = N / 2
    return np.where(
        eta == 1,
        0,
        arccsc(
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
                            + 2
                            * n
                            * (45 - 72 * n + 31 * n**2 - 2 * n**3)
                            * eta**4
                            + 2
                            * n
                            * (-9 + 24 * n - 15 * n**2 + 2 * n**3)
                            * eta**5
                        )
                    )
                )
                / ((-1 + eta) * (-1 + (6 - 4 * n) * eta + (-6 + 4 * n) * eta**2))
            )
        ),
    )


def phase_uncertainty_at_optimal_phi_vhd(
    N: int | npt.NDArray[np.int_],
    eta: float | npt.NDArray[np.float_] = 1,
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
    eta : float | npt.NDArray[np.float_], optional
        Quantum efficiency of the detector, must be between 0 and 1, by default 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Optimal resolution.
    """
    n = N / 2
    return np.where(
        eta == 1,
        (np.sqrt(2) / 2) * (1 / np.sqrt((N / 2) * (1 + N / 2))),
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
        ),
    )


def asymptotic_ratio_phase_uncertainty_to_SQL_at_optimal_phi_vhd(
    eta: float | npt.NDArray[np.float_],
) -> float | npt.NDArray[np.float_]:
    """Returns the asymptotic limit (as the number of particles goes to infinity) of the ratio between:
        - the phase uncertainty at the optimal phase and considering the variance of the half difference
        of particles at the output as the observable of interest
        - the SQL 1/sqrt(eta N).

    It only depends on the quantum efficiency of the detectors.

    Parameters
    ----------
    eta : float | npt.NDArray[np.float_]
        Quantum efficiency of the detector, must be between 0 and 1.

    Returns
    -------
    float | npt.NDArray[np.float_]
        Optimal phase uncertainty to SQL ratio in the asymptotic limit of N.
    """
    return np.sqrt(3) * np.sqrt(1 - eta)
