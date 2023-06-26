import numpy as np
import pytest
from qsipy import tfs


def relative_deviation(array_tested, array_tabulated):
    return np.where(
        array_tabulated != 0,
        np.abs(array_tested - array_tabulated) / array_tabulated,
        np.abs(array_tested - array_tabulated),
    )


def test_ev_variance_difference_perfect_qe():
    N_equal_2 = np.array(
        [
            0.0,
            0.00996671107937919,
            0.0394695029985575,
            0.0873321925451609,
            0.151646645326417,
            0.22984884706593,
            0.318821122761663,
            0.41501642854988,
            0.514599761150644,
            0.613601047346544,
            0.708073418273571,
        ]
    )
    check = (
        relative_deviation(
            array_tested=tfs.ev_variance_difference_perfect_qe(
                np.linspace(0, 1, 11), 2
            ),
            array_tabulated=N_equal_2,
        )
        < 1e-5
    )

    check = check.all()
    assert check
