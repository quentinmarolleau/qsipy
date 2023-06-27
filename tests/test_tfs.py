import numpy as np
import pytest
from qsipy import tfs


def relative_deviation(array_tested, array_tabulated):
    return np.abs(array_tested - array_tabulated) / array_tabulated


def test_ev_variance_difference_perfect_qe(RELATIVE_DIFF=1e-14):
    # check value at zero
    assert tfs.ev_variance_difference_perfect_qe(0, 2) == 0
    assert tfs.ev_variance_difference_perfect_qe(0, 10) == 0
    assert tfs.ev_variance_difference_perfect_qe(0, 100) == 0

    # check nonzero values
    phi_inputs = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
    tabulated_N_equal_2 = np.array(
        [
            0.0000999966667111108,
            0.00249791736098712,
            0.00996671107937919,
            0.22984884706593,
            0.708073418273571,
        ]
    )
    tabulated_N_equal_100 = np.array(
        [
            0.127495750056666,
            3.18484463525857,
            12.7075566262085,
            293.057280009061,
            902.793608298803,
        ]
    )

    check = (
        relative_deviation(
            array_tested=tfs.ev_variance_difference_perfect_qe(phi_inputs, 2),
            array_tabulated=tabulated_N_equal_2,
        )
        < RELATIVE_DIFF
    )
    check = check.all()
    assert check

    check = (
        relative_deviation(
            array_tested=tfs.ev_variance_difference_perfect_qe(phi_inputs, 100),
            array_tabulated=tabulated_N_equal_100,
        )
        < RELATIVE_DIFF
    )
    check = check.all()
    assert check
