"""Numpy does not incorporate the secant, cosecant functions, as well as their
inverse functions. We do implement it here.

We remind that:
sec = 1/cos
scs = 1/sin
"""

import numpy as np
import numpy.typing as npt


def arccsc(x: float | npt.NDArray[np.float_]) -> float | npt.NDArray[np.float_]:
    return np.arcsin(1 / x)


def arcsec(x: float | npt.NDArray[np.float_]) -> float | npt.NDArray[np.float_]:
    return np.arccos(1 / x)
