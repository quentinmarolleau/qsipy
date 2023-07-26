import numpy as np
from qsipy import tfs, tms
from matplotlib import pyplot as plt

from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["ECGaramond"]})
rc("text", usetex=True)

plt.rc("legend", fontsize=7.5)

red = "#B00028"


def cm_to_inch(value):
    return value / 2.54


def linear_to_db(x):
    return 20 * np.log10(x)


def db_to_linear(x):
    return 10 ** (x / 20)


width = cm_to_inch(15)
height = 0.6 * width


N_exp = np.linspace(np.log10(2), np.log10(1000), 200)
N = 10**N_exp
eta = 0.95

res_tf = np.sqrt(eta * N) * tfs.phase_uncertainty_at_optimal_phi_vhd(N, eta)
res_tms = np.sqrt(eta * N) * tms.phase_uncertainty_at_optimal_phi_vhd(N, eta)
res_tf_perfect = np.sqrt(eta * N) * tfs.phase_uncertainty_at_optimal_phi_vhd(N)
res_tms_perfect = np.sqrt(eta * N) * tms.phase_uncertainty_at_optimal_phi_vhd(N)

fig, ax = plt.subplots(figsize=(width, height))

ax.plot(N, res_tf, label="TF state", lw=1.5, color="tab:blue")
ax.plot(N, res_tms, label="TMS state", lw=1.5, color=red)
ax.plot(
    N,
    tfs.asymptotic_ratio_phase_uncertainty_to_SQL_at_optimal_phi_vhd(eta)
    * np.ones_like(N),
    color="tab:blue",
    lw=1.5,
    linestyle=(0, (0.1, 2)),
    dash_capstyle="round",
)
ax.plot(
    N,
    tms.asymptotic_ratio_phase_uncertainty_to_SQL_at_optimal_phi_vhd(eta)
    * np.ones_like(N),
    color=red,
    lw=1.5,
    linestyle=(0, (0.1, 2)),
    dash_capstyle="round",
)

ax.set_xscale("log")

ax.set_xlabel("$N$")
ax.set_ylabel("$\sqrt{\eta N} \, \Delta\phi_0$ (rad)")

secax_y = ax.secondary_yaxis("right", functions=(linear_to_db, db_to_linear))
secax_y.set_ylabel(r"$G$ (dB)", fontsize="small")

ax.legend(loc="upper right")

plt.tight_layout()

fig.savefig(
    "figures/qsipy_asymptotic_behaviour.png",
    bbox_inches="tight",
    pad_inches=0.1,
)
