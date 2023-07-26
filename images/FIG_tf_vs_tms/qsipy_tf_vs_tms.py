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


width = cm_to_inch(15)
height = 0.7 * width

N = 100
phi = np.linspace(0, 0.2, 200)
eta = 0.95

perfect_TF = np.sqrt(N) * tfs.phase_uncertainty_vhd(phi, N)
perfect_TMS = np.sqrt(N) * tms.phase_uncertainty_vhd(phi, N)
realistic_TF = np.sqrt(eta * N) * tfs.phase_uncertainty_vhd(phi, N, eta)
realistic_TMS = np.sqrt(eta * N) * tms.phase_uncertainty_vhd(phi, N, eta)

phi0_tf = tfs.optimal_phi_vhd(N, eta)
res_phi0_tf = np.sqrt(eta * N) * tfs.phase_uncertainty_vhd(phi0_tf, N, eta)
phi0_tms = tms.optimal_phi_vhd(N, eta)
res_phi0_tms = np.sqrt(eta * N) * tms.phase_uncertainty_vhd(phi0_tms, N, eta)

fig, ax = plt.subplots(figsize=(width, height))

ax.semilogy(
    [phi0_tf, phi0_tf, 0.2],
    [0.08, res_phi0_tf, res_phi0_tf],
    ls="dashed",
    color="k",
    lw=0.8,
    alpha=0.75,
)
ax.semilogy(
    [phi0_tms, phi0_tms, 0.2],
    [0.08, res_phi0_tms, res_phi0_tms],
    ls="dashed",
    color="k",
    lw=0.8,
    alpha=0.7,
)

ax.semilogy(phi, np.ones_like(phi), label="SQL", lw=1.5, color="k", alpha=0.7)
ax.semilogy(phi, perfect_TF, label="", lw=2, color="tab:blue", ls=(0, (3, 1)))
ax.semilogy(phi, perfect_TMS, label="", lw=2, color=red, ls=(0, (3, 1)))
ax.semilogy(phi, realistic_TF, label="TF state", lw=2, color="tab:blue")
ax.semilogy(phi, realistic_TMS, label="TMS state", lw=2, color=red)


ax.set_xlim([0, 0.2])
ax.set_ylim((0.08, 3 * 10**0))
ax.set_xticks(
    [0, phi0_tms, phi0_tf, 0.1, 0.15, 0.2],
    ["0", "$\phi_0^{\mathrm{tms}}$", "$\phi_0^{\mathrm{tf}}$", "0.1", "0.15", "0.2"],
)
ax.set_yticks([0.1, 1, 2, 3], ["0.1", "1", "2", "3"])

ax.set_xlabel("$\phi$ (rad)")
ax.set_ylabel(
    "$\sqrt{\eta N} \, \Delta\phi$ (rad)",
    fontsize="small",
)

ax2 = ax.twinx()
ax2.set_yticks([5, 0, 20 * np.log10(res_phi0_tf), 20 * np.log10(res_phi0_tms), -20])
ax2.set_yticklabels(
    [
        "5",
        "0",
        f"{20 * np.log10(res_phi0_tf):.1f}",
        f"{20 * np.log10(res_phi0_tms):.1f}",
        "-20",
    ],
    fontsize=9,
)
ax2.set_ylim((20 * np.log10(0.08), 20 * np.log10(3 * 10**0)))
ax2.set_ylabel(r"$G$ (dB)", fontsize="x-small")


ax.legend()

plt.tight_layout()
fig.savefig("figures/qsipy_tf_vs_tms.png", bbox_inches="tight", pad_inches=0.1)
