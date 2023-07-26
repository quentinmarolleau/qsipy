import numpy as np
from qsipy import tfs, tms
from matplotlib import pyplot as plt

from matplotlib import rc
import matplotlib.colors as colors

rc("font", **{"family": "serif", "serif": ["ECGaramond"]})
rc("text", usetex=True)

red = "#B00028"
plt.rc("legend", fontsize=7.5)
plt.rcParams.update({"hatch.color": red})


def cm_to_inch(value):
    return value / 2.54


width = cm_to_inch(20)
height = 0.4 * width


eta = np.linspace(0.5, 0.99999, 500)
N = np.geomspace(2, 1000, 500)

# making the grid
NN, etaa = np.meshgrid(N, eta)

# PART RELATED TO QSIPY FUNCTIONS
# (the rest is only plotting and decoration)
# --------------------------------------------------------------------------------------
phi_tfs = tfs.optimal_phi_vhd(NN, etaa)
res_tfs = tfs.phase_uncertainty_at_optimal_phi_vhd(NN, etaa) * np.sqrt(etaa * NN)

phi_tms = tms.optimal_phi_vhd(NN, etaa)
res_tms = tms.phase_uncertainty_at_optimal_phi_vhd(NN, etaa) * np.sqrt(etaa * NN)
# --------------------------------------------------------------------------------------

cbar_min = min(np.min(phi_tms), np.min(phi_tms))
cbar_max = max(np.max(phi_tms), np.max(phi_tms))

fig, axs = plt.subplots(
    1, 2, sharey="row", figsize=(width, height), constrained_layout=True
)

im = axs[0].pcolormesh(
    etaa,
    NN,
    phi_tfs,
    norm=colors.LogNorm(vmin=cbar_min, vmax=cbar_max),
    shading="auto",
    linewidth=0,
    rasterized=True,
)

contour_sql = axs[0].contourf(
    etaa,
    NN,
    res_tfs,
    [1, 3],
    hatches=["//"],
    colors="none",
)
contour_sql = axs[0].contour(
    etaa,
    NN,
    res_tfs,
    [1],
    linewidths=2,
    colors=red,
)
contours = axs[0].contour(
    etaa,
    NN,
    phi_tfs,
    levels=[0.025, 0.05, 0.1, 0.2, 0.35],
    colors="black",
    linewidths=1.2,
)

xy = [
    (0.98, 190),
    (0.9449675530947794, 93.82878021702504),
    (0.8813556913827655, 51.432385283750364),
    (0.8140227054108216, 18.910097909504202),
    (0.7034042284569137, 7.720602458750363),
]
labels_tfs = axs[0].clabel(contours, fontsize=10, inline_spacing=3, manual=xy)

axs[0].set_yscale("log")
axs[0].set_ylabel("$N$")
axs[0].set_xlabel("$\eta$")
axs[0].text(
    0.95,
    500,
    "TF",
    bbox={"facecolor": "white", "pad": 0.2, "boxstyle": "round", "alpha": 0.5},
)

im = axs[1].pcolormesh(
    etaa,
    NN,
    phi_tms,
    norm=colors.LogNorm(vmin=cbar_min, vmax=cbar_max),
    shading="auto",
    linewidth=0,
    rasterized=True,
)

contour_sql = axs[1].contourf(
    etaa,
    NN,
    res_tms,
    [1, 3],
    hatches=["//"],
    colors="none",
)
contour_sql = axs[1].contour(
    etaa,
    NN,
    res_tms,
    [1],
    linewidths=2,
    colors=red,
)
contours = axs[1].contour(
    etaa,
    NN,
    phi_tms,
    [0.01, 0.025, 0.05, 0.1, 0.2],
    colors="black",
    linewidths=1.2,
)

xy = [
    (0.98, 190),
    (0.9210915176301411, 107.6049879639371),
    (0.8438175587855326, 57.01455901083921),
    (0.7605158316633267, 22.9802416889043),
    (0.6683333066132264, 7.540068568023514),
]
labels_tms = axs[1].clabel(contours, fontsize=10, inline_spacing=3, manual=xy)

axs[1].set_yscale("log")

axs[1].set_xlabel("$\eta$")
axs[1].text(
    0.93,
    500,
    "TMS",
    bbox={"facecolor": "white", "pad": 0.2, "boxstyle": "round", "alpha": 0.5},
)
cb = fig.colorbar(im, ax=axs.flat, shrink=1, location="right")
cb.set_label("$\phi_0$ (rad)")

fig.savefig("figures/qsipy_optimal_phi.png", bbox_inches="tight", pad_inches=0.1)
