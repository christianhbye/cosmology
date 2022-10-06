import matplotlib.pyplot as plt
import numpy as np


c = 299792.458  # km/s


def lum_dist(z, om_m, om_l, h=0.7):
    """
    Compute the luminosity distance at a redshift z, given the present values
    of the density parameters Omega_m and Omega_Lambda and of the Hubble
    parameter h, where h is defined such that H0 = 100h.
    """
    H0 = 100 * h  # in km/Mpc/s
    q0 = om_m / 2 - om_l  # deceleration parameter
    r = c / H0 * (z - (1 + q0) * z**2 / 2)  # small-z approximation
    return r * (1 + z)  # lum dist is related to comoving by (1+z)


def dm(dL):
    """
    Compute the distance modulus given the luminoisty distance dL in Mpc.
    """
    return 5 * (np.log10(dL) + 5)  # dL in Mpc so 10pc = 1e-5 Mpc


if __name__ == "__main__":
    HW_DIR = "../ps3/"
    z, mu, sig = np.loadtxt(
        HW_DIR + "hstsn.data", skiprows=5, usecols=(1, 2, 3)
    ).T
    omegas = [[0.27, 0.73], [0.27, 0.0], [1.0, 0.0]]
    z_grid = np.linspace(0.8 * z.min(), 1.2 * z.max(), num=200)
    plt.figure(figsize=(6.0, 3.0))
    plt.xlabel("$z$")
    plt.ylabel("$\\mu$")
    plt.xlim(z_grid.min(), z_grid.max())
    for i in range(len(omegas)):
        om_m, om_l = omegas[i]
        dms = dm(lum_dist(z_grid, om_m, om_l))
        label = "$(\\Omega_{0, m}, \\Omega_{0, \Lambda}) =$"
        label += f"({om_m:.2f}, {om_l:.2f})"
        plt.plot(z_grid, dms, label=label)
    plt.legend()
    plt.savefig(HW_DIR + "2a.eps", bbox_inches="tight")

    plt.errorbar(
        z,
        mu,
        yerr=sig,
        fmt="none",
        capsize=2,
        label="Riess et al (2007)",
        c="k",
    )
    plt.legend()
    plt.savefig(HW_DIR + "2b.eps", bbox_inches="tight")
