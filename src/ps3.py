import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

c = 299792.458  # km/s


def ld_flat_matter(z, h=0.7):
    """
    Luminosity distance for a flat, matter dominated universe
    """
    H0 = 100 * h
    sq = np.sqrt(1 + z)
    return 2 * c / H0 * sq * (sq - 1)


def ld_open_matter(z, om_m, h=0.7):
    """
    Luminosity distance for an open, matter dominated universe
    """
    H0 = 100 * h
    om_k = 1 - om_m
    sq = np.sqrt(om_m * (1 + z) / om_k + 1)
    frac = (sq - 1) / (sq + 1)
    return (1 + z) * c / (H0 * np.sqrt(om_k)) * frac


def ld_flat_int(z, om_m, h=0.7):
    """
    Integrand for comoving distance of flat universe with Lambda
    """
    om_l = 1 - om_m
    return 1 / np.sqrt(om_m * (1 + z) ** 3 + om_l)


def ld_flat_lambda(z, om_m, h=0.7):
    """
    Luminoisty distance for a flat universe with a cosmolgical constant
    """
    H0 = 100 * h
    om_l = 1 - om_m
    z = np.array(z).ravel()
    integrals = np.empty(z.size)
    for i in range(z.size):
        integrals[i] = quad(ld_flat_int, 0, z[i], args=(om_m))[0]
    return (1 + z) * c / H0 * integrals


def lum_dist(z, om_m, om_l, h=0.7):
    """
    Compute the luminosity distance at a redshift z, given the present values
    of the density parameters Omega_m and Omega_Lambda and of the Hubble
    parameter h, where h is defined such that H0 = 100h.
    """
    if om_l + om_m > 1:  # closed
        raise NotImplementedError
    if om_l > 0 and (om_l + om_m < 1):  # open with Lambda
        raise NotImplementedError
    if np.isclose(om_m + om_l, 1):  # flat
        if np.isclose(om_l, 0):  # matter dominated
            return ld_flat_matter(z, h=h)
        else:
            return ld_flat_lambda(z, om_m, h=h)
    else:
        return ld_open_matter(z, om_m, h=h)


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
    omegas = [[0.27, 0.73], [1.0, 0.0], [0.27, 0.0]]
    z_grid = np.linspace(0.8 * z.min(), 1.2 * z.max(), num=200)
    plt.figure(figsize=(6.0, 3.0))
    plt.xlabel("$z$")
    plt.ylabel("$\\mu$")
    plt.xlim(z_grid.min(), z_grid.max())
    plt.ylim(34, 46)
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
