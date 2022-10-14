import matplotlib.pyplot as plt
import numpy as np
from scipy.special import zeta

# constants (SI units)
c = 2.99792458e8
kB = 1.380649e-23
kB_ev = 8.617333262e-5
hbar = 1.054571817e-34
khc = kB / (hbar * c)
G = 6.67430e-11
mpc = 3.0856775815e19  # mpc to km
rho_c = 3750 / (np.pi * G) / mpc**2 * 1e-6  # h^2 kg/cm^3
Tcmb = 2.725
Tnu = Tcmb * (4 / 11) ** (1 / 3)
m_proton = 1.673e-27
omega_m = 0.32
omega_lambda = 0.68
h = 0.7


def n_boson(T, g):
    """
    Number density of relativistic bosons in cm^-3 (hence the 1e6 factor)
    """
    return g / np.pi**2 * zeta(3) * (khc * T) ** 3 / 1e6


def u_boson(T, g):
    """
    Energy density of relativistic bosons in J / cm^3
    """
    return np.pi**2 / 30 * g * (khc * T) ** 3 * kB * T / 1e6


def om_m_z(z, omega_m0):
    """
    Redshift evolution of omega matter
    """
    return omega_m0 * (z + 1) ** 3


def om_l_z(z, omega_l0):
    """
    Redshift evolution of omega lambda
    """
    return omega_l0 * np.ones_like(z)


def om_r_z(z, omega_r0):
    """
    Redshift evolution of omega radiation
    """
    return omega_r0 * (z + 1) ** 4


if __name__ == "__main__":
    # 1a
    n_cmb = n_boson(Tcmb, 2)
    print(f"Number density of CMB photons: {n_cmb:.4g} / cm³")

    # 1b
    u_cmb = u_boson(Tcmb, 2)
    print(f"Critical density: {rho_c:.4g} h² kg / cm³.")
    print(f"Energy density of CMB photons: {u_cmb:.4g} J / cm³")
    omega_gamma = u_cmb / (rho_c * c**2)
    print(f"Present photon density parameter: {omega_gamma:.4g}.")

    # 2a
    print(f"Temperature of neutrino background: {Tnu:.4g} K.")

    # 2d
    m_nu_rel = kB_ev * Tnu
    print(f"Neutrino mass required to be relativistic: {m_nu_rel:.4g} eV.")
    neutrino_factor = 1 + 7 / (2 ** (1 / 3) * 11 ** (4 / 3))
    print(f"{neutrino_factor=}")
    omega_rad = omega_gamma * neutrino_factor
    print(f"Present radiation density parameter: {omega_rad:.4g}.")

    # 3
    beta = n_cmb * m_proton / rho_c
    print(f"Beta = {beta:.4g}.")

    eta_min = 5.95e-10
    eta_max = 6.33e-10
    print(f"{eta_min*beta:.4g} <= omega_b h² <= {eta_max*beta:.4g}")

    # 4a
    zeq = omega_m / (omega_rad / h**2) - 1
    print(om_r_z(zeq, omega_rad / h**2) - om_m_z(zeq, omega_m))
    print(f"Matter-radiation equality redshift: {zeq:.4g}.")

    # 4b
    zeq_ml = 0.285641
    redshifts = np.geomspace(1, 1e7, num=100) - 1
    om_ms = om_m_z(redshifts, omega_m)
    om_ls = om_l_z(redshifts, omega_lambda)
    om_rs = om_r_z(redshifts, omega_rad / h**2)
    plt.figure()
    plt.plot(redshifts, om_ms, label="$\Omega_m$")
    plt.plot(redshifts, om_ls, label="$\Omega_\Lambda$")
    plt.plot(redshifts, om_rs, label="$\Omega_r$")
    plt.scatter(zeq, om_m_z(zeq, omega_m), c="k", zorder=20)
    plt.scatter(zeq_ml, om_m_z(zeq_ml, omega_m), c="k", zorder=10)
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.xlabel("$z$")
    plt.ylabel("Density [units of critical density]")
    plt.savefig("../ps5/q4b.eps", bbox_inches="tight")
