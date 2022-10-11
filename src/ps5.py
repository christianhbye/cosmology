import numpy as np
from scipy.special import zeta

# constants (SI units)
c = 2.99792458e8
kB = 1.380649e-23
hbar = 1.054571817e-34
khc = kB / (hbar * c)
G = 6.67430e-11
mpc = 3.0856775815e19  # mpc to km
rho_c = 3750 / (np.pi * G) / mpc**2 * 1e-6 # h^2 kg/cm^3
Tcmb = 2.725
Tnu = Tcmb * (4/11)**(1/3)
m_proton = 1.673e-27

def n_boson(T, g):
    """
    Number density of relativistic bosons in cm^-3 (hence the 1e6 factor)
    """
    return g / np.pi**2 * zeta(3) * (khc * T)**3 / 1e6

def u_boson(T, g):
    """
    Energy density of relativistic bosons in J / cm^3
    """
    return np.pi**2 / 30 * g * (khc * T)**3 * kB * T / 1e6

if __name__=="__main__":
    # 1a
    n_cmb = n_boson(Tcmb, 2)
    print(f"Number density of CMB photons: {n_cmb:.4g} / cm³")

    # 1b
    u_cmb = u_boson(Tcmb, 2)
    print(f"Critical density: {rho_c:.4g} h² kg / cm³.")
    print(f"Energy density of CMB photons: {u_cmb:.4g} J / cm³")
    omega_gamma = u_cmb / (rho_c * c**2)
    print(f"Present radiation density parameter: {omega_gamma:.4g}.")

    # 2a
    print(f"Temperature of neutrino background: {Tnu:.4g} K.")

    # 3
    beta = n_cmb * m_proton / rho_c
    print(f"Beta = {beta:.4g}.")

    eta_min = 5.95e-10
    eta_max = 6.33e-10
    print(f"{eta_min*beta:.4g} <= omega_b h² <= {eta_max*beta:.4g}")
