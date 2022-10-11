import matplotlib.pyplot as plt
import numpy as np

from ps2 import t0 as t0_matter

def t0_flat(omega_m, h):
    """
    Compute t0 for a flat universe given omega_m and h. Since the universe
    is flat, omega_lambda = 1 - omega_m.
    """
    omega_l = 1 - omega_m
    H0 = 100 * h
    log_arg = (1 + np.sqrt(omega_l)) / np.sqrt(omega_m) 
    t = 2 / (3 * H0 * np.sqrt(omega_l)) * np.log(log_arg)
    return t

def H0_flat(omega_m, t0):
    """
    Invert t0(omega_m, H0) to solve for H0. This function exploits that t0 and
    H0 are inversely proportional in the flat universe - hence the inverse of
    the t0_flat function can be obtained by swapping the roles of t0 and H0
    """
    h = t0_flat(omega_m, t0)
    H0 = 100 * h
    return H0

def t_vs_z(z, omega_m, omega_lambda, h=7.2e-4):
    """
    Compute time corresponding to redshift z.
    """
    H0 = 100 * h
    a = 1 / (1 + z)
    if omega_m == 1:  # flat, matter dominated
        return 2 / (3 * H0) * a ** (3/2)
    elif omega_lambda + omega_m == 1:  # flat with lambda
        u = np.sqrt(omega_m / a**3 + omega_lambda)
        log_arg = u + np.sqrt(omega_lambda)
        log_arg /= u - np.sqrt(omega_lambda)
        return np.log(log_arg) / (3 * H0 * np.sqrt(omega_lambda))
    elif omega_lambda == 0 and omega_m < 1:  # open, matter dominated
        coshx = 1 + 2 * a * (1 - omega_m) / omega_m
        sinhx = np.sqrt(coshx ** 2 - 1)
        x = np.arccosh(coshx)
        t = 1 / (2 * H0) * omega_m / (1-omega_m)**(3/2) * (sinhx - x)
        return t
    else:
        raise NotImplementedError


if __name__=="__main__":
    HW_DIR = "../ps4/"

    # t0 always scales with 1/h so if we set h=1, we effectively get 1/h units
    omega_m = np.linspace(1e-6, 3, num=100)
    omega_m.sort()
    age = np.empty_like(omega_m)
    for i, om0 in enumerate(omega_m):
        age[i] = t0_matter(om0, 1)  # age of universe

    # 1a
    plt.figure()
    plt.plot(omega_m, age, label="$\\Omega_{0, \\Lambda}=0$")
    plt.xlabel("$\Omega_{0, m}$")
    plt.ylabel("$t_0 [h^{-1} \mathrm{Gyr}]$")
    plt.xlim(omega_m.min(), omega_m.max())
    plt.legend()
    plt.grid()
    plt.savefig(HW_DIR + "q1a.eps", bbox_inches="tight")

    # 1b
    omega_m = np.linspace(1e-6, 1-1e-6, num=100)
    age = t0_flat(omega_m, 1)
    plt.plot(omega_m, age, label="$\\Omega_{0, \\Lambda}=1-\\Omega_{0, m} $")
    plt.legend()
    plt.savefig(HW_DIR + "q1b.eps", bbox_inches="tight")

    # 1c
    t0_vals = np.array([11, 13.8, 18])  # Gyr
    H0_vals = H0_flat(omega_m[None, :], t0_vals[:, None])
    # convert H0 from 1/Gyr to km/s/Mpc
    H0_vals /= 1e9 * 3.154e7  # 1/Gyr -> 1/s
    H0_vals *= 3.086e19  # convert to km/Mpc

    # H0 for matter dominated universe with t0 >11.5 Gyr
    H0_max_matter = 2 / (3*11.5)
    H0_max_matter /= 1e9 * 3.154e7
    H0_max_matter *= 3.086e19
    print(H0_max_matter)

    plt.figure()
    for i in range(len(t0_vals)):
        plt.plot(omega_m, H0_vals[i], label=f"$t_0 = {t0_vals[i]:.1f}$ Gyr") 
    plt.axhline(70, ls="--", c="k")
    plt.fill_between(
        omega_m, H0_vals[0], color="lightsteelblue", label="$t_0 \geq 11$ Gyr"
    )
    plt.fill_between(
        omega_m,
        67,
        73,
        color="lightpink",
        label="$H_0 = (70 \pm 3) \mathrm{km s^{-1} Mpc^{-1}}$",
    )
    plt.xlabel("$\Omega_{0, m}$")
    plt.ylabel("$H_0 [\mathrm{km s}^{-1} \mathrm{Mpc}^{-1}]$")
    plt.legend()
    plt.grid()
    plt.xlim(0, 1)
    plt.ylim(0, 300)
    plt.yticks([0, 70, 100, 200, 300])
    plt.savefig(HW_DIR + "q1c.eps", bbox_inches="tight")

    # 2
    z_cmb = 1100
    z_gal = 13
    om_ms = [0.32, 0.32, 1.]
    om_ls = [0.68, 0., 0.]
    for z in [z_cmb, z_gal]:
        for i in range(3):
            om = om_ms[i]
            ol = om_ls[i]
            print(om, ol, z)
            print(t_vs_z(z, om, ol))


    # 3
    def flat_th(z, L=20):
        y = 1.2e-3 * (1+z)
        y /= 2 - 2/np.sqrt(1+z)
        return y * L

    def integral(z, omega_m):
        a = np.sqrt(np.abs(omega_m - 1))
        y = 2 / a
        u = np.sqrt(omega_m * (1+z) + (1-omega_m))
        y *= np.arctan2(u, a) - np.arctan2(1, a)
        return y

    def open_th(z, om_m=0.3, L=20):
        y = 1.2e-3 * (1+z) * np.sqrt(1-om_m) * L
        x = np.sqrt(1-om_m) * integral(z, om_m)
        y /= np.sinh(x)
        return y
    
    def closed_th(z, om_m=3., L=20):
        y = 1.2e-3 * (1+z) * np.sqrt(om_m-1) * L
        x = np.sqrt(om_m-1) * integral(z, om_m)
        y /= np.sin(x)
        return y

    z = np.geomspace(1e-2, 10)

    plt.figure()
    plt.plot(z, flat_th(z), label="$\Omega_{0, m}=1$")
    plt.plot(z, open_th(z), label="$\Omega_{0, m}=0.3$")
    plt.plot(z, closed_th(z), label="$\Omega_{0, m}=3.$")
    plt.xlabel("$z$")
    plt.ylabel("$\\theta$ [h arcsec]")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig("q3b.eps", bbox_inches="tight")
