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
    omega_m = np.linspace(1e-6, 1, num=100)
    age = t0_flat(omega_m, 1)
    plt.plot(omega_m, age, label="$\\Omega_{0, \\Lambda}=1-\\Omega_{0, m} $")
    plt.legend()
    plt.savefig(HW_DIR + "q1b.eps", bbox_inches="tight")
