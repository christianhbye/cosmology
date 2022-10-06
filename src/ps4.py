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


ekte = t0_flat(0.3, 7.2e-4)
print(ekte)

HW_DIR = "../ps4/"

# t0 always scales with 1/h so if we set h=1, we effectively get 1/h units
omega_m = np.linspace(0.1, 3, num=100)
omega_0 = omega_m  # omega_lambda = 0
age = np.empty_like(omega_0)
for i, om0 in enumerate(omega_0):
    age[i] = t0_matter(om0, 1)  # age of universe

# 1a
plt.figure()
plt.plot(omega_m, age, label="$\\Omega_{0, \\Lambda}=0$")
plt.xlabel("$\Omega_{0, m}$")
plt.ylabel("$t_0 [h^{-1} \mathrm{Gyr}]$")
plt.xlim(omega_0.min(), omega_0.max())
plt.legend()
plt.savefig(HW_DIR + "q1a.eps", bbox_inches="tight")

# 1b
omega_m = np.linspace(0, 1, num=100)
age = t0_flat(omega_m, 1)
plt.plot(omega_m, age, label="$\\Omega_{0, \\Lambda}=1-\\Omega_{0, m} $")
plt.legend()
plt.savefig(HW_DIR + "q1b.eps", bbox_inches="tight")

