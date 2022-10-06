import matplotlib.pyplot as plt
import numpy as np


def omega(a, omega_m, omega_lam, omega_r):
    """
    Density parameter vs scale factor as a function of the present values of
    the matter (m), cosmological constant (lam) and radiation components.
    """
    omega_0 = omega_m + omega_lam
    x = 1 - omega_0
    x /= 1 - omega_0 + omega_lam * a**2 + omega_m / a + omega_r / a**2
    return 1 - x


if __name__ == "__main__":
    print("Values at a = 1e-3 (matter dominated)")
    om_m = [0.32, 0.32, 1.0, 5.0]
    om_l = [0.0, 0.68, 0.0, 0.0]
    a = np.geomspace(1e-3, 1, num=500)
    ls = ["-", "-", "--", "-"]
    plt.figure()
    for i in range(len(om_m)):
        print(f"omega_m, omega_lambda = {om_m[i], om_l[i]}")
        om = omega(a, om_m[i], om_l[i], 0)
        print(f"Omega(a=1e-3) = {om[0]}")
        label = "$(\\Omega_{0, m}, \\Omega_{0, \\Lambda}) = $"
        label += f"{om_m[i], om_l[i]}"
        plt.plot(a, om, label=label, ls=ls[i])
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("a")
    plt.ylabel("$\\Omega (a)$")
    plt.show()
