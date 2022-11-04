import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

def theta(a, omega0):
    a = np.array(a).reshape(-1, 1)
    omega0 = np.array(omega0).reshape(1, -1)
    x = 1 + 2 * a * (1-omega0) / omega0
    return np.arccosh(x)

def delta_prop(theta):
    """
    Function of theta that delta is proportional to
    """
    d = 3 * np.sinh(theta) * (np.sinh(theta) - theta)
    d /= (np.cosh(theta) - 1)**2
    return d - 2

def open_delta(theta, theta_ref):
    return 1e-3 * delta_prop(theta) / delta_prop(theta_ref)

def integrand(a, omega_m, omega_l):
    """
    Function to integrate in Eq 46
    """
    return (omega_m / a + a **2 * omega_l)**(-3/2)

def f(a, omega_m):
    """
    The function f as defined in Eq 46
    """
    omega_l = 1 - omega_m
    a = np.array(a).ravel()
    integral = np.empty_like(a)
    for i in range(a.size):
        integral[i] = quad(integrand, 0, a[i], args=(omega_m, omega_l))[0]
    return np.sqrt(omega_m/a**3 + omega_l) * integral

def delta_lambda(a, omega_m):
    deltas = np.empty((omega_m.size, a.size))
    for i in range(omega_m.size):
        om_m = omega_m[i]
        f_ref = f(1e-3, om_m)
        deltas[i] = 1e-3 * f(a, om_m) / f_ref
    return deltas

if __name__=="__main__":
    a = np.geomspace(1e-3, 1, num=100)
    om_open = np.array([1e-2, .1, .3])[::-1]
    th = theta(a, om_open)
    th_ref = th[0]  # at a = 1e-3
    deltas = open_delta(th, th_ref)
    plt.figure(figsize=(10, 5))
    plt.plot(a, a, label="$\\Omega_{0, m} = 1.0$")
    for i in range(om_open.size):
        plt.plot(a, deltas[:, i], label=f"$\\Omega_{{0, m}} = {om_open[i]}$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlabel("$a$")
    plt.ylabel("$\delta$")
    plt.savefig("../ps7/3a.eps", bbox_inches="tight")

    omega_m = np.array([0.1, 0.3])[::-1]
    deltas_de = delta_lambda(a, omega_m)
    omega_l = 1 - omega_m
    for i in range(omega_m.size):
        label = f"$\\Omega_{{0, m}} = {omega_m[i]},$ "
        label += f"$\\Omega_{{0, \Lambda}} = {omega_l[i]}$"
        plt.plot(a, deltas_de[i], label=label)
    plt.legend()
    plt.savefig("../ps7/3b.eps", bbox_inches="tight")
