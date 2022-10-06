import matplotlib.pyplot as plt
import numpy as np


def a(x, Omega0, h):
    """
    Compute the scale factor for a matter-dominated universe. This
    is computed at the parameter x, which is related to time t by a parametric
    equation of sin or sinh depending on whether the universe is open or
    closed. For a flat universe, x = t.

    The function also returns the corresponding time t, given x.

    Note that h must be in inverse time units.
    x is a dimensionless (except in the flat case) parameter, with x >= 0.
    """
    H0 = 100 * h
    if Omega0 == 1:  # flat
        t = x
        a = (3 * H0 * t / 2) ** (2 / 3)
    else:  # open or closed
        b = 1 / 2 * Omega0 / np.abs(1 - Omega0)
        t = 1 / H0 * b / np.sqrt(np.abs(1 - Omega0))
        if Omega0 > 1:  # closed
            t *= x - np.sin(x)
            a = b * (1 - np.cos(x))
        else:  # open
            t *= np.sinh(x) - x
            a = b * (np.cosh(x) - 1)
    return t, a


def t0(Omega0, h):
    """
    Compute the age of a matter-dominated universe given Omega0 and h.
    The units of the age is inverse the units of H0.

    These equations were derived in PS2, question 1e.
    """
    H0 = 100 * h
    if Omega0 == 1:  # flat
        return 2 / (3 * H0)

    # in the closed or open case we need to find x0 first
    y = 1 + 2 * (1 - Omega0) / Omega0
    if Omega0 > 1:  # closed
        x0 = np.arccos(y)
    else:  # open
        x0 = np.arccosh(y)

    t0, a0 = a(x0, Omega0, H0)
    assert np.isclose(a0, 1)  # check that we indeed found the right t0
    return t0


if __name__ == "__main__":
    HW_DIR = "../ps2/"
    h = 7.2e-4  # inverse Giga years

    # flat
    omega = 1
    t0_flat = t0(omega, h)
    x = np.linspace(0, 50 + t0_flat, num=100)  # x = t, Gyrs
    t_flat, a_flat = a(x, omega, h)

    # closed
    omega = 3.0
    t0_closed = t0(omega, h)
    x = np.linspace(0, 2 * np.pi, num=100)  # from Big Bang to Big Crunch
    t_closed, a_closed = a(x, omega, h)

    # open
    omega = 0.3
    t0_open = t0(omega, h)
    x = np.linspace(0, 3.8, num=100)
    t_open, a_open = a(x, 0.3, h)

    # shift time axis by subtracting t0 from t
    t_flat -= t0_flat
    t_closed -= t0_closed
    t_open -= t0_open

    plt.figure()
    plt.plot(t_flat, a_flat, label="$\\Omega_0 = 1$")
    plt.plot(t_closed, a_closed, label="$\\Omega_0 = 3.0$")
    plt.plot(t_open, a_open, label="$\\Omega_0 = 0.3$")
    plt.scatter(-t0_flat, 0, c="red", label="Big Bang")
    plt.scatter(-t0_closed, 0, c="red")
    plt.scatter(-t0_open, 0, c="red")
    plt.scatter(
        (t_closed.max() - t0_closed) / 2,
        a_closed.max(),
        c="black",
        label="Maximum Expansion",
    )
    plt.scatter(t_closed.max(), 0, c="blue", label="Big Crunch")
    plt.legend()
    plt.xlabel("$t-t_0$ [Gyrs]")
    plt.ylabel("$a$")
    plt.xlim(-20, 50)
    plt.ylim(-0.1, a_open.max())
    plt.grid()
    plt.savefig(HW_DIR + "ps2_q1f.eps")
