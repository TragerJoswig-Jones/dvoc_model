import numpy as np


def calculate_power(v, i):
    p = 1.5 * (v.alpha * i.alpha + v.beta * i.beta)
    q = 1.5 * (v.beta * i.alpha - v.alpha * i.beta)
    return p, q


def calculate_current(v, p, q):
    i_alpha = 2 * (p * v.alpha + q * v.beta) / (v.alpha ** 2 + v.beta ** 2) / 3
    i_beta = 2 * (p * v.beta - q * v.alpha) / (v.alpha ** 2 + v.beta ** 2) / 3
    return i_alpha, i_beta


def angle_diff(theta1, theta2, deg=False):
    delta = theta1 - theta2
    if deg:
        try:
            delta = [-360. + d if d > 180. else d for d in delta]
            delta = [360. + d if d < -180. else d for d in delta]
        except TypeError:
            if delta > 180:
                delta = -360. + delta
            if delta < -180:
                delta = 360. + delta
    else:
        try:
            delta = [2*np.pi - d if d > np.pi else d for d in delta]
        except TypeError:
            if delta > np.pi:
                delta = 2 * np.pi - delta
    return delta


def power_flow_2_bus(v1, theta1, v2, theta2, g, b):
    """ Calculate the power flow from bus 1 to bus 2 across the line with admittance g+jb
        from the given voltages and angles.
    """
    p = 0
    q = 0

    # Build admittance matrix values
    g12, b12 = -g, -b
    g11, b11 = g, b

    p += v1 * v2 * (g12 * np.cos(theta1 - theta2) + b12 * np.sin(theta1 - theta2))
    q += v1 * v2 * (g12 * np.sin(theta1 - theta2) - b12 * np.cos(theta1 - theta2))

    p += v1 * v1 *   g11
    q += v1 * v1 * - b11
    return p, q

if __name__ == "__main__":
    from numpy import testing
    # Test power flow equations with purely reactive line
    p1, q1 = power_flow_2_bus(1, 0, 0.9458, -0.0529, 0, -10)
    p2, q2 = power_flow_2_bus(0.9458, -0.0529, 1, 0, 0, -10)
    testing.assert_allclose([p1, q1], [0.5, 0.55], atol=1e-2)
    testing.assert_allclose([p2, q2], [-0.5, -0.5], atol=1e-2)
    print("p1, q1: ", p1, q1)
    print("p2, q2: ", p2, q2)