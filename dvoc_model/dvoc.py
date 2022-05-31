from math import pi, sin, cos
import numpy as np

from dvoc_model.reference_frames import SinCos, Abc, Dq0, AlphaBeta, RefFrames
from dvoc_model.constants import *
from dvoc_model.simulate import simulate, shift_controller_angle_half
from dvoc_model import shift_angle
from dvoc_model.elements import Node
from dvoc_model.calculations import *


class Dvoc(Node):
    def __init__(self,
                 p_ref,
                 q_ref,
                 eps: float = 15.,
                 xi: float = 15.,
                 k_v: float = 120.,
                 k_i: float = 0.2,
                 v_nom: float = 120.,
                 s_rated: float = 1000.,
                 hz_nom: float = 60,
                 varphi: float = pi / 2,
                 l: float = 26.268e-6,
                 c: float = 0.2679,
                 ref: RefFrames = RefFrames.POLAR,
                 start_eq: bool = True,
                 dt: float = 1.0 / 10e3,
                 p_refs=None,
                 q_refs=None,
                 output_hold_periods: int = 1,
                 ):
        self.v_nom = v_nom  # RMS amplitude, v_nom
        self.hz_nom = hz_nom
        l = (1 / (hz_nom * 2 * np.pi)) ** 2 / c  # Adjust l to make omega_nom exactly 60 Hz
        self.omega_nom = 1 / sqrt(l * c)
        self.c = c
        self.x_nom = 1.0

        # initialize state variables, V (Peak Voltage), Theta (Radians)
        if ref is RefFrames.POLAR:
            super().__init__((self.v_nom, 0), ref)
        else:
            v = AlphaBeta.from_polar(self.v_nom, 0)
            super().__init__((v.alpha, v.beta), ref)
        if output_hold_periods > 1:
            self.output_hold_periods = output_hold_periods
            self.output = self.collect_states()

        # set dvoc controller parameters
        self.eps = eps
        self.xi = xi
        self.k_i = 3 * v_nom / s_rated  # TODO: Max power in experiment
        self.k_v = v_nom
        self.sin_phi = sin(varphi)
        self.cos_phi = cos(varphi)

        # initialize power tracking variables
        self.p = 0
        self.q = 0

        self.line = None
        self.p_ref = p_ref
        self.q_ref = q_ref
        self.p_refs = p_refs
        self.q_refs = q_refs
        self.dt = dt

        # TODO: Add dependant components array
        self.dependent_cmpnts = [self.line]
        self.n_states = self.states.shape[0]

        if start_eq:
            shift_controller_angle_half(self, self.ref, self.omega_nom, self.dt)

        if ref is RefFrames.POLAR:
            self.state_names = ["v", "theta"]
        else:
            self.state_names = ["v,alpha", "v,beta"]

    def polar_dynamics(self, x=None, t=None, u=None):
        v, theta = self.collect_voltage_states(x)
        v_ab = AlphaBeta.from_polar(v, theta)
        i = self.line.i_alpha_beta() if u is None else u[0].to_alpha_beta()
        # TODO: Determine if this helps multi-step methods. It does not... first method helps FE that's it
        """
        if u is None:  # Only when not using a system solver
            # Trying to find average power over interrupt period
            #idq = i.to_dq0(SinCos.from_theta(self.v_polar()[1]))  # Convert to DQ using starting voltage angle
            #theta_ = self.v_polar()[1]
            #theta_dt = theta_ + self.omega_nom * self.dt
            #p1 = sin(theta_dt) * (v_ab.alpha * idq.d + v_ab.beta * idq.q) + cos(theta_dt) * (v_ab.alpha * idq.q - v_ab.beta * idq.d)
            #p2 = sin(theta_) * (v_ab.alpha * idq.d + v_ab.beta * idq.q) + cos(theta_) * (v_ab.alpha * idq.q - v_ab.beta * idq.d)
            #p = 1.5 / self.dt * (p2 - p1) / self.omega_nom
            #q1 = sin(theta_dt) * (v_ab.beta * idq.d - v_ab.alpha * idq.q) + cos(theta_dt) * (v_ab.alpha * idq.d + v_ab.beta * idq.q)
            #q2 = sin(theta_) * (v_ab.beta * idq.d - v_ab.alpha * idq.q) + cos(theta_) * (v_ab.alpha * idq.d + v_ab.beta * idq.q)
            #q = 1.5 / self.dt * (q2 - q1) / self.omega_nom
            #self.p, self.q = p, q

            # Trying to shift curent with voltage angle
            # idq = i.to_dq0(SinCos.from_theta(self.v_polar()[1]))  # Convert to DQ using starting voltage angle
            #i = idq.to_alpha_beta(SinCos.from_theta(theta))  # Convert back to ab using predicted voltage angle

            # Try shifting current halfway such that P = Pref at the middle of the interrupt period
            idq = i.to_dq0(SinCos.from_theta(0))  # Convert to DQ using starting voltage angle
            i = idq.to_alpha_beta(SinCos.from_theta(-self.omega_nom * self.dt / 2))  # Convert back to ab using predicted voltage angle
        """

        # Power Calculation
        #self.p, self.q = calculate_power(shift_angle(v_ab, -self.omega_nom*self.dt/2), i)
        self.p, self.q = calculate_power(v_ab, i)

        # TODO: Should this be phase shifted to compensate for ZOH?
        # TODO: Phase-shift in current due to ADC sampling ZOH makes this moot?

        # dVOC Control
        kvki_3cv = self.k_v * self.k_i / (3 * self.c * v)
        v_dt = self.xi / self.k_v ** 2 * v * (2 * self.v_nom ** 2 - 2 * v ** 2) - kvki_3cv * (self.q - self.q_ref)
        theta_dt = self.omega_nom - kvki_3cv / v * (self.p - self.p_ref)

        return np.array([v_dt, theta_dt])

    def alpha_beta_dynamics(self, x=None, t=None, u=None):
        v_alpha, v_beta = self.collect_voltage_states(x)
        i = self.line.i_alpha_beta() if u is None else AlphaBeta(u[0][0], u[0][1], 0)
        v = AlphaBeta(v_alpha, v_beta, 0)

        ia_ref, ib_ref = calculate_current(v, self.p_ref, self.q_ref)
        i_ref = AlphaBeta(ia_ref, ib_ref, 0)
        i_err = i - i_ref

        tmp = self.xi / (self.k_v ** 2) * (
                    2 * self.v_nom ** 2 - v_alpha ** 2 - v_beta ** 2)
        dadt = tmp * v_alpha \
               - self.omega_nom * v_beta \
               - self.k_v * self.k_i / self.c * (self.cos_phi * i_err.alpha - self.sin_phi * i_err.beta)

        dbdt = tmp * v_beta \
               + self.omega_nom * v_alpha \
               - self.k_v * self.k_i / self.c * (self.sin_phi * i_err.alpha + self.cos_phi * i_err.beta)

        return np.array([dadt, dbdt])

    def ddot(self, x=None, t=None, u=None, xdot=None):
        va, vb = self.collect_voltage_states(x)
        vad, vbd = self.collect_voltage_states(xdot)

        va = va + vad * self.dt  # TODO: Semi-implicit has any benefits?
        vb = vb + vbd * self.dt  # Yes it improves power tracking for both P & Q

        temp1 = 2 * self.v_nom**2 - va**2 - vb**2
        temp2 = -2 * va * vad - 2 * vb * vbd
        k = self.xi / self.k_v**2
        vadd = -self.omega_nom * vbd + k * (va * temp2 + vad * temp1)
        vbdd = self.omega_nom * vad + k * (vb * temp2 + vbd * temp1)

        return np.array([vadd, vbdd])

    def tustin_step(self, x=None, t=None, u=None):
        """ Approximate Tustin / bilinear discretized dynamics of the dVOC controller.
        Approximations are made when solving for x[t+1]:
        In the AlphaBeta reference frame voltage magnitude and line currents are assumed constant from x[t] -> x[t+1]
        In the Polar reference frame voltage magnitude, powers, and voltage magnitudes in the power error term are
        assumed constant from x[t] -> x[t+1].
        """
        x1, x2 = self.collect_voltage_states(x)
        i = self.line.i_alpha_beta() if u is None else AlphaBeta(u[0][0], u[0][1], 0)

        if self.ref == RefFrames.ALPHA_BETA:
            v_alpha, v_beta = x1, x2
            v_ab = AlphaBeta(v_alpha, v_beta, 0)

            ia_ref, ib_ref = calculate_current(v_ab, self.p_ref, self.q_ref)
            i_ref = AlphaBeta(ia_ref, ib_ref, 0)
            i_err = i - i_ref

            u1 = self.k_i * (self.cos_phi * i_err.alpha - self.sin_phi * i_err.beta) / self.c
            u2 = self.k_i * (self.sin_phi * i_err.alpha + self.cos_phi * i_err.beta) / self.c

            v_alpha = v_alpha / self.k_v
            v_beta = v_beta / self.k_v

            temp1 = 1 - 0.5 * self.dt * self.xi * (2 * self.x_nom ** 2 - (v_alpha ** 2 + v_beta ** 2))
            temp2 = 0.5 * self.dt * self.omega_nom
            temp3 = 1 / (temp1 * temp1 + temp2 * temp2)
            temp4 = 1 + 0.5 * self.dt * self.xi * (2 * self.x_nom ** 2 - (v_alpha ** 2 + v_beta ** 2))

            temp5 = temp4 * v_alpha - temp2 * v_beta - self.dt * u1
            temp6 = temp2 * v_alpha + temp4 * v_beta - self.dt * u2

            va = temp3 * (temp1 * temp5 - temp2 * temp6) * self.k_v
            vb = temp3 * (temp2 * temp5 + temp1 * temp6) * self.k_v

            return np.array([va, vb])
        elif self.ref == RefFrames.POLAR:
            v, theta = x1, x2
            v_ab = AlphaBeta.from_polar(v, theta)

            # Power Calculation
            self.p, self.q = calculate_power(v_ab, i)

            # Power Errors
            u2 = (self.p - self.p_ref) * self.k_i * self.k_v / (3 * self.c)
            u1 = (self.q - self.q_ref) * self.k_i * self.k_v / (3 * self.c)

            # Implicit Step Calculation
            a = -4 * self.xi * self.v_nom ** 2 + u1 / (self.v_nom ** 2)
            b = 4 * self.xi * self.v_nom ** 3 - 2.0 * u1 / self.v_nom
            temp1 = 1.0 + a / 2 * self.dt
            temp2 = 1.0 - a / 2 * self.dt
            temp3 = b * self.dt

            v_t1 = temp1 / temp2 * v + temp3 / temp2
            theta_t1 = theta + 0.5 * self.dt * (2.0 * self.omega_nom - u2 / (v_t1 ** 2) - u2 / (v ** 2))
            return np.array([v_t1, theta_t1])

    def backward_step(self, x=None, t=None, u=None):
        """ Approximate Backward Euler discretized dynamics of the dVOC controller.
        Approximations in the dynamics equations are made when solving for x[t+1]. These approximations are assuming
        that some states in the dynamics equations are constant from x[t] -> x[t+1].
        AlphaBeta reference frame: voltage magnitude and line currents are assumed constant.
        Polar reference frame: voltage magnitude, powers, and voltage magnitudes in the power error term are
        assumed constant.
        """
        x1, x2 = self.collect_voltage_states(x)
        i = self.line.i_alpha_beta() if u is None else AlphaBeta(u[0][0], u[0][1], 0)

        if self.ref == RefFrames.ALPHA_BETA:
            v_alpha, v_beta = x1, x2
            v_ab = AlphaBeta(v_alpha, v_beta, 0)

            ia_ref, ib_ref = calculate_current(v_ab, self.p_ref, self.q_ref)
            i_ref = AlphaBeta(ia_ref, ib_ref, 0)
            i_err = i - i_ref

            # Current Error Terms
            u1 = self.k_i * (self.cos_phi * i_err.alpha - self.sin_phi * i_err.beta) / self.c
            u2 = self.k_i * (self.sin_phi * i_err.alpha + self.cos_phi * i_err.beta) / self.c

            v_alpha = v_alpha / self.k_v
            v_beta = v_beta / self.k_v

            # Implicit Step Calculation
            temp1 = 1 - self.dt * self.xi * (2 * self.x_nom ** 2 - (v_alpha ** 2 + v_beta ** 2))
            temp2 = self.dt * self.omega_nom
            temp3 = 1 / (temp1 ** 2 + temp2 ** 2)

            va = temp3 * (temp1 * (v_alpha - self.dt * u1) - temp2 * (v_beta - self.dt * u2)) * self.k_v
            vb = temp3 * (temp1 * (v_beta - self.dt * u2) + temp2 * (v_alpha - self.dt * u1)) * self.k_v

            return np.array([va, vb])
        elif self.ref == RefFrames.POLAR:
            v, theta = x1, x2
            v_ab = AlphaBeta.from_polar(v, theta)

            # Power Calculation
            self.p, self.q = calculate_power(v_ab, i)

            u2 = (self.p - self.p_ref) * self.k_i * self.k_v / (3 * self.c)
            u1 = (self.q - self.q_ref) * self.k_i * self.k_v / (3 * self.c)

            # Implicit Step Calculations
            a = -4 * self.xi * self.v_nom ** 2 + u1 / (self.v_nom ** 2)
            b = 4 * self.xi * self.v_nom ** 3 - 2.0 * u1 / self.v_nom
            temp1 = 1.0 - a * self.dt
            temp2 = b * self.dt

            v_t1 = 1 / temp1 * v + temp2 / temp1
            theta_t1 = theta + self.dt * (self.omega_nom - u2 / (v_t1 ** 2))

            return np.array([v_t1, theta_t1])


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    v_nom = 120
    omega_c = 2 * pi * 30  # Changing this value changes how quickly P & Q filt reach the average cycle values
    q_ = 0

    # grid parameters
    grid = Dq0(SQRT_2 * v_nom, 0, 0)
    grid_omega = TWO_PI * 60

    # simulation time parameters
    dt = 1 / 10e3
    t = 1000e-3
    ts = np.arange(0, t, dt)
    steps = len(ts)

    # create a step function for dispatch (3A to 6A)
    q_ref = q_ * np.ones(steps)
    p_ref = 0 * np.ones(steps)

    p_ref[len(ts) // 2:] = 500  # Add a step in the Active Power reference

    controller = Dvoc(p_ref[0], q_ref[0], ref=RefFrames.ALPHA_BETA, start_eq=True)

    data = simulate(controller, p_ref, q_ref, dt, t, Rf=0.4)  # Critical Rf value found to be 0.24-0.25

    plt.show()
