from math import pi, sin, cos
import numpy as np

from dvoc_model.reference_frames import SinCos, Abc, Dq0, AlphaBeta, RefFrames
from dvoc_model.constants import *
from dvoc_model.simulate import simulate
from dvoc_model.elements import Node
from dvoc_model.calculations import *


class Dvoc(Node):
    def __init__(self,
                 p_ref,
                 q_ref,
                 eps: float = 15.,
                 xi: float = 1.,
                 k_v: float = 120.,
                 k_i: float = 0.2,
                 v_nom: float = 120.,
                 hz_nom: float = 60,
                 varphi: float = pi / 2,
                 l: float = 26.268e-6,
                 c: float = 0.2679,
                 ref: RefFrames = RefFrames.POLAR,
                 start_eq: bool = True,
                 dt: float = 1.0 / 10e3,
                 ):
        self.v_nom = v_nom  # RMS amplitude, v_nom
        self.hz_nom = hz_nom
        l = (1 / (hz_nom * 2 * np.pi))**2 / c  # Adjust l to make omega_nom exactly 60 Hz
        self.omega_nom = 1 / sqrt(l * c)
        self.c = c

        # initialize state variables, V (Peak Voltage), Theta (Radians)
        if ref is RefFrames.POLAR:
            super().__init__((self.v_nom, 0), ref)
        else:
            v = AlphaBeta.from_polar(self.v_nom, 0)
            super().__init__((v.alpha, v.beta), ref)

        # set dvoc controller parameters
        self.eps = eps
        self.xi = xi
        self.k_i = k_i
        self.k_v = k_v
        self.sin_phi = sin(varphi)
        self.cos_phi = cos(varphi)

        # initialize power tracking variables
        self.p = 0
        self.q = 0

        self.line = None
        self.p_ref = p_ref
        self.q_ref = q_ref
        self.dt = dt
        if ref is RefFrames.POLAR:
            self.state_names = ["v", "theta"]
            if start_eq:
                self.states[1,0] += self.omega_nom / 2 * self.dt
        else:
            self.state_names = ["v,alpha", "v,beta"]
            if start_eq:
                v = AlphaBeta.from_polar(self.v_nom, self.omega_nom / 2 * self.dt)
                self.states[0,0] = v.alpha
                self.states[1,0] = v.beta

    def collect_states(self, x):
        if self.ref is RefFrames.ALPHA_BETA:
            if x is None:
                v = self.v_alpha_beta()
                v_alpha = v.alpha
                v_beta = v.beta
            else:
                v_alpha, v_beta = x[0], x[1]
            return v_alpha, v_beta
        elif self.ref is RefFrames.POLAR:
            if x is None:
                v, theta = self.states[:,0]
            else:
                v, theta = x[0], x[1]
            return v, theta
        else:
            NotImplementedError()
            return None, None


    def polar_dynamics(self, x=None, t=None, u=None):
        v, theta = self.collect_states(x)
        v_ab = AlphaBeta.from_polar(v, theta)
        i = self.line.i_alpha_beta()

        # Power Calculation
        self.p, self.q = calculate_power(v_ab, i)

        # dVOC Control
        kvki_3cv = self.k_v * self.k_i / (3 * self.c * v)
        v_dt = self.eps / self.k_v ** 2 * v * (2 * self.v_nom ** 2 - 2 * v ** 2) - kvki_3cv * (self.q - self.q_ref)
        theta_dt = self.omega_nom - kvki_3cv / v * (self.p - self.p_ref)

        return np.array([v_dt, theta_dt])

    def alpha_beta_dynamics(self, x=None, t=None, u=None):
        v_alpha, v_beta = self.collect_states(x)
        v = AlphaBeta(v_alpha, v_beta, 0)

        ia_ref, ib_ref = calculate_current(v, self.p_ref, self.q_ref)
        i_ref = AlphaBeta(ia_ref, ib_ref, 0)
        i = self.line.i_alpha_beta()
        i_err = i - i_ref

        tmp = self.eps / (self.k_v ** 2) * (2 * self.v_nom ** 2 - v_alpha ** 2 - v_beta ** 2)  #TEST: (2 * self.v_nom ** 2 - v_alpha ** 2 - v_beta ** 2)
        dadt = tmp * v_alpha \
               - self.omega_nom * v_beta \
               - self.k_v * self.k_i / self.c * (self.cos_phi * i_err.alpha - self.sin_phi * i_err.beta)

        dbdt = tmp * v_beta \
               + self.omega_nom * v_alpha \
               - self.k_v * self.k_i / self.c * (self.sin_phi * i_err.alpha + self.cos_phi * i_err.beta)

        return np.array([dadt, dbdt])

    def tustin_step(self, x=None, t=None, u=None):
        """ Approximate Tustin / bilinear discretized dynamics of the dVOC controller.
        Approximations are made when solving for x[t+1]:
        In the AlphaBeta reference frame voltage magnitude and line currents are assumed constant from x[t] -> x[t+1]
        In the Polar reference frame voltage magnitude, powers, and voltage magnitudes in the power error term are
        assumed constant from x[t] -> x[t+1].
        """
        x1, x2 = self.collect_states(x)
        i = self.line.i_alpha_beta()

        if self.ref == RefFrames.ALPHA_BETA:
            v_alpha, v_beta = x1, x2
            v_ab = AlphaBeta(v_alpha, v_beta, 0)

            ia_ref, ib_ref = calculate_current(v_ab, self.p_ref, self.q_ref)
            i_ref = AlphaBeta(ia_ref, ib_ref, 0)
            i_err = i - i_ref

            u1 = self.k_v * self.k_i / self.c * (self.cos_phi * i_err.alpha - self.sin_phi * i_err.beta)
            u2 = self.k_v * self.k_i / self.c * (self.sin_phi * i_err.alpha + self.cos_phi * i_err.beta)

            temp1 = 1 - 0.5 * self.dt * self.k_i * (2*self.v_nom**2 - (v_alpha**2 + v_beta**2))
            temp2 = 0.5 * self.dt * self.omega_nom
            temp3 = 1 / (temp1 * temp1 + temp2 * temp2)
            temp4 = 1 + 0.5 * self.dt * self.k_i * (2*self.v_nom**2 - (v_alpha**2 + v_beta**2))

            temp5 = temp4 * v_alpha - temp2 * v_beta - self.dt * u1
            temp6 = temp2 * v_alpha + temp4 * v_beta - self.dt * u2

            va = temp3 * (temp1 * temp5 - temp2 * temp6)
            vb = temp3 * (temp2 * temp5 + temp1 * temp6)

            return np.array([va, vb])
        elif self.ref == RefFrames.POLAR:
            v, theta = x1, x2
            v_ab = AlphaBeta.from_polar(v, theta)

            # Power Calculation
            self.p, self.q = calculate_power(v_ab, i)

            # Power Errors
            u1 = (self.p - self.p_ref) * self.k_i / (3 * self.k_v * self.c)
            u2 = (self.q - self.q_ref) * self.k_i / (3 * self.k_v * self.c)

            # Implicit Step Calculation
            temp1 =   self.k_i * (self.v_nom**2 - v**2) * self.dt + 1.0
            temp2 = - self.k_i * (self.v_nom**2 - v**2) * self.dt + 1.0
            temp3 = - u1  * self.dt / v

            v_t1 = temp1 / temp2 * v + temp3 / temp2
            theta_t1 = theta + 0.5 * self.dt * (2.0 * self.omega_nom -  u2 / (v_t1**2) - u2 / (v**2))
            return np.array([v_t1, theta_t1])

    def backward_step(self, x=None, t=None, u=None):
        """ Approximate Backward Euler discretized dynamics of the dVOC controller.
        Approximations in the dynamics equations are made when solving for x[t+1]. These approximations are assuming
        that some states in the dynamics equations are constant from x[t] -> x[t+1].
        AlphaBeta reference frame: voltage magnitude and line currents are assumed constant.
        Polar reference frame: voltage magnitude, powers, and voltage magnitudes in the power error term are
        assumed constant.
        """
        x1, x2 = self.collect_states(x)
        i = self.line.i_alpha_beta()

        if self.ref == RefFrames.ALPHA_BETA:
            v_alpha, v_beta = x1, x2
            v_ab = AlphaBeta(v_alpha, v_beta, 0)

            ia_ref, ib_ref = calculate_current(v_ab, self.p_ref, self.q_ref)
            i_ref = AlphaBeta(ia_ref, ib_ref, 0)
            i_err = i - i_ref

            # Current Error Terms
            u1 = self.k_v * self.k_i / self.c * (self.cos_phi * i_err.alpha - self.sin_phi * i_err.beta)
            u2 = self.k_v * self.k_i / self.c * (self.sin_phi * i_err.alpha + self.cos_phi * i_err.beta)

            # Implicit Step Calculation
            temp1 = 1 - self.dt * self.k_i * (2*self.v_nom**2 - (v_alpha**2 + v_beta**2))
            temp2 = self.dt * self.omega_nom
            temp3 = 1 / (temp1**2 + temp2**2)

            va = temp3 * (temp1 * (v_alpha - self.dt * u1) - temp2 * (v_beta - self.dt * u2))
            vb = temp3 * (temp1 * (v_beta - self.dt * u2) + temp2 * (v_alpha - self.dt * u1))

            return np.array([va, vb])
        elif self.ref == RefFrames.POLAR:
            v, theta = x1, x2
            v_ab = AlphaBeta.from_polar(v, theta)

            # Power Calculation
            self.p, self.q = calculate_power(v_ab, i)

            u1 = (self.p - self.p_ref) * self.k_i / (3 * self.k_v * self.c * v)
            u2 = (self.q - self.q_ref) * self.k_i / (3 * self.k_v * self.c * v)

            # Dynamics Calculation
            temp1 = 1.0 - 2.0 * self.k_i * (self.v_nom**2 - v**2) * self.dt
            temp2 = - u1  * self.dt / v

            v_t1 = 1 / temp1 * v + temp2 / temp1
            theta_t1 = theta + self.dt * (self.omega_nom - u2 / (v_t1**2));
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
