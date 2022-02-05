from math import pi, sin, cos
import numpy as np

from dvoc_model.reference_frames import SinCos, Abc, Dq0, AlphaBeta
from dvoc_model.constants import *
from dvoc_model.simulate import simulate
from dvoc_model.elements import Node, RefFrames


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
    

    def polar_dynamics(self, x=None, t=None, u=None):
        i = self.line.i_alpha_beta()
        p_ref = self.p_ref
        q_ref = self.q_ref

        if x is None:
            v, theta = self.states[:,0]
            v_ab = AlphaBeta.from_polar(v, theta)
        else:
            v, theta = x[0], x[1]
            v_ab = AlphaBeta.from_polar(v, theta)

        # Power Calculation
        self.p = 1.5 * (v_ab.alpha * i.alpha + v_ab.beta * i.beta)
        self.q = 1.5 * (v_ab.beta * i.alpha - v_ab.alpha * i.beta)

        # dVOC Control
        kvki_3cv = self.k_v * self.k_i / (3 * self.c * v)
        v_dt = self.eps / self.k_v ** 2 * v * (2 * self.v_nom ** 2 - 2 * v ** 2) - kvki_3cv * (self.q - q_ref)
        theta_dt = self.omega_nom - kvki_3cv / v * (self.p - p_ref)

        return np.array([v_dt, theta_dt])

    def alpha_beta_dynamics(self, x=None, t=None, u=None):
        i = self.line.i_alpha_beta()
        if x is None:
            v = self.v_alpha_beta()
            v_alpha = v.alpha
            v_beta = v.beta
        else:
            v_alpha, v_beta = x[0], x[1]

        p_ref = self.p_ref
        q_ref = self.q_ref

        ia_ref = 2 * (p_ref * v_alpha + q_ref * v_beta) / (v_alpha ** 2 + v_beta ** 2) / 3
        ib_ref = 2 * (p_ref * v_beta - q_ref * v_alpha) / (v_alpha ** 2 + v_beta ** 2) / 3
        i_ref = AlphaBeta(ia_ref, ib_ref, 0)
        i_err = i - i_ref

        tmp = self.eps / (self.k_v ** 2) * (2 * self.v_nom ** 2 - v_alpha ** 2 - v_beta ** 2)  #TEST: (2 * self.v_nom ** 2 - v_alpha ** 2 - v_beta ** 2)
        dadt = tmp * v_alpha \
               - self.omega_nom * v_beta \
               - self.k_v * self.k_i / self.c * (self.cos_phi * i_err.alpha - self.sin_phi * i_err.beta)

        dbdt = tmp * v_beta \
               + self.omega_nom * v_alpha \
               - self.k_v * self.k_i / self.c * (self.sin_phi * i_err.alpha + self.cos_phi * i_err.beta)

        return np.array([dadt, dbdt])

    def tustin_dynamics(self, x=None, t=None, u=None):
        """ Not Implemented Yet """
        i = self.line.i_alpha_beta()
        if x is None:
            v = self.v_alpha_beta()
            v_alpha = v.alpha
            v_beta = v.beta
        else:
            v = AlphaBeta(x[0], x[1], 0)
            v_alpha = x[0]
            v_beta = x[1]
        dt = self.dt

        p_ref = self.p_ref
        q_ref = self.q_ref

        ia_ref = 2 * (p_ref * v.alpha + q_ref * v.beta) / (v.alpha ** 2 + v.beta ** 2) / 3
        ib_ref = 2 * (p_ref * v.beta - q_ref * v.alpha) / (v.alpha ** 2 + v.beta ** 2) / 3
        i_ref = AlphaBeta(ia_ref, ib_ref, 0)
        i_err = i - i_ref

        u1 = self.k_v * self.k_i / self.c * (self.cos_phi * i_err.alpha - self.sin_phi * i_err.beta)
        u2 = self.k_v * self.k_i / self.c * (self.sin_phi * i_err.alpha + self.cos_phi * i_err.beta)

        if self.ref == RefFrames.ALPHA_BETA:
            temp1 = 1 - 0.5 * dt * self.k_i * (2*self.v_nom**2 - (v_alpha**2 + v_beta**2))
            temp2 = 0.5 * dt * self.omega_nom
            temp3 = 1 / (temp1 * temp1 + temp2 * temp2)
            temp4 = 1 + 0.5 * dt * self.k_i * (2*self.v_nom**2 - (v_alpha**2 + v_beta**2))

            temp5 = temp4 * v_alpha - temp2 * v_beta - dt * u1
            temp6 = temp2 * v_alpha + temp4 * v_beta - dt * u2

            va = temp3 * (temp1 * temp5 - temp2 * temp6)
            vb = temp3 * (temp2 * temp5 + temp1 * temp6)

            return np.array([va, vb])
        elif self.ref == RefFrames.POLAR:
            NotImplementedError()

    def backward_dynamics(self, x=None, t=None, u=None):
        """ Not Implemented Yet """
        i = self.line.i_alpha_beta()
        if x is None:
            v = self.v_alpha_beta()
            v_alpha = v.alpha
            v_beta = v.beta
        else:
            v = AlphaBeta(x[0], x[1], 0)
            v_alpha = x[0]
            v_beta = x[1]
        dt = self.dt

        p_ref = self.p_ref
        q_ref = self.q_ref

        ia_ref = 2 * (p_ref * v.alpha + q_ref * v.beta) / (v.alpha ** 2 + v.beta ** 2) / 3
        ib_ref = 2 * (p_ref * v.beta - q_ref * v.alpha) / (v.alpha ** 2 + v.beta ** 2) / 3
        i_ref = AlphaBeta(ia_ref, ib_ref, 0)
        i_err = i - i_ref

        u1 = self.k_v * self.k_i / self.c * (self.cos_phi * i_err.alpha - self.sin_phi * i_err.beta)
        u2 = self.k_v * self.k_i / self.c * (self.sin_phi * i_err.alpha + self.cos_phi * i_err.beta)

        if self.ref == RefFrames.ALPHA_BETA:

            temp1 = 1 - dt * self.k_i * (2*self.v_nom**2 - (v_alpha**2 + v_beta**2))
            temp2 = dt * self.omega_nom
            temp3 = 1 / (temp1**2 + temp2**2)

            va = temp3 * (temp1 * (v_alpha - dt * u1) - temp2 * (v_beta - dt * u2))
            vb = temp3 * (temp1 * (v_beta - dt * u2) + temp2 * (v_alpha - dt * u1))

            return np.array([va, vb])
        elif self.ref == RefFrames.POLAR:
            NotImplementedError()


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

    controller = Dvoc(p_ref[0], q_ref[0], ref=RefFrames.POLAR, start_eq=False)

    data = simulate(controller, p_ref, q_ref, dt, t, Rf=0.4)  # Critical Rf value found to be 0.24-0.25

    plt.show()
