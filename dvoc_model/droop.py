from math import pi, sin, cos
import numpy as np

from dvoc_model.reference_frames import SinCos, Abc, Dq0, AlphaBeta
from dvoc_model.constants import *
from simulate import simulate
from elements import Node, RefFrames


class Droop(Node):
    def __init__(self,
                 p_ref: float,
                 q_ref: float,
                 m_p: float = 2.6e-3,
                 m_q: float = 5.0e-3,
                 v_nom: float = 120.,
                 hz_nom: float = 60,
                 varphi: float = pi / 2,
                 omega_c: float = 2 * pi * 30,
                 ref: RefFrames = RefFrames.POLAR,
                 dt: float = 1.0 / 10e3,
                 start_eq: bool = True,
                 ):

        self.v_nom = v_nom
        self.hz_nom = hz_nom
        #self.omega_nom = 1 / sqrt(l*c)
        self.omega_nom = 2 * pi * hz_nom  # TODO: Check if this is correct for droop control
        self.p_ref = p_ref
        self.q_ref = q_ref
        self.line = None


        # initialize state variables
        #self.v = v_nom  # TODO: Should this be sqrt(2) * v_nom?
        #self.theta = 0
        super().__init__(v_nom*SQRT_2, 0, ref)

        # set droop controller parameters
        self.m_p = m_p
        self.m_q = m_q
        self.sin_phi = sin(varphi)
        self.cos_phi = cos(varphi)

        # set low-pass filter initial values
        self.omega_c = omega_c
        self.p_filt = 0
        self.q_filt = 0
        self.p = 0
        self.q = 0

        if start_eq:
            self.theta += self.omega_nom / 2 * dt
        self.states = np.array([self.v, self.theta, self.p_filt, self.q_filt])

    def low_pass_dynamics(self, x, y_filt):  # TODO: Search how to derive discretization of low pass
        return self.omega_c * (x - y_filt)

    def update_states(self):
        self.v = self.states[0]
        self.theta = self.states[1]
        self.p_filt = self.states[2]
        self.q_filt = self.states[3]

    def alpha_beta_dynamics(self, x=None, t=None, u=None):
        # Power Calculation
        v = AlphaBeta.from_polar(self.v, self.theta)
        i = self.line.i
        p_calc = 1.5 * (v.alpha * i.alpha + v.beta * i.beta)
        q_calc = 1.5 * (v.beta * i.alpha - v.alpha * i.beta)

        self.p = p_calc
        self.q = q_calc

        # Low-Pass Filter
        p_filt_dt = self.low_pass_dynamics(p_calc, self.p_filt)
        q_filt_dt = self.low_pass_dynamics(q_calc, self.q_filt)

        p_err = self.p_filt - self.p_ref
        q_err = self.q_filt - self.q_ref

        # Droop Control
        dadt = None
        dbdt = None

        dvdt = dbdt  # TODO: Implement this?
        omega = dadt  # Todo: Implement this?
        return np.array([dvdt, omega, p_filt_dt, q_filt_dt])

    def polar_dynamics(self, x=None, t=None, u=None):
        # Power Calculation
        if x is None:
            v_ab = AlphaBeta.from_polar(self.v, self.theta)
            v = self.v
            theta = self.theta
            p_filt = self.p_filt
            q_filt = self.q_filt
        else:
            v_ab = AlphaBeta.from_polar(x[0], x[1])
            v = x[0]
            theta = x[1]
            p_filt = x[2]
            q_filt = x[3]

        i = self.line.i_alpha_beta()
        p_calc = 1.5 * (v_ab.alpha * i.alpha + v_ab.beta * i.beta)
        q_calc = 1.5 * (v_ab.beta * i.alpha - v_ab.alpha * i.beta)

        self.p = p_calc
        self.q = q_calc

        # Low-Pass Filter
        p_filt_dt = self.low_pass_dynamics(p_calc, p_filt)
        q_filt_dt = self.low_pass_dynamics(q_calc, q_filt)

        p_err = self.p_filt - self.p_ref
        q_err = self.q_filt - self.q_ref

        # Droop Control
        dvdt = (self.v_nom*SQRT_2 - self.m_q * q_err) - v
        omega = self.omega_nom - self.m_p * p_err
        return np.array([dvdt, omega, p_filt_dt, q_filt_dt])


if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot as plt

    v_nom = 120
    omega_c = 2*pi*30  # Changing this value changes how quickly P & Q filt reach the average cycle values
    q_ = 0

    # grid parameters
    grid = Dq0(SQRT_2*v_nom, 0, 0)
    grid_omega = TWO_PI * 60

    # simulation time parameters
    dt = 1 / 10e3
    t = 1000e-3
    ts = np.arange(0, t, dt)
    steps = len(ts)

    # create a step function for dispatch (3A to 6A)
    q_ref = q_ * np.ones(steps)

    p_ref = 0 * np.ones(steps)

    p_ref[len(ts)//8:] = 250  # Add a step in the Active Power reference
    p_ref[len(ts)//4:] = 500  # Add a step in the Active Power reference
    p_ref[len(ts)//2:] = 750  # Add a step in the Active Power reference

    controller = Droop(0., 0.)
    data = simulate(controller, p_ref, q_ref, dt, t, Rf=0.8)#, id0=1.93, iq0=-1.23)

    plt.show()
