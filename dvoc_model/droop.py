from math import pi, sin, cos
import numpy as np

from dvoc_model.reference_frames import SinCos, Abc, Dq0, AlphaBeta
from dvoc_model.constants import *
from dvoc_model.simulate import simulate, shift_controller_angle
from dvoc_model.elements import Node, RefFrames
from dvoc_model.calculations import calculate_power


class Droop(Node):
    def __init__(self,
                 p_ref: float,
                 q_ref: float,
                 m_p: float = 2.6e-3,
                 m_q: float = 5.0e-3,
                 v_nom: float = 120.,
                 s_rated: float = 1000.,
                 hz_nom: float = 60,
                 varphi: float = pi / 2,
                 omega_c: float = 2 * pi * 30,
                 ref: RefFrames = RefFrames.POLAR,
                 dt: float = 1.0 / 10e3,
                 start_eq: bool = True,
                 output_hold_periods: int = 1,
                 ):

        # set droop controller parameters
        self.v_nom = v_nom
        self.omega_nom = 2 * pi * hz_nom
        self.omega_c = omega_c
        self.m_p = m_p
        self.m_q = m_q
        self.sin_phi = sin(varphi)
        self.cos_phi = cos(varphi)

        self.p_ref = p_ref
        self.q_ref = q_ref
        self.dt = dt
        self.line = None

        # set low-pass filter initial values
        p_filt = 0
        q_filt = 0
        self.p = 0
        self.q = 0

        # initialize state variables
        if ref is RefFrames.POLAR:
            super().__init__((self.v_nom, 0, p_filt, q_filt), ref)
        else:
            v = AlphaBeta.from_polar(self.v_nom, 0)
            super().__init__((v.alpha, v.beta, p_filt, q_filt), ref)

        self.dependent_cmpnts = []
        self.n_states = self.states.shape[0]

        if start_eq:
            shift_controller_angle(self, self.ref, self.omega_nom, self.dt, 0.5)

        if ref is RefFrames.POLAR:
            self.state_names = ["v", "theta", "p,filt", "q,filt"]
        else:
            self.state_names = ["v,alpha", "v,beta", "p,filt", "q,filt"]

    def low_pass_dynamics(self, x, y_filt):
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
        v, theta, p_filt, q_filt = self.collect_states_fr_x(x)
        v_ab = AlphaBeta.from_polar(v, theta)
        i = self.line.i_alpha_beta() if u is None else u[0].to_alpha_beta()

        # Power Calculation
        p, q = calculate_power(v_ab, i)

        # Low-Pass Filter
        p_filt_dt = self.low_pass_dynamics(p, p_filt)
        q_filt_dt = self.low_pass_dynamics(q, q_filt)

        p_err = p_filt - self.p_ref
        q_err = q_filt - self.q_ref

        # Droop Control
        dvdt = - self.m_q * q_filt_dt
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
