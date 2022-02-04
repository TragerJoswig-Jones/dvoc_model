from enum import Enum
from dvoc_model.reference_frames import SinCos, Abc, Dq0, AlphaBeta
from dvoc_model.constants import *
import numpy as np


class RefFrames(Enum):
    ALPHA_BETA = 1
    POLAR = 2
    DQ0 = 3


class Component:
    def __init__(self, ref=RefFrames.ALPHA_BETA):
        self.ref = ref

    def dynamics(self, x=None, t=None, u=None):
        if self.ref == RefFrames.ALPHA_BETA:
            return self.alpha_beta_dynamics(x, t, u)
        elif self.ref == RefFrames.POLAR:
            return self.polar_dynamics(x, t, u)
        elif self.ref == RefFrames.DQ0:
            return self.dq_dynamics(x, t, u)


class Node(Component):
    def __init__(self, v=0., theta=0., ref=RefFrames.ALPHA_BETA):
        super().__init__(ref)
        if ref is RefFrames.ALPHA_BETA:
            v = AlphaBeta.from_polar(v, theta)
            self.v_alpha = v.alpha
            self.v_beta = v.beta
        else:
            self.v = v
            self.theta = theta

    def v_alpha_beta(self):
        if self.ref is RefFrames.ALPHA_BETA:
            return AlphaBeta(self.v_alpha, self.v_beta, 0)
        elif self.ref is RefFrames.POLAR:
            return AlphaBeta.from_polar(self.v, self.theta)
        return None


class Edge(Component):
    def __init__(self, ref=RefFrames.ALPHA_BETA):
        super().__init__(ref)


class Grid(Node):
    def __init__(self, v: float, theta: float, omega: float = 2 * np.pi * 60, ref: RefFrames = RefFrames.POLAR):
        super().__init__(v, theta, ref)
        self.omega = omega
        self.states = np.array([self.v, self.theta])

    def polar_dynamics(self, x=None, t=None, u=None):
        return np.array([0, self.omega])

    def step(self, dt, dx):
        self.states += dx * dt

    def update_states(self):
        self.v = self.states[0]
        self.theta = self.states[1] % (2*np.pi)

    def v_alpha_beta(self):
        return AlphaBeta.from_polar(self.v, self.theta)


class Line(Edge):
    def __init__(self,
                 fr: Node,
                 to: Node,
                 rf: float,
                 lf: float,
                 i: AlphaBeta = AlphaBeta.from_polar(0, 0),
                 ref: RefFrames = RefFrames.ALPHA_BETA
                 ):
        super().__init__(ref)
        self.rf = rf
        self.lf = lf
        self.i_alpha = i.alpha
        self.i_beta = i.beta
        self.fr = fr
        self.to = to
        self.states = np.array([i.alpha, i.beta])

    def alpha_beta_dynamics(self, x=[None, None], t=0, u: AlphaBeta = None):
        v1 = self.fr.v_alpha_beta()
        if u is None:
            v2 = self.to.v_alpha_beta()
        else:
            v2 = u

        if x is None:
            i_alpha = self.i.alpha
            i_beta = self.i.beta
        else:
            i_alpha = x[0]
            i_beta = x[1]

        i_dx_alpha = 1/self.lf*(v1.alpha - v2.alpha - self.rf*i_alpha)
        i_dx_beta = 1/self.lf*(v1.beta - v2.beta - self.rf*i_beta)
        return [i_dx_alpha, i_dx_beta]

    def update_states(self):
        self.i_alpha = self.states[0]
        self.i_beta = self.states[1]

    def i_alpha_beta(self):
        if self.ref is RefFrames.ALPHA_BETA:
            return AlphaBeta(self.i_alpha, self.i_beta, 0)
        else:
            raise NotImplemented


class LineToGrid:
    def __init__(self,
                 line: Line,
                 grid: Grid,
                 ):
        self.line = line
        self.grid = grid
        self.states = np.array([line.i_alpha, line.i_beta, grid.v, grid.theta])
        self.components = [line, grid]

    def step(self, dt, dx):
        self.states += dx * dt

    def v_alpha_beta(self):
        return self.grid.v_alpha_beta()

    def i_alpha_beta(self):
        return self.line.i_alpha_beta()

    def dynamics(self, t=0, x=[None, None]):
        v1 = self.line.fr.v_alpha_beta()

        if x is None:
            i_alpha = self.line.i_alpha
            i_beta = self.line.i_beta
            v_grid = self.grid.v
            theta_grid = self.grid.theta
        else:
            i_alpha = x[0]
            i_beta = x[1]
            v_grid = x[2]
            theta_grid = x[3]

        i_dx_alpha, i_dx_beta = self.line.dynamics([i_alpha, i_beta], t, u=AlphaBeta.from_polar(v_grid, theta_grid))
        v_dx, theta_dx = self.grid.dynamics([v_grid, theta_grid], t)

        return [i_dx_alpha, i_dx_beta, v_dx, theta_dx]

    def update_states(self):
        self.line.i_alpha = self.states[0]
        self.line.i_beta = self.states[1]
        self.grid.v = self.states[2]
        self.grid.theta = self.states[3]

    def polar_dynamics(self, x=None, t=None):
        return np.array([0, self.grid.omega])