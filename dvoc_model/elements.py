from dvoc_model.reference_frames import SinCos, Abc, Dq0, AlphaBeta, RefFrames, convert_state_ref
from dvoc_model.constants import *
import numpy as np


class Component:
    def __init__(self, x=(None, None), ref=RefFrames.ALPHA_BETA):
        self.ref = ref
        if x is AlphaBeta:
            if ref is RefFrames.ALPHA_BETA:
                x1 = np.array([x.alpha, None])
                x2 = np.array([x.beta, None])
            else:
                x = x.to_polar()
                x1 = np.array([x[0], None])
                x2 = np.array([x[1], None])
        else:
            x1 = np.array([x[0], None])
            x2 = np.array([x[1], None])
        self.states = np.array([x1, x2])

    def dynamics(self, x=None, t=None, u=None):
        if self.ref == RefFrames.ALPHA_BETA:
            return self.alpha_beta_dynamics(x, t, u)
        elif self.ref == RefFrames.POLAR:
            return self.polar_dynamics(x, t, u)
        elif self.ref == RefFrames.DQ0:
            return self.dq_dynamics(x, t, u)
        else:
            NotImplementedError("%s reference frame not implemented" % self.ref)
    
    def step_states(self):
        self.states[:,0] = self.states[:,1]
        self.states[:,1] = None
        if self.ref is RefFrames.POLAR:
            self.states[1,0] %= (2*np.pi)


class Node(Component):
    def __init__(self, x, ref=RefFrames.ALPHA_BETA):
        super().__init__(x, ref)

    def curr_states(self):
        if self.ref is RefFrames.POLAR:
            states = np.array([self.v[0], self.theta[0]])
        elif self.ref is RefFrames.ALPHA_BETA:
            states = np.array([self.v_alpha[0], self.v_beta[0]])
        else:
            states = None
            NotImplementedError()
        return states

    def v_polar(self):
        if self.ref is RefFrames.ALPHA_BETA:
            return AlphaBeta(self.states[0,0], self.states[1,0], 0).to_polar()
        elif self.ref is RefFrames.POLAR:
            return self.v[0], self.theta[0]
        return None

    def v_alpha_beta(self):
        if self.ref is RefFrames.ALPHA_BETA:
            return AlphaBeta(self.states[0,0], self.states[1,0], 0)
        elif self.ref is RefFrames.POLAR:
            return AlphaBeta.from_polar(self.states[0,0], self.states[1,0])
        return None

    def v_alpha(self):
        if self.ref is RefFrames.ALPHA_BETA:
            return self.states[0,0]
        elif self.ref is RefFrames.POLAR:
            return AlphaBeta.from_polar(self.v[0], self.theta[0]).alpha
        return None

    def v_beta(self):
        if self.ref is RefFrames.ALPHA_BETA:
            return self.states[1,0]
        elif self.ref is RefFrames.POLAR:
            return AlphaBeta.from_polar(self.v[0], self.theta[0]).beta
        return None


class Edge(Component):
    def __init__(self, x, ref=RefFrames.ALPHA_BETA):
        super().__init__(x, ref)

    def curr_states(self):
        return self.states[:,0]


class Grid(Node):
    def __init__(self, v: float, theta: float, omega: float = 2 * np.pi * 60, ref: RefFrames = RefFrames.POLAR):
        super().__init__((v, theta), ref)
        self.omega = omega
        self.state_names = ["v", "theta"]
        self.n_states = len(self.state_names)

    def polar_dynamics(self, x=None, t=None, u=None):
        return np.array([0, self.omega])

    def v_alpha_beta(self):
        return AlphaBeta.from_polar(self.states[0,0], self.states[1,0])


class Line(Edge):
    def __init__(self,
                 fr: Node,
                 to: Node,
                 rf: float,
                 lf: float,
                 i_ab: AlphaBeta = AlphaBeta.from_polar(0, 0),
                 ref: RefFrames = RefFrames.ALPHA_BETA
                 ):
        super().__init__((i_ab.alpha, i_ab.beta), ref)
        self.rf = rf
        self.lf = lf
        self.fr = fr
        self.to = to
        self.dependent_cmpnts = [fr, to]
        if self.ref is RefFrames.ALPHA_BETA:
            self.state_names = np.array(["i,alpha", "i,beta"])
        else:
            self.state_names = np.array([None, None])
        self.n_states = len(self.state_names)

    def alpha_beta_dynamics(self, x=(None, None), t=0, u=None):
        if u is None:
            v1 = self.fr.v_alpha_beta()
            v2 = self.to.v_alpha_beta()
        else:
            x_fr = u[0]
            x_to = u[1]
            v1 = AlphaBeta(x_fr[0], x_fr[1], 0)
            v2 = AlphaBeta(x_to[0], x_to[1], 0)

        if x is None:
            i_alpha, i_beta = self.states[:,0]
        else:
            i_alpha, i_beta = x[0], x[1]

        i_dx_alpha = 1/self.lf*(v1.alpha - v2.alpha - self.rf*i_alpha)
        i_dx_beta = 1/self.lf*(v1.beta - v2.beta - self.rf*i_beta)
        return np.array([i_dx_alpha, i_dx_beta])

    def i_alpha_beta(self):
        if self.ref is RefFrames.ALPHA_BETA:
            return AlphaBeta(self.states[0,0], self.states[1,0], 0)
        else:
            raise NotImplemented


class LineToGrid:
    def __init__(self,
                 line: Line,
                 grid: Grid,
                 ):
        self.line = line
        self.grid = grid
        self.states = np.vstack([line.states, grid.states])
        self.components = [line, grid]

    def v_alpha_beta(self):
        return self.grid.v_alpha_beta()

    def i_alpha_beta(self):
        return self.line.i_alpha_beta()

    def dynamics(self, t=0, x=None):
        if x is None:
            i_alpha, i_beta = self.line.states[:,0]
            v_grid, theta_grid = self.grid.states[:,0]
        else:
            i_alpha = x[0]
            i_beta = x[1]
            v_grid = x[2]
            theta_grid = x[3]

        vg_ab = AlphaBeta.from_polar(v_grid, theta_grid)
        v_ab = self.line.fr.v_alpha_beta()
        i_dx_alpha, i_dx_beta = self.line.dynamics([i_alpha, i_beta], t, u=[(v_ab.alpha, v_ab.beta),
                                                                            (vg_ab.alpha, vg_ab.beta)])
        v_dx, theta_dx = self.grid.dynamics([v_grid, theta_grid], t)

        return [i_dx_alpha, i_dx_beta, v_dx, theta_dx]

    def step_states(self):
        self.line.states = self.states[:2,:]
        self.grid.states = self.states[2:,:]
        self.line.step_states()
        self.grid.step_states()


class Load(Node):
    """ Resistive load object
    TODO: Update this do have a reactive component?
    """
    def __init__(self, r, fr: Line, ref: RefFrames = RefFrames.ALPHA_BETA):
        super().__init__((0, 0), ref)
        self.r = r
        self.source = source  # Line that sources current to the load
        self.state_names = ["v,alpha", "v,beta"]
        self.n_states = len(self.state_names)
        self.dependent_cmpnts = [source]

    def alpha_beta_dynamics(self, x=None, t=None, u=None):
        i = self.source.i_alpha_beta()
        v_alpha = i.alpha * self.r
        v_beta = i.beta * self.r
        return np.array([v_alpha, v_beta]) - self.states[:,0]

    def v_alpha_beta(self):
        return AlphaBeta(self.states[0,0], self.states[1,0], 0)


class System:
    def __init__(self,
                 components
                 ):
        self.components = components
        self.states = np.vstack([cmpnt.states for cmpnt in components])
        curr_state_idx = 0
        for cmpnt in components:
            cmpnt.state_idx = curr_state_idx
            curr_state_idx += cmpnt.n_states


    def dynamics(self, t=0, x=None):
        system_dynamics = []
        for cmpnt in self.components:
            cmpnt_states = x[cmpnt.state_idx: cmpnt.state_idx + cmpnt.n_states,0]

            if hasattr(cmpnt, 'dependent_cmpnts'):
                u = []
                for dependent_cmpnt in cmpnt.dependent_cmpnts:
                    if hasattr(dependent_cmpnt, 'state_idx'):
                        u_cmpnt = x[dependent_cmpnt.state_idx:dependent_cmpnt.state_idx + dependent_cmpnt.n_states,0]
                    else:
                        u_cmpnt = dependent_cmpnt.states[:,0]
                    u_cmpnt_ref = convert_state_ref(u_cmpnt, dependent_cmpnt.ref, cmpnt.ref)
                    u.append(u_cmpnt_ref)  # TODO: Or use extend and get a full list?
            else:
                u = None
            cmpnt_dxdt = cmpnt.dynamics(cmpnt_states, t, u=u)
            system_dynamics.append(cmpnt_dxdt)

        return system_dynamics

    def step_states(self):
        for cmpnt in self.components:
            # TODO: Make this directly put the new states values in the current index of the cmpnt states?
            cmpnt.states = self.states[cmpnt.state_idx: cmpnt.state_idx + cmpnt.n_states,:]
            cmpnt.step_states()