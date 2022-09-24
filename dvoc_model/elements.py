from dvoc_model.reference_frames import SinCos, Abc, Dq0, AlphaBeta, RefFrames, convert_state_ref
from dvoc_model.constants import *
from dvoc_model.calculations import calculate_power
from dvoc_model import shift_angle
import numpy as np


class Component:
    def __init__(self, x=(None, None), ref=RefFrames.ALPHA_BETA):
        self.ref = ref
        if isinstance(x, AlphaBeta):
            if ref is RefFrames.ALPHA_BETA:
                x1 = np.array([x.alpha, x.alpha])
                x2 = np.array([x.beta, x.beta])
            else:
                x = x.to_polar()
                x1 = np.array([x[0], x[0]])
                x2 = np.array([x[1], x[1]])
            self.states = np.array([x1, x2])
        else:
            states = []
            for val in x:
                states.append([val, val])
            self.states = np.array(states)

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
        # TEST: STORE LAST STATE FOR ADAM BASHFORTH TEST
        last_state = np.array(self.states[:, 0])
        self.states[:, 0] = self.states[:, 1]
        #self.states[:, 1] = None
        self.states[:, 1] = last_state
        if self.ref is RefFrames.POLAR:
            self.states[1, 0] %= (2*np.pi)

    def update_output(self):
        if hasattr(self, 'output'):
            self.output = self.states[:, 0].copy()

    def collect_states(self, internal=False):
        if hasattr(self, 'output') and not internal:
            states = self.output
        else:
            states = self.states[:, 0]
        return states


class Node(Component):
    def __init__(self, x, ref=RefFrames.ALPHA_BETA):
        super().__init__(x, ref)

    def set_line(self, line):
        self.line = line
        if hasattr(self, 'dependent_cmpnts'):
            self.dependent_cmpnts.append(line)
        else:
            self.dependent_cmpnts = [line]

    def curr_states(self):
        if self.ref is RefFrames.POLAR:
            states = np.array([self.v[0], self.theta[0]])
        elif self.ref is RefFrames.ALPHA_BETA:
            states = np.array([self.v_alpha[0], self.v_beta[0]])
        else:
            states = None
            NotImplementedError()
        return states

    def v_polar(self, internal=False):
        states = self.collect_states(internal=internal)
        if self.ref is RefFrames.ALPHA_BETA:
            v_polar = AlphaBeta(states[0], states[1], 0).to_polar()
            return v_polar.r, v_polar.theta
        elif self.ref is RefFrames.POLAR:
            return states[0], states[1]
        return None

    def v_alpha_beta(self, internal=False):
        states = self.collect_states(internal=internal)
        if self.ref is RefFrames.ALPHA_BETA:
            return AlphaBeta(states[0], states[1], 0)
        elif self.ref is RefFrames.POLAR:
            return AlphaBeta.from_polar(states[0], states[1])
        return None

    def v_alpha(self, internal=False):
        return self.v_alpha_beta(internal=internal).alpha

    def v_beta(self, internal=False):
        return self.v_alpha_beta(internal=internal).beta

    def collect_states_fr_x(self, x):
        if x is not None:
            return x
        else:
            return self.states[:, 0]

    def collect_voltage_states(self, x):
        if self.ref is RefFrames.ALPHA_BETA:
            if x is None:
                v = self.v_alpha_beta(internal=True)
                v_alpha = v.alpha
                v_beta = v.beta
            else:
                v_alpha, v_beta = x[0], x[1]
            return v_alpha, v_beta
        elif self.ref is RefFrames.POLAR:
            if x is None:
                v, theta = self.v_polar(internal=True)
            else:
                v, theta = x[0], x[1]
            return v, theta
        else:
            NotImplementedError()
            return None, None


class Edge(Component):
    def __init__(self, x, ref=RefFrames.ALPHA_BETA):
        super().__init__(x, ref)

    def curr_states(self):
        return self.states[:, 0]

    def i_alpha_beta(self, internal=False):
        states = self.collect_states(internal=internal)
        if self.ref is RefFrames.ALPHA_BETA:
            return AlphaBeta(states[0], states[1], 0)
        else:
            raise NotImplemented


class Grid(Node):
    def __init__(self, v: float, theta: float, omega: float = 2 * np.pi * 60, ref: RefFrames = RefFrames.POLAR):
        super().__init__((v, theta), ref)
        self.omega = omega
        self.state_names = ["v", "theta"]
        self.n_states = len(self.state_names)

    def polar_dynamics(self, x=None, t=None, u=None):
        return np.array([0, self.omega])

    #def v_alpha_beta(self):
    #    return AlphaBeta.from_polar(self.states[0,0], self.states[1,0])


class Line(Edge):
    def __init__(self,
                 fr: Node,
                 to: Node,
                 rf: float,
                 lf: float,
                 i_ab: AlphaBeta = AlphaBeta.from_polar(0, 0),
                 ref: RefFrames = RefFrames.ALPHA_BETA,
                 v_nom: float = 120.,
                 ):
        super().__init__((i_ab.alpha, i_ab.beta), ref)
        self.rf = rf
        self.lf = lf
        self.fr = fr
        self.to = to
        fr.set_line(self)
        to.set_line(self)
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

    def i_alpha_beta(self, internal=False):
        return self.line.i_alpha_beta(internal=internal)

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


class LCL_Filter(Edge):  # TODO: Split this into line and capacitor elements
    def __init__(self,
                 fr: Node,
                 to: Node,
                 rf1: float,
                 lf1: float,
                 c: float = 10e-6,  # 4.7uF used before
                 rc: float = 0.05,
                 rf2: float = None,
                 lf2: float = None,
                 v_nom: float = 120.,
                 i_ab: AlphaBeta = AlphaBeta.from_polar(0, 0),
                 ref: RefFrames = RefFrames.ALPHA_BETA,
                 meas_side=0,
                 ):
        v = AlphaBeta.from_polar(v_nom, 0)
        super().__init__((i_ab.alpha, i_ab.beta, v.alpha, v.beta, i_ab.alpha, i_ab.beta), ref)
        self.rf1 = rf1
        self.lf1 = lf1
        self.rf2 = rf1 if rf2 is None else rf2
        self.lf2 = lf1 if lf2 is None else lf2
        self.c = c
        self.rc = rc
        self.fr = fr
        self.to = to
        fr.set_line(self)
        to.set_line(self)
        self.dependent_cmpnts = [fr, to]

        if self.ref is RefFrames.ALPHA_BETA:
            self.state_names = np.array(["i1,alpha", "i1,beta", "v,alpha", "v,beta", "i2,alpha", "i2,beta"])
        else:
            self.state_names = np.array([None, None])
        self.n_states = len(self.state_names)

        self.meas_side = meas_side

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
            i1_alpha, i1_beta = self.states[0:2,0]
            vc_alpha, vc_beta = self.states[2:4:,0]
            i2_alpha, i2_beta = self.states[4:,0]
        else:
            i1_alpha, i1_beta = x[0], x[1]
            vc_alpha, vc_beta = x[2], x[3]
            i2_alpha, i2_beta = x[4], x[5]

        ic_alpha = i1_alpha - i2_alpha
        ic_beta = i1_beta - i2_beta

        i1_dx_alpha = 1/self.lf1*(v1.alpha - (vc_alpha + self.rc * ic_alpha) - self.rf1*i1_alpha)
        i1_dx_beta = 1/self.lf1*(v1.beta - (vc_beta + self.rc * ic_beta) - self.rf1*i1_beta)

        v_dx_alpha = 1/self.c * ic_alpha
        v_dx_beta = 1/self.c * ic_beta

        i2_dx_alpha = 1/self.lf2*(vc_alpha - v2.alpha - self.rf2*i2_alpha)
        i2_dx_beta = 1/self.lf2*(vc_beta - v2.beta - self.rf2*i2_beta)

        return np.array([i1_dx_alpha, i1_dx_beta, v_dx_alpha, v_dx_beta, i2_dx_alpha, i2_dx_beta])

    def i_alpha_beta(self, internal=True):
        if self.meas_side == 0:
            return AlphaBeta(self.states[0, 0], self.states[1, 0], 0)
        else:
            return AlphaBeta(self.states[4, 0], self.states[5, 0], 0)


class Load(Node):
    """ Resistive load object
    TODO: Update this do have a reactive component?
    """
    def __init__(self, r, v_nom=120., ref: RefFrames = RefFrames.ALPHA_BETA, line: Line = None):
        super().__init__(AlphaBeta.from_polar(v_nom, 0), ref)
        self.r = r

        self.state_names = ["v,alpha", "v,beta"]
        self.n_states = len(self.state_names)

        if not(line is None):
            self.line = line  # Line that sources current to the load
            self.dependent_cmpnts = [line]

    def alpha_beta_dynamics(self, x=None, t=None, u=None):
        if u is None:
            i = self.line.i_alpha_beta()
        else:
            u = u[0]
            i = AlphaBeta(u[0], u[1], 0)
        v_alpha = i.alpha * self.r
        v_beta = i.beta * self.r
        return np.array([v_alpha, v_beta]) - x

    def v_alpha_beta(self):
        return AlphaBeta(self.states[0,0], self.states[1,0], 0)


class System:
    def __init__(self,
                 components
                 ):
        self.components = components
        self.gather_cmp_states()  # TODO: Test if this works
        curr_state_idx = 0
        for cmpnt in components:
            cmpnt.state_idx = curr_state_idx
            curr_state_idx += cmpnt.n_states

    def gather_cmp_states(self):
        self.states = np.vstack([cmpnt.states for cmpnt in self.components])

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
                        u_cmpnt = dependent_cmpnt.collect_states()  # TODO: Change this to a function?
                    u_cmpnt_ref = convert_state_ref(u_cmpnt, dependent_cmpnt.ref, cmpnt.ref)
                    u.append(u_cmpnt_ref)  # TODO: Or use extend and get a full list?
            else:
                u = None
            if issubclass(type(cmpnt), Component):
                cmpnt_dxdt = cmpnt.dynamics(cmpnt_states, t, u=u)
                system_dynamics.extend(cmpnt_dxdt)
            elif issubclass(type(cmpnt), Meter):  # TODO: Unsure how to deal with the variable step
                meas_dot = cmpnt.measure(cmpnt_states, t, u=u)
                # meas_delta = meas - cmpnt_states
                system_dynamics.extend(meas_dot)

        return system_dynamics

    def step_states(self):
        for cmpnt in self.components:
            # TODO: Make this directly put the new states values in the current index of the cmpnt states?
            cmpnt.states = self.states[cmpnt.state_idx: cmpnt.state_idx + cmpnt.n_states,:]
            cmpnt.step_states()


class Meter:  # TODO: Should this be a subclass of a component for use in a System?
    def __init__(self):
        pass

    def measure(self, x, t, u):
        return None

    def step_states(self):
        self.states[:, [1, 0]] = self.states[:, [0, 1]]


class PowerMeter(Meter):
    def __init__(self, volt_src, curr_src, wc):
        super().__init__()
        self.volt_src = volt_src
        self.curr_src = curr_src
        self.wc = wc
        self.dependent_cmpnts = [volt_src, curr_src]
        self.n_states = 2
        self.states = np.array([[0, None], [0, None]])
        self.ref = RefFrames.ALPHA_BETA  # Power calculations done in alpha-beta reference frame

    def measure(self, x, t, u=None):
        v = self.volt_src.v_alpha_beta() if u is None else AlphaBeta(u[0][0], u[0][1], 0)
        i = self.curr_src.i_alpha_beta() if u is None else AlphaBeta(u[1][0], u[1][1], 0)
        p, q = calculate_power(v, i)
        pdot = self.wc * (p - x[0])
        qdot = self.wc * (q - x[1])
        return pdot, qdot
