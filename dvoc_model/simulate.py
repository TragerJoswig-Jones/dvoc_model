import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as mp

from dvoc_model.reference_frames import SinCos, Abc, Dq0, AlphaBeta
from dvoc_model.constants import TWO_PI
from dvoc_model.elements import Grid, Line
from dvoc_model.integration_methods import forward_euler_step


""" ODE Solvers """


# Takes a component and updates its states based on its dynamics and sampled zero-order hold values of values that
# impact these dynamics
def ode_solver_step(dt, components, params={}, set_states=True, update_states=False):
    dxs = []
    rtol = params['rtol'] if 'rtol' in params else 1e-10
    atol = params['atol'] if 'atol' in params else 1e-10
    for component in components:
        x0, x1 = integrate.odeint(component.dynamics, component.states[:,0].tolist(), [0, dt], rtol=rtol, atol=atol)
        # TODO extract all info from odeint and return it
        if set_states:
            component.states[:,1] = x1  # step states
        dxs.append(x1 - x0)
    if update_states:
        for component in components:
            component.step_states()
    return dxs


def ode_solver_system(dt, system, params={}, set_states=True, update_states=False):
    dxs = []
    rtol = params['rtol'] if 'rtol' in params else 1e-10
    atol = params['atol'] if 'atol' in params else 1e-10
    # TODO: extract all info from odeint and return it
    result = integrate.solve_ivp(system.dynamics, [0, dt], system.states[:,0].tolist(), method='RK45',
                                            rtol=rtol, atol=atol, vectorized=True)

    t = result.t
    y = result.y
    x0 = result.y[:, 0]
    x1 = result.y[:, -1]
    dxs.append(x1 - x0)

    if set_states:
        system.states[:,1] = x1  # step states

    if update_states:
        system.step_states()

    return dxs, t, y


def simulate(controller, p_refs, q_refs, dt=1 / 10e3, t=500e-3, Lf=1.5e-3, Rf=0.8,
             grid_omega=TWO_PI * 60, id0=0, iq0=0, discretization_step=ode_solver_step):
    ts = np.arange(0, t, dt)

    # dictionary for containing simulation results
    data = {'v_a': [],
            'v_b': [],
            'v_c': [],
            'i_a': [],
            'i_b': [],
            'i_c': [],
            'p': [],
            'q': []}

    grid = Grid(v_nom, 0., grid_omega)
    line = Line(controller, grid, Rf, Lf)
    controller.line = line  # Set the line that the GFM is connected to
    system = LineToGrid(line, grid)
    discrete_components = [controller]

    # run simulation
    for p_ref, q_ref, t in zip(p_refs, q_refs, ts):
        dxs, t_ode, y_ode = ode_solver_system(dt, system, params=params)
        discretization_step(dt, discrete_components)

        # Update states to calculated step values
        system.step_states()
        for component in discrete_components:
            component.step_states()

        # update the data
        controller.p_ref = p_ref
        controller.q_ref = q_ref
        v = controller.v_alpha_beta()
        v_abc = v.to_abc()
        i_abc = line.i_alpha_beta().to_abc()
        data['v_a'].append(v_abc.a)
        data['v_b'].append(v_abc.b)
        data['v_c'].append(v_abc.c)
        data['i_a'].append(i_abc.a)
        data['i_b'].append(i_abc.b)
        data['i_c'].append(i_abc.c)
        data['p'].append(controller.p)
        data['q'].append(controller.q)

    # plot the results
    data = pd.DataFrame(index=ts, data=data)
    plot_current = True
    plot_voltage = True
    plot_power = True

    if plot_current:
        ax = data.plot(y='i_a')
        data.plot(y='i_b', ax=ax)
        data.plot(y='i_c', ax=ax)
    # mp.ylim(-10,10)

    if plot_voltage:
        ax = data.plot(y='v_a')
        data.plot(y='v_b', ax=ax)
        data.plot(y='v_c', ax=ax)

    if plot_power:
        ax = data.plot(y='p')
        data.plot(y='q', ax=ax)

    # mp.show()
    return data
