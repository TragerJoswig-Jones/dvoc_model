import numpy as np
import pandas as pd
from copy import copy
from scipy import integrate
import matplotlib.pyplot as mp

from dvoc_model.reference_frames import SinCos, Abc, Dq0, AlphaBeta
from dvoc_model.constants import *
from elements import Grid, Line

""" Explicit Solvers """


# TODO: Make this equation modular / allow any number of components
def forward_euler_step(dt, components, set_states=True, update_states=False):
    # Todo: figure out how to set up trapezoidal
    # Call each component to get the dynamics for next step est, then calc dynamics at est step, then make trpzdl step
    dxs = []
    for component in components:
        dxdt = component.dynamics()
        dxs.append(dxdt * dt)
    if set_states:
        for component, dx in zip(components, dxs):
            component.states = component.states + dx
    if update_states:
        for component in components:
            component.update_states()
    return dxs


def backward_euler_step(dt, components, set_states=True, update_states=False):
    initial_states = []
    for component in components:
        initial_states.append(copy(component.states))
    k1s = forward_euler_step(dt, components, set_states=set_states, update_states=False)  # Estimation Step
    dxs = []
    for component, x0, k1 in zip(components, initial_states, k1s):
        dx2 = component.dynamics(x0 + k1)
        if set_states:
            component.states = x0 + dx2 * dt
        dxs.append(dx2 * dt)
    if update_states:
        for component in components:
            component.update_states()
    return dxs


def semi_implicit_euler_step(dt, components, set_states=True, update_states=False):
    initial_states = []
    for component in components:
        initial_states.append(copy(component.states))
    k1s = forward_euler_step(dt, components, set_states=set_states, update_states=False)  # Estimation Step
    for k1 in k1s: k1[1] = 0  # Zero out step for v_beta / theta
    dxs = []
    for component, x0, k1 in zip(components, initial_states, k1s):
        k2dt = component.dynamics(x0 + k1)
        k2dt[0] = 0
        dx = k1 + k2dt * dt
        if set_states:
            component.states = x0 + dx
        dxs.append(dx)
    if update_states:
        for component in components:
            component.update_states()
    return dxs


def rk2_step(dt, components, set_states=True, update_states=False):
    initial_states = []
    for component in components:
        initial_states.append(copy(component.states))
    k1s = forward_euler_step(dt, components, set_states=set_states, update_states=False)  # Estimation Step

    dxs = []
    for component, x0, k1 in zip(components, initial_states, k1s):
        k2dt = component.dynamics(x0 + k1)
        dx = (k1 + k2dt * dt) * 0.5  # k1 already took dt into account
        if set_states:
            component.states = x0 + dx
        dxs.append(dx)
    if update_states:
        for component in components:
            component.update_states()
    return dxs


def rk2_shift_step(dt, components, set_states=True, update_states=False):
    initial_states = []
    for component in components:
        initial_states.append(copy(component.states))
    k1s = forward_euler_step(dt, components, set_states=set_states, update_states=False)  # Estimation Step

    dxs = []
    for component, x0, k1 in zip(components, initial_states, k1s):
        k2dt = component.dynamics(x0 + k1)
        dx = (k1 + k2dt * dt) * 0.5  # k1 already took dt into account
        dx[0] += 0.008 * dt  # TEST: Apply LTE diff error here
        dx[1] -= 5e-5 * dt
        if set_states:
            component.states = x0 + dx
        dxs.append(dx)
    if update_states:
        for component in components:
            component.update_states()
    return dxs


def euler_rk2_step(dt, components, set_states=True, update_states=False):
    initial_states = []
    for component in components:
        initial_states.append(copy(component.states))
    k1s = forward_euler_step(dt, components, set_states=set_states, update_states=False)  # Estimation Step

    dxs = []
    for component, x0, k1 in zip(components, initial_states, k1s):
        k2dt = component.dynamics(x0 + k1)
        k2dt[0] = k1[0]  # First state uses forward euler
        dx = (k1 + k2dt * dt) * 0.5  # k1 already took dt into account
        if set_states:
            component.states = x0 + dx
        dxs.append(dx)
    if update_states:
        for component in components:
            component.update_states()
    return dxs


def rk4_step(dt, components, set_states=True, update_states=False):
    initial_states = []
    for component in components:
        initial_states.append(copy(component.states))
    k1s = []
    for component, x0 in zip(components, initial_states):
        dxdt = component.dynamics(x0)
        k1s.append(dxdt)
    k2s = []
    for component, k1, x0 in zip(components, k1s, initial_states):
        dxdt = component.dynamics(x0 + k1 * dt / 2)
        k2s.append(dxdt)
    k3s = []
    for component, k2, x0 in zip(components, k2s, initial_states):
        dxdt = component.dynamics(x0 + k2 * dt / 2)
        k3s.append(dxdt)
    k4s = []
    for component, k3, x0 in zip(components, k3s, initial_states):
        dxdt = component.dynamics(x0 + k3 * dt)
        k4s.append(dxdt)

    dxs = []
    for component, init_state, k1, k2, k3, k4 in zip(components, initial_states, k1s, k2s, k3s, k4s):
        dxdt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if set_states:
            component.states = init_state + dxdt * dt
        dxs.append(dxdt * dt)
    if update_states:
        for component in components:
            component.update_states()
    return dxs


""" Implicit Solvers """


def tustin_step(dt, components, set_states=True, update_states=False):
    dxs = []
    for component in components:
        x0 = component.states
        x1 = component.tustin_dynamics(x0, dt)

        if set_states:
            component.states = x1  # step states
        dxs.append(x1 - x0)
    if update_states:
        for component in components:
            component.update_states()
    return dxs


def backward_step(dt, components, set_states=True, update_states=False):
    dxs = []
    for component in components:
        x0 = component.states
        x1 = component.backward_dynamics(x0, dt)

        if set_states:
            component.states = x1  # step states
        dxs.append(x1 - x0)
    if update_states:
        for component in components:
            component.update_states()
    return dxs


""" ODE Solvers """


# Takes a component and updates its states based on its dynamics and sampled zero-order hold values of values that
# impact these dynamics
def ode_solver_step(dt, components, params={}, set_states=True, update_states=False):
    dxs = []
    rtol = params['rtol'] if 'rtol' in params else 1e-10
    atol = params['atol'] if 'atol' in params else 1e-10
    # TODO: Look into other parameters and tolerances of ode solver
    for component in components:
        x0, x1 = integrate.odeint(component.dynamics, component.states, [0, dt], rtol=rtol, atol=atol)
        # TODO extract all info from odeint and somehow save it
        if set_states:
            component.states = x1  # step states
        dxs.append(x1 - x0)
    if update_states:
        for component in components:
            component.update_states()
    return dxs


def ode_solver_system(dt, system, params={}, set_states=True, update_states=False):
    dxs = []
    rtol = params['rtol'] if 'rtol' in params else 1e-10
    atol = params['atol'] if 'atol' in params else 1e-10
    # TODO: Look into other parameters and tolerances of ode solver
    # TODO extract all info from odeint and somehow save it
    result = integrate.solve_ivp(system.dynamics, [0, dt], system.states, method='RK45',
                                            rtol=rtol, atol=atol, vectorized=True)

    t = result.t
    y = result.y
    x0 = result.y[:, 0]
    x1 = result.y[:, -1]
    dxs.append(x1 - x0)

    if set_states:
        system.states = x1  # step states

    if update_states:
        system.update_states()

    return dxs, t, y


def simulate(controller, p_refs, q_refs, dt=1 / 10e3, t=500e-3, Lf=1.5e-3, Rf=0.8,
             grid=Dq0(SQRT_2 * 120, 0, 0), grid_omega=TWO_PI * 60, id0=0, iq0=0):
    ts = np.arange(0, t, dt)

    # dictionary for containing simulation results
    euler_data = {'v_a': [],
                  'v_b': [],
                  'v_c': [],
                  'i_a': [],
                  'i_b': [],
                  'i_c': [],
                  'p': [],
                  'q': []}

    grid = Grid(SQRT_2 * 120., 0., grid_omega)
    line = Line(controller, grid, Rf, Lf)
    controller.line = line
    discrete_components = [grid, controller]
    components = [line]

    # run simulation
    for p_ref, q_ref, t in zip(p_refs, q_refs, ts):
        forward_euler_step(dt, components)

        # update the data
        controller.p_ref = p_ref
        controller.q_ref = q_ref
        v = AlphaBeta.from_polar(controller.v, controller.theta)
        v_abc = v.to_abc()
        i_abc = line.i.to_abc()
        data = euler_data
        data['v_a'].append(v_abc.a)
        data['v_b'].append(v_abc.b)
        data['v_c'].append(v_abc.c)
        data['i_a'].append(i_abc.a)
        data['i_b'].append(i_abc.b)
        data['i_c'].append(i_abc.c)
        data['p'].append(controller.p)
        data['q'].append(controller.q)

    # plot the results
    data = pd.DataFrame(index=ts, data=euler_data)
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
