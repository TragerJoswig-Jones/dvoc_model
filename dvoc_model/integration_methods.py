import scipy.optimize
import numpy as np

""" Explicit Solvers """


# TODO: Make this equation modular / allow any number of components
def forward_euler_step(dt, components, set_states=True, update_states=False):
    dxs = []
    for component in components:
        dxdt = component.dynamics()
        dxs.append(dxdt * dt)
    if set_states:
        for component, dx in zip(components, dxs):
            component.states[:,1] = component.states[:,0] + dx
    if update_states:
        for component in components:
            component.step_states()
    return dxs


def backward_euler_step(dt, components, set_states=True, update_states=False):
    k1s = forward_euler_step(dt, components, set_states=set_states, update_states=False)  # Estimation Step
    dxs = []
    for component, k1 in zip(components, k1s):
        dx2 = component.dynamics(component.states[:,0] + k1)
        if set_states:
            component.states[:,1] = component.states[:,0] + dx2 * dt
        dxs.append(dx2 * dt)
    if update_states:
        for component in components:
            component.step_states()
    return dxs


def semi_implicit_euler_step(dt, components, set_states=True, update_states=False):
    k1s = forward_euler_step(dt, components, set_states=set_states, update_states=False)  # Estimation Step
    for k1 in k1s: k1[1] = 0  # Zero out step for v_beta / theta
    dxs = []
    for component, k1 in zip(components, k1s):
        k2dt = component.dynamics(component.states[:,0] + k1)
        k2dt[0] = 0
        dx = k1 + k2dt * dt
        if set_states:
            component.states[:,1] = component.states[:,0] + dx
        dxs.append(dx)
    if update_states:
        for component in components:
            component.step_states()
    return dxs


def rk2_step(dt, components, set_states=True, update_states=False):
    k1s = forward_euler_step(dt, components, set_states=set_states, update_states=False)  # Estimation Step

    dxs = []
    for component, k1 in zip(components, k1s):
        k2dt = component.dynamics(component.states[:,0] + k1)
        dx = (k1 + k2dt * dt) * 0.5  # k1 already took dt into account
        if set_states:
            component.states[:,1] = component.states[:,0] + dx
        dxs.append(dx)
    if update_states:
        for component in components:
            component.step_states()
    return dxs


def taylor2_step(dt, components, set_states=True, update_states=False):
    """ Only to be used with dVOC controller """
    dxs = []
    for component in components:
        k1dt = component.dynamics(x=component.states[:,0])
        k1ddt = component.ddot(x=component.states[:,0], xdot=k1dt)
        # v_norm = component.states[0,0]**2 + component.states[1,0]**2
        v, theta = component.v_polar()
        v_norm = 2 * v**2 if component.ref.value == 1 else v**2
        k = component.xi / component.k_v**2
        dx = (k1dt + k1ddt * dt * 0.5) * dt  # TODO: 1.9994: (120.0001) in iso, 1.985 match cont in cont_sim
        if set_states:
            component.states[:,1] = component.states[:,0] + dx
        dxs.append(dx)
    if update_states:
        for component in components:
            component.step_states()
    return dxs


def rk4_step(dt, components, set_states=True, update_states=False):
    k1s = []
    for component in components:
        dxdt = component.dynamics(component.states[:, 0])
        k1s.append(dxdt)
    k2s = []
    for component, k1 in zip(components, k1s):
        dxdt = component.dynamics(component.states[:,0] + k1 * dt / 2)
        k2s.append(dxdt)
    k3s = []
    for component, k2 in zip(components, k2s):
        dxdt = component.dynamics(component.states[:,0] + k2 * dt / 2)
        k3s.append(dxdt)
    k4s = []
    for component, k3 in zip(components, k3s):
        dxdt = component.dynamics(component.states[:,0] + k3 * dt)
        k4s.append(dxdt)

    dxs = []
    for component, k1, k2, k3, k4 in zip(components, k1s, k2s, k3s, k4s):
        dxdt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if set_states:
            component.states[:,1] = component.states[:,0]  + dxdt * dt
        dxs.append(dxdt * dt)
    if update_states:
        for component in components:
            component.step_states()
    return dxs


""" Implicit Solvers """


def tustin_step(dt, components, set_states=True, update_states=False):
    dxs = []
    for component in components:
        x0 = component.states[:, 0]
        x1 = component.tustin_step(x=x0)

        if set_states:
            component.states[:, 1] = x1  # step states
        dxs.append(x1 - x0)
    if update_states:
        for component in components:
            component.step_states()
    return dxs


def backward_step(dt, components, set_states=True, update_states=False):
    dxs = []
    for component in components:
        x0 = component.states[:, 0]
        x1 = component.backward_step(x=x0)

        if set_states:
            component.states[:, 1] = x1  # step states
        dxs.append(x1 - x0)
    if update_states:
        for component in components:
            component.step_states()
    return dxs


"""
# NUMBAKIT ODE SOLVERS
# TODO: Not working as dvoc_model framework is based around objects
#       Would need to have independent dynamic function for the whole system,
#       without relying on classes.
def nbkode_step(dt, components, set_states=True, update_states=False):
    dxs = []
    for component in components:
        x0 = component.states[:,0]
        solver = nbkode.ForwardEuler(component.dynamics, 0.0, x0)
        ts, xs = solver.run([0, dt])
        x1 = xs[-1]
        dxs.append(x1 - x0)
    if set_states:
        for component, dx in zip(components, dxs):
            component.states[:,1] = component.states[:,0] + dx
    if update_states:
        for component in components:
            component.step_states()
    return dxs
"""
