import numpy as np
import pandas as pd
from scipy import integrate

from dvoc_model.reference_frames import AlphaBeta, RefFrames
from dvoc_model.constants import TWO_PI
from dvoc_model.elements import Grid, Line, LineToGrid
from dvoc_model.calculations import calculate_power, shift_angle


""" ODE Solvers """


# Takes a component and updates its states based on its dynamics and sampled zero-order hold values of values that
# impact these dynamics. Does not continuously sample the current.
def ode_solver_step(dt, components, params={}, set_states=True, update_states=False):
    dxs = []
    rtol = params['rtol'] if 'rtol' in params else 1e-10
    atol = params['atol'] if 'atol' in params else 1e-10
    max_step = params['hmax'] if 'hmax' in params else np.inf
    for component in components:
        x0, x1 = integrate.odeint(component.dynamics, component.states[:,0].tolist(), [0, dt],
                                  rtol=rtol, atol=atol, hmax=max_step)
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
    max_step = params['hmax'] if 'hmax' in params else np.inf
    # TODO: extract all info from odeint and return it
    result = integrate.solve_ivp(system.dynamics, [0, dt], system.states[:,0].tolist(), method='DOP853',  # TODO: Select method,  ‘DOP853’ or 'RK45'
                                 rtol=rtol, atol=atol, max_step=max_step, vectorized=True)

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


def continuous_step(dt, components):
    """ Empty function for continuous controller """
    return None


def simulate(controller, p_refs, q_refs, dt=1 / 10e3, t=500e-3, Lf=1.5e-3, Rf=0.8,
             grid_omega=TWO_PI * 60, v_nom=120., id0=0, iq0=0, discretization_step=ode_solver_step):
    ts = np.arange(0, t, dt)

    atol = 1e-3
    rtol = 1e-3
    params = {'rtol': rtol, 'atol': atol}

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
        i = line.i_alpha_beta()
        i_abc = i.to_abc()
        p, q = calculate_power(v, i)
        data['v_a'].append(v_abc.a)
        data['v_b'].append(v_abc.b)
        data['v_c'].append(v_abc.c)
        data['i_a'].append(i_abc.a)
        data['i_b'].append(i_abc.b)
        data['i_c'].append(i_abc.c)
        data['p'].append(p)
        data['q'].append(q)

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


def shift_controller_angle(controller, ref, omega_nom, dt, periods):
    """ Shifts the angle of the given controller voltage by half of a step according to the nominal frequency.
        This shifts the controllers voltage to be closer to the equilibrium if the output is a zero order hold.
    """
    if ref is RefFrames.POLAR:
        controller.states[1,0] += omega_nom * periods * dt  # (+) Shifts the controller voltage waveform left / backward
    else:
        vrms = np.sqrt(controller.states[0,0]**2 + controller.states[1,0]**2) / np.sqrt(2)
        theta = np.atan2(controller.states[1,0] / controller.states[0,0])
        v = AlphaBeta.from_polar(vrms, theta + omega_nom * periods * dt)
        controller.states[0,0] = v.alpha
        controller.states[1,0] = v.beta


def collect_sim_info(data_dict, controller, grid, line, method,
                     ode_dict=None, t_ode=None, y_ode=None, continuous=False):
    ref_frame = controller.ref

    # update the data
    v, theta = controller.v_polar(internal=False)
    vg = grid.v_alpha_beta(internal=False)
    vab = controller.v_alpha_beta(internal=False)
    iab = line.i_alpha_beta(internal=False)
    i_grid_ab = line.states[4:6, 0]
    iab_grid = AlphaBeta(i_grid_ab[0], i_grid_ab[1], 0)
    v_abc = vab.to_abc()
    i_abc = iab.to_abc()
    p, q = calculate_power(vab, iab)
    if not continuous:
        #p, q = calculate_power(shift_angle(vab, -controller.omega_nom * controller.dt / 2), iab)
        # theta = theta - controller.omega_nom * controller.dt / 2  # Shift to theta at center of stepped waveform steps
        pass
    data_dict['v_alpha, %s, %s' % (method, ref_frame.name)].append(vab.alpha)
    data_dict['v_beta, %s, %s' % (method, ref_frame.name)].append(vab.beta)
    data_dict['v, %s, %s' % (method, ref_frame.name)].append(v)
    data_dict['theta, %s, %s' % (method, ref_frame.name)].append(theta)
    data_dict['vg_alpha, %s, %s' % (method, ref_frame.name)].append(vg.alpha)
    data_dict['vg_beta, %s, %s' % (method, ref_frame.name)].append(vg.beta)
    data_dict['v_a, %s, %s' % (method, ref_frame.name)].append(v_abc.a)
    data_dict['v_b, %s, %s' % (method, ref_frame.name)].append(v_abc.b)
    data_dict['v_c, %s, %s' % (method, ref_frame.name)].append(v_abc.c)
    data_dict['i_alpha, %s, %s' % (method, ref_frame.name)].append(iab.alpha)
    data_dict['i_beta, %s, %s' % (method, ref_frame.name)].append(iab.beta)
    data_dict['i_a, %s, %s' % (method, ref_frame.name)].append(i_abc.a)
    data_dict['i_b, %s, %s' % (method, ref_frame.name)].append(i_abc.b)
    data_dict['i_c, %s, %s' % (method, ref_frame.name)].append(i_abc.c)
    data_dict['p, %s, %s' % (method, ref_frame.name)].append(p)
    data_dict['q, %s, %s' % (method, ref_frame.name)].append(q)
    data_dict['ig_alpha, %s, %s' % (method, ref_frame.name)].append(iab_grid.alpha)  # TEST
    data_dict['ig_beta, %s, %s' % (method, ref_frame.name)].append(iab_grid.beta)  # TEST

    if ode_dict is not None:
        ode_dict['T_ode, %s, %s' % (method, ref_frame.name)] = t_ode
        ode_dict['Y_ode, %s, %s' % (method, ref_frame.name)] = y_ode

    return p, q
