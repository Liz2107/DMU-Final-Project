import numpy as np
from scipy.integrate import solve_ivp

def ivp(yp, y0, tspan, integrator, **kwargs):
    rtol = 1E-10
    max_step = np.abs(tspan[1]) / 100
    jac = np.array(np.nan)

    if 'rtol' in kwargs:
        rtol = kwargs['rtol']
        del kwargs['rtol']
    if 'max_step' in kwargs:
        max_step = kwargs['max_step']
        del kwargs['max_step']
    if 'jac' in kwargs:
        jac = kwargs['jac']
        del kwargs['jac']
    if 'events' in kwargs:
        events = kwargs['events']
        del kwargs['events']
    else:
        events = None
    
    match integrator:
        case 'RK45':
            return solve_ivp(yp, tspan, y0, method='RK45', args=kwargs.values(), rtol=rtol, max_step=max_step, events=events)
        case 'RK23':
            return solve_ivp(yp, tspan, y0, method='RK23', args=kwargs.values(), rtol=rtol, max_step=max_step, events=events)
        case 'DOP853':
            return solve_ivp(yp, tspan, y0, method='DOP853', args=kwargs.values(), rtol=rtol, max_step=max_step, events=events)
        case 'Radau':
            return solve_ivp(yp, tspan, y0, method='Radau', args=kwargs.values(), rtol=rtol, max_step=max_step, jac=jac, events=events)
        case 'BDF':
            return solve_ivp(yp, tspan, y0, method='BDF', args=kwargs.values(), rtol=rtol, max_step=max_step, jac=jac, events=events)
        case 'LSODA':
            return solve_ivp(yp, tspan, y0, method='LSODA', args=kwargs.values(), rtol=rtol, max_step=max_step, jac=jac, events=events)
        