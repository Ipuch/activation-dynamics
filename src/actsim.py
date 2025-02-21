import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from .constants import MODEL_FUNCTIONS

def actdyn(t, a, model, Tact, Tdeact, u):
    """
    Generic time-derivative function that delegates to one of the
    model-specific activation dynamics implementations.

    Parameters:
      t       : current time (unused by the model but required by ODE solver)
      a       : current activation level
      model   : string key identifying the activation model
      Tact    : time constant for activation
      Tdeact  : time constant for deactivation
      u       : control input (excitation)

    Returns:
      [adot]  : derivative of activation, as a one-element list
    """
    a_val = a[0] if isinstance(a, (list, np.ndarray)) else a

    if model not in MODEL_FUNCTIONS:
        raise ValueError(f"Unknown model: {model}")

    adot = MODEL_FUNCTIONS[model](a_val, Tact, Tdeact, u)
    return [adot]


def actsim(model='McLean2003',
           umax=1.0,
           uperiod=0.1,
           dutycycle=0.5,
           makeplot=True):
    """
    Simulate activation dynamics with a square-wave excitation input.

    Args:
      model      : One of ['McLean2003','McLean2003Improved',
                           'DeGroote2016Original','DeGroote2016']
      umax       : Peak (on) value of excitation (default 1.0)
      uperiod    : Period of the input square wave
      dutycycle  : Fraction of each period where excitation == umax
      makeplot   : If True, produce a plot of excitation and activation over time.

    Returns:
      mean_u         : Mean excitation over the final cycle
      mean_usquared  : Mean of (excitation^2) over the final cycle
      mean_a         : Mean activation over the final cycle
    """
    # Default activation/deactivation time constants (from the MATLAB code)
    Tact = 0.015
    Tdeact = 0.060

    sim_name = f"{model} with {1.0 / uperiod:.3f} Hz square wave input"

    # Simulate ~5 seconds to let any transients settle
    total_duration = 5.0
    ncycles = int(round(total_duration / uperiod))

    # Initialize solution arrays
    tt = [0.0]  # time
    aa = [0.0]  # activation
    uu = [0.0]  # excitation

    # Helper function to integrate the activation ODE over one phase
    def run_activation_phase(t_start, duration, a_init, u_val):
        """
        Integrate the ODE for 'duration' seconds, starting from a_init,
        with constant input u_val. Returns arrays of time, activation, excitation.
        """
        sol = solve_ivp(
            fun=lambda t, y: actdyn(t, y, model, Tact, Tdeact, u_val),
            t_span=(0, duration),
            y0=[a_init],
            max_step=duration / 50  # limit step size
        )

        # Local times
        t_local = sol.t.copy()
        if len(t_local) > 0:
            # Avoid duplication at the boundary for interpolation
            t_local[0] += 1e-10

        a_local = sol.y[0]
        # Shift time by t_start for continuity
        t_global = t_start + t_local
        # Build a constant-u array
        u_array = np.full_like(a_local, u_val)
        return t_global, a_local, u_array

    # Loop over cycles
    for _ in range(ncycles):
        # 1) "On" phase
        dur_on = dutycycle * uperiod
        t_on, a_on, u_on = run_activation_phase(tt[-1], dur_on, aa[-1], umax)
        tt.extend(t_on)
        aa.extend(a_on)
        uu.extend(u_on)

        # 2) "Off" phase
        dur_off = (1.0 - dutycycle) * uperiod
        t_off, a_off, u_off = run_activation_phase(tt[-1], dur_off, aa[-1], 0.0)
        tt.extend(t_off)
        aa.extend(a_off)
        uu.extend(u_off)

    # Convert to numpy arrays
    tt = np.array(tt)
    aa = np.array(aa)
    uu = np.array(uu)

    # Plot (last 10 cycles) if requested
    if makeplot:
        plt.figure(figsize=(8, 4))
        plt.plot(tt, uu, label='excitation')
        plt.plot(tt, aa, label='activation')
        plt.xlabel('time (s)')
        plt.title(sim_name)
        t_end = tt[-1]
        t_start_plot = max(0.0, t_end - 10.0 * uperiod)
        plt.xlim([t_start_plot, t_end])
        plt.ylim([-0.2, 1.2])
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Resample final cycle to compute mean values
    npoints = 1001
    t_period_start = tt[-1] - uperiod
    tnew = np.linspace(t_period_start, tt[-1], npoints)
    u_resamp = np.interp(tnew, tt, uu)
    a_resamp = np.interp(tnew, tt, aa)
    # Exclude last point to avoid overlap
    mean_u = np.mean(u_resamp[:-1])
    mean_usquared = np.mean(u_resamp[:-1] ** 2)
    mean_a = np.mean(a_resamp[:-1])

    # Print summary
    print(f"Simulation result for {sim_name}")
    print(f"    mean excitation:         {mean_u:7.4f}")
    print(f"    mean excitation-squared: {mean_usquared:7.4f}")
    print(f"    mean activation:         {mean_a:7.4f}")

    return mean_u, mean_usquared, mean_a


# Example usage (if running as a script):
if __name__ == "__main__":
    actsim()  # runs the default simulation
