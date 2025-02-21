from bioptim import PenaltyController, BiMapping
from casadi import MX


def mean_value_overtime(
    controllers: list[PenaltyController],
    key: str,
    # target: float,
) -> MX:
    """
    Mean value of a variable over time through all the phases with the trapezoidal rule

    Parameters
    ----------
    controllers: list[PenaltyController]
        All the controller for the penalties

    Returns
    -------
    The constraint such that: c(x) = 0
    """

    total_trapz = 0

    for c, controller in enumerate(controllers):

        u = controller.states[key]
        u_complete = u.mapping.to_second.map(u.cx)

        state_cx_end = controller.integrate(
            t_span=controller.t_span.cx,
            x0=controller.states.cx_start,
            u=controller.controls.cx_start,
            p=controller.parameters.cx,
            a=controller.algebraic_states.cx_start,
            d=controller.numerical_timeseries.cx_start,
        )["xf"]

        # get u in the state_cx_end vector now
        u_cx_end = state_cx_end[controller.states[key].index]

        local_trapz = (u_complete + u_cx_end) / 2 * controller.dt.cx
        total_trapz += local_trapz

    return 1 / controllers[0].tf.cx * total_trapz - 0.5
