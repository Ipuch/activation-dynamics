import platform

import numpy as np

from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsList,
    ConfigureProblem,
    DynamicsFcn,
    DynamicsFunctions,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    NonLinearProgram,
    Solver,
    DynamicsEvaluation,
    PhaseDynamics,
    InterpolationType,
    ObjectiveList,
    InitialGuessList,
    PhaseTransitionList,
    PhaseTransitionFcn,
    QuadratureRule,
    ControlType,
    MultinodeConstraintFcn,
    MultinodeConstraintList,
)

from casadi import sin, MX, Function, SX, vertcat
from typing import Callable

from .constants import MODEL_FUNCTIONS
from .custom_multinode_constraint import mean_value_overtime


class ActivationModel:
    """
    This is a custom model that inherits from bioptim.CustomModel
    As CustomModel is an abstract class, some methods must be implemented.
    """

    def __init__(self, model_type: str):
        self.model = MODEL_FUNCTIONS[model_type]
        self.t_act = 0.015
        self.t_deact = 0.060
        self._symbolic_variable()

    def _symbolic_variable(self):
        self.a_sym = MX.sym("a", 1)
        self.u_sym = MX.sym("u", 1)

    def serialize(self) -> tuple[Callable, dict]:
        pass

    @property
    def name(self) -> str:
        return "ActivationDynamics"

    @property
    def name_dof(self):
        return ["excitation", "activation"]

    def activation_dynamics(self) -> Callable:
        return Function(
            "activation_dynamics",
            [self.a_sym, self.u_sym],
            [self.model(self.a_sym, self.t_act, self.t_deact, self.u_sym)],
        )


def custom_dynamics(
    time: MX | SX,
    states: MX | SX,
    controls: MX | SX,
    parameters: MX | SX,
    algebraic_states: MX | SX,
    numerical_timeseries: MX | SX,
    nlp: NonLinearProgram,
) -> DynamicsEvaluation:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

    Parameters
    ----------
    time: MX | SX
        The time of the system
    states: MX | SX
        The state of the system
    controls: MX | SX
        The controls of the system
    parameters: MX | SX
        The parameters acting on the system
    algebraic_states: MX | SX
        The algebraic states of the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[MX | SX] format
    """

    a = DynamicsFunctions.get(nlp.states["a"], states)
    # u = DynamicsFunctions.get(nlp.states["u"], states)
    # udot = DynamicsFunctions.get(nlp.controls["udot"], controls)
    u = DynamicsFunctions.get(nlp.controls["u"], controls)

    dxdt = nlp.model.activation_dynamics()(a, u)
    # dxdt = vertcat(dxdt, udot)

    return DynamicsEvaluation(dxdt=dxdt, defects=None)


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram, numerical_data_timeseries=None):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    name = "a"
    name_a = ["activation"]
    ConfigureProblem.configure_new_variable(
        name,
        name_a,
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
    )

    name = "u"
    name_u = ["excitation"]
    ConfigureProblem.configure_new_variable(
        name,
        name_u,
        ocp,
        nlp,
        as_states=False,
        as_controls=True,
        # as_states=True,
        # as_controls=False,
    )

    # name = "udot"
    # name_u = ["excitation_dot"]
    # ConfigureProblem.configure_new_variable(
    #     name,
    #     name_u,
    #     ocp,
    #     nlp,
    #     as_states=False,
    #     as_controls=True,
    # )

    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamics)


def prepare_ocp(
    model_type: str,
    n_shooting: int = 100,
    ode_solver: OdeSolverBase = OdeSolver.RK2(n_integration_steps=2),
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    model_type: str
        The type of model to use among McLean2003, McLean2003Improved, DeGroote2016Original, DeGroote2016
    ode_solver: OdeSolverBase
        The type of ode solver used
    use_sx: bool
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # BioModel path
    bio_model = ActivationModel(model_type)

    # Problem parameters
    n_shooting = n_shooting
    final_time = 1

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="u", weight=1, quadratic=True)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="u", weight=1, quadratic=True)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
        # ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="u",
        weight=1,
        quadratic=True,
        integration_rule=QuadratureRule.TRAPEZOIDAL,
    )
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="a", weight=1, quadratic=True, target=0.5)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(
        custom_configure,
        dynamic_function=custom_dynamics,
        expand_dynamics=expand_dynamics,
        # phase_dynamics=phase_dynamics,
        phase_dynamics=PhaseDynamics.ONE_PER_NODE,
    )

    # Constraints
    constraints = ConstraintList()
    # constraints.add(ConstraintFcn.TRACK_STATE, node=Node.MID, key="a", target=0.5)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("a", min_bound=np.array([[0, 0, 0]]), max_bound=np.array([[1, 1, 1]]))
    # x_bounds.add("u", min_bound=np.array([[0, 0, 0]]), max_bound=np.array([[1, 1, 1]]))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add("u", min_bound=np.array([[0, 0, 0]]), max_bound=np.array([[0, 1, 1]]))

    from numpy.random import random

    u_init = InitialGuessList()
    # u_init.add(
    #     "udot", initial_guess=(random((1, n_shooting + 1)) - 0.5) * 200, interpolation=InterpolationType.EACH_FRAME
    # )
    u_initial_guess = np.zeros((1, n_shooting + 1))
    u_initial_guess[0, 0::2] = 1

    u_init.add("u", initial_guess=u_initial_guess, interpolation=InterpolationType.EACH_FRAME)

    x_init = InitialGuessList()
    x_init.add("a", initial_guess=random((1, n_shooting + 1)), interpolation=InterpolationType.EACH_FRAME)
    # alternatively one and zeros for u

    phase_transition = PhaseTransitionList()
    phase_transition.add(transition=PhaseTransitionFcn.CYCLIC, phase_pre_idx=0)

    multinode_constraints = MultinodeConstraintList()
    # hard constraint
    multinode_constraints.add(
        mean_value_overtime,
        nodes_phase=tuple([0 for _ in range(n_shooting)]),
        nodes=tuple([i for i in range(n_shooting)]),
        key="a",
        # target=0.5,
    )

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        constraints=constraints,
        phase_transitions=phase_transition,
        multinode_constraints=multinode_constraints,
        ode_solver=ode_solver,
        use_sx=use_sx,
        control_type=ControlType.LINEAR_CONTINUOUS,
        n_threads=8,
    )
