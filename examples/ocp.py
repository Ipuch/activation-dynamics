from bioptim import Solver, OdeSolver, OnlineOptim, CostType, SolutionMerge

from src.bioptim_ocp import prepare_ocp


def main():
    """
    Runs the optimization and animates it
    """

    model_type = "McLean2003"
    # model_type = "McLean2003Improved"
    # model_type = "DeGroote2016Original"
    # model_type = "DeGroote2016"

    ocp = prepare_ocp(
        model_type=model_type,
        n_shooting=10000,
        # ode_solver=OdeSolver.COLLOCATION(polynomial_degree=9),
        # ode_solver=OdeSolver.COLLOCATION(polynomial_degree=2),
        ode_solver=OdeSolver.TRAPEZOIDAL(),
        # ode_solver=OdeSolver.RK1(n_integration_steps=2),
    )
    # ocp.add_plot_penalty(CostType.ALL)
    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
    sol.print_cost()
    print(sol.decision_states(to_merge=SolutionMerge.NODES))
    sol.graphs()


if __name__ == "__main__":
    main()
