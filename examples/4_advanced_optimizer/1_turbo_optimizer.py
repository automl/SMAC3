# """
# Synthetic Function with TuRBO as optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# An example of applying SMAC with trust region BO (TuRBO) to optimize a
# synthetic function (2d rosenbrock function).

# Eriksson et al. Scalable Global Optimization via Local {Bayesian} Optimization,
# http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization.pdf

# TurBO gradually shrinks its search space to the vicinity of the optimum configuration that is ever optimized.
# TuRBO optimizer requires EPMChooserTurBO to suggest the next configuration. Currently, it only supports pure numerical
# hyperparameters.
# """
# from ConfigSpace import Configuration, ConfigurationSpace, Float

# # Import SMAC-utilities
# from smac.scenario import Scenario

# # Import components required by TuRBO optimizers
# from smac.facade.blackbox_facade import BlackBoxFacade
# from smac.main.turbo import TuRBOSMBO


# class Rosenbrock2D:
#     @property
#     def configspace(self) -> ConfigurationSpace:
#         cs = ConfigurationSpace(seed=0)
#         x0 = Float("x0", (-5, 10), default=-3)
#         x1 = Float("x1", (-5, 10), default=-4)
#         cs.add_hyperparameters([x0, x1])

#         return cs

#     def train(self, config: Configuration) -> float:
#         """The 2-dimensional Rosenbrock function as a toy model.
#         The Rosenbrock function is well know in the optimization community and
#         often serves as a toy problem. It can be defined for arbitrary
#         dimensions. The minimium is always at x_i = 1 with a function value of
#         zero. All input parameters are continuous. The search domain for
#         all x's is the interval [-5, 10].
#         """
#         x1 = config["x0"]
#         x2 = config["x1"]

#         cost = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
#         return cost


# class TuRBOFacade(BlackBoxFacade):
#     """A wrapper that allows to run TuRBO optimizer. Its arguments are described under smac.main.turbo.TuRBOSMBO"""

#     def _init_optimizer(
#         self,
#         length_init=0.8,
#         length_min=0.5**8,
#         length_max=1.6,
#         success_tol=3,
#         failure_tol_min=4,
#         n_init_x_params=2,
#         n_candidate_max=5000,
#     ) -> None:
#         self.optimizer = TuRBOSMBO(
#             length_init=length_init,
#             length_min=length_min,
#             length_max=length_max,
#             success_tol=success_tol,
#             failure_tol_min=failure_tol_min,
#             n_init_x_params=n_init_x_params,
#             n_candidate_max=n_candidate_max,
#             scenario=self._scenario,
#             stats=self.stats,
#             runner=self.runner,
#             initial_design=self.initial_design,
#             runhistory=self.runhistory,
#             runhistory_encoder=self.runhistory_encoder,
#             intensifier=self.intensifier,
#             model=self.model,
#             acquisition_function=self.acquisition_function,
#             acquisition_optimizer=self.acquisition_optimizer,
#             random_design=self.random_design,
#             seed=self.seed,
#         )


# if __name__ == "__main__":
#     exit()

#     model = Rosenbrock2D()

#     # Scenario object
#     scenario = Scenario(model.configspace, n_trials=100)

#     # Example call of the target function
#     default_value = model.train(model.configspace.get_default_configuration())
#     print(f"Default value: {round(default_value, 2)}")

#     # Optimize, using a SMAC-object
#     print("Optimizing! Depending on your machine, this might take a few minutes.")
#     smac = TuRBOFacade(
#         scenario=scenario,
#         target_function=model.train,
#         overwrite=True,
#     )
#     smac.optimize()

#     incumbent = smac.optimize()

#     incumbent_value = model.train(incumbent)
#     print(f"Incumbent value: {round(incumbent_value, 2)}")
