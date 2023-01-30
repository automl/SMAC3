# """
# Synthetic Function with BOinG as optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# An example of applying SMAC with BO inside Grove (BOinG) to optimize a
# synthetic function (2d rosenbrock function).

# BOinG optimizer requires a SMAC4BOING wrapper to optimize the target function. It is a two stage BO algorithm.
# In the first stage, BOinG constructs an RF to capture the global loss landscape. Then in the second stage, it only
# optimizes inside a subregion near the candidate suggested by the RF model with a GP model to focus only on the most
# promising region.
# """

# # Import SMAC-utilities
# from smac.scenario import Scenario

# from smac.facade.boing_facade import BOinGFacade

# # from .syn_func import Rosenbrock2D

# from ConfigSpace import Configuration, ConfigurationSpace, Float


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
#     smac = BOinGFacade(
#         scenario=scenario,
#         target_function=model.train,
#         overwrite=True,
#     )

#     incumbent = smac.optimize()

#     incumbent_value = model.train(incumbent)
#     print(f"Incumbent value: {round(incumbent_value, 2)}")
