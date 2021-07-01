import time

# --------------------------------------------------------------
# When working with multiprocessing, we need to provide a pickable
# function and use __main__. Details can be found on:
# https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
# This function below is used by the rosenbrock function minimization parallel
# example, and  to be compliant with multiprocessing API,
# this function is implemented in its own file. Dask in particular,
# has a check that makes sure that rosenbrock_2d passed to smac
# is the same as the one passed to the workers. This is not the case
# if this function is in the fmin example directly as main().rosenbrock_2d
# is different than rosenbrock_2d.
# Below is a work-around to have a packaged function called rosenbrock_2d
# --------------------------------------------------------------


def rosenbrock_2d(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 10].
    """
    x1 = x[0]
    x2 = x[1]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    time.sleep(3)
    return val
