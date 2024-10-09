Adaptive Capping
=================

When minimizing the runtime of an algorithm across multiple instances is the objective, we can use adaptive capping to
allocate budgets to configurations dynamically.


Setting up Capping
--------------------------

Capping is useful, when given that we have seen an incumbent configuration
perform on a subset of instances, we can expect that a challenger configurations will not outperform the incumbent anymore,
because the challenger already accumulated too much runtime across instances to be compatible.
In that case we want to stop the evaluation of the challenger configuration early to save resources.

To achieve this, the user must take active care in the termination of their target function.
The capped problem.train will receive a budget keyword argument, detailing the seconds allocated to the configuration.
Below is an example of a capped problem that will return the used budget if the computation exceeds the budget.


.. code-block:: python

    class TimeoutException(Exception):
        pass


    @contextmanager
    def timeout(seconds):
        def handler(signum, frame):
            raise TimeoutException(f"Function call exceeded timeout of {seconds} seconds")

        # Set the signal handler for the alarm signal
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)  # Schedule an alarm after the given number of seconds

        try:
            yield
        finally:
            # Cancel the alarm if the block finishes before timeout
            signal.alarm(0)


    class CappedProblem:
        @property
        def configspace(self) -> ConfigurationSpace:
            ...

        def train(self, config: Configuration, instance:str, budget, seed: int = 0) -> float:

            try:
                with timeout(int(math.ceil(budget))):
                    start_time = time.time()
                    ... # heavy computation
                    runtime = time.time() - start_time
                    return runtime
            except TimeoutException as e:
                print(f"Timeout for configuration {config} with runtime budget {budget}")
                return budget # here the runtime is capped and we return the used budget.


In order to enable adaptive capping in smac, we need to create problem instances :doc:`../4_instances` to optimize over ( and specify a
global runtime cutoff in the intensifier. Then we optimize as usual.


 .. code-block:: python



    from smac.intensifier import Intensifier
    from smac.scenario.scenario import Scenario

    scenario = Scenario(
        capped_problem.configspace,
        ...
        instances=['1', '2', '3'], # add problem instances we want to solve
        instance_features={'1': [1], '2': [2], '3': [3]} # in the absence of actual features add dummy features for identification
    )

    intensifier = Intensifier(
    scenario,
    runtime_cutoff=10 # specify an absolute runtime cutoff (sum over instances) never to be exceeded
    )

    smac = HyperparameterOptimizationFacade(
        scenario,
        capped_problem.train,
        intensifier=intensifier,
        ...
    )

    incumbent = smac.optimize()


