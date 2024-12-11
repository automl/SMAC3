Adaptive Capping
=================

Adaptive capping is a feature that can be used to speedup the evaluation of candidate configurations when the objective
is to minimize runtime of an algorithm across a set of instances. The basic idea is to terminate unpromising candidates
early and adapting the timeout for solving a single instance dynamically based on the incumbent's runtime and the.
runtime already used by the challenging configuration.

Theoretical Background
----
When comparing a challenger configuration with the current incumbent for a (sub-)set of instances, we already know how
much cost (in terms of runtime) was incurred by the incumbent to solve the set of instances. As soon as the challenger
configuration exceeds the cost of the incumbent, it is evident that the challenger will not become the new incumbent
since the costs accumulate over time and are strictly positive, i.e., solving an instance cannot have negative runtime.

Example:
*Let the incumbent be evaluated for two instances with observed runtimes 3s and 4s. When a challenger configuration is
evaluated and compared against the incumbent, it is first evaluated on a first instance. For example, we observe a
runtime of 2s. As the challenger appears to be a promising configuration, its evaluation is intensified and the budget
is doubled, i.e., the budget is increased to 2. For solving the second instance, adaptive capping will allow a timeout
of 5s since the sum of runtimes for the incumbent is 7s and the challenger used up 2s for solving the first instance so
far so that 5s remain until the costs of the incumbent are exceeded. Even if the challenger configuration would need 10s
to solve the second instance, its execution would be aborted. In this example, by adaptive capping we thus save 5s of
evaluation costs for the challenger to notice that it will not replace the current incumbent.*

In combination with random online aggressive racing, we can further speedup the evaluation of challenger configurations
as we increase the horizon for adaptive capping step by step with every step of intensification. Note that
intensification will double the number of instances to which the challenger configuration (and eventually also the
incumbent configuration) are applied to. Furthermore, to increase the trust into the current incumbent, the incumbent is
regularly subject to intensification.


Setting up Adaptive Capping
--------------------------

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


