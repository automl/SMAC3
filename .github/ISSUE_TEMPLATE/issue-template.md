---
name: Issue Template
about: Describe this issue template's purpose here.
labels: 

---

<!--
Please file an issue for bug reports (label as `bug`), usage questions (label as `question`), feature requests (label as `feature request`), to notify us about upcoming contributions and any other topic that you think may be important discussing with us.
-->

<!-- Instructions For Filing a Bug: https://github.com/automl/SMAC3/blob/master/CONTRIBUTING.md -->

#### Description
<!-- Example: error: argument --initial-incumbent/--initial_incumbent: invalid choice: 'sobol' -->

#### Steps/Code to Reproduce
<!--
Example:
```
from smac.facade.func_facade import fmin_smac

def rosenbrock_2d(x):
    x1 = x[0]
    x2 = x[1]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val

x, cost, _ = fmin_smac(func=rosenbrock_2d,
                       x0=[-3, -4],
                       bounds=[(-5, 5), (-5, 5)],
                       maxfun=325,
                       scenario_args={"initial_incumbent": "sobol"},
                       rng=3)
```
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->

#### Expected Results
<!-- Example: No error is thrown. Please paste or describe the expected results.-->

#### Actual Results
<!-- Please paste or specifically describe the actual output or traceback. -->

#### Versions
<!--
Please run the following snippet and paste the output below.
`import smac; print(smac.__version__)`
-->

<!-- Thanks for contributing! -->
