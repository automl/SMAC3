Problem: Parameters and :math: not working properly

Hypothesis: These seem to be sphinx (.rst) type, so that means SMAC might have at some point changed from 
sphinx to mkdocs, which is causing the problem.

Solution: That actually turned out to be true, SMAC changed from sphinx to mkdocs in v2.3.0 (commit 64058a9).
So the most straight forward option seems to be to change docstring_type to numpy (from google), and replace 
:math: with $$ which is supported in mkdocs.