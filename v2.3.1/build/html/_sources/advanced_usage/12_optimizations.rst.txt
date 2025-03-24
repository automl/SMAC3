Optimizations
=============

SMAC might run faster or slower depending on the user specifications. In general it
applies that the more you know about the underlying target function, the better you can optimize the optimization
process.

The following list might help you to make the optimization process more efficient:

- Intensifier -> ``max_config_calls``: Higher numbers lead to less configurations.
- ConfigSelector -> ``retrain_after``: The lower the number, the more often the model is retrained. Recommendation:

  - High target function evaluation times: Low ``retrain_after`` (e.g., 1).
  - Low target function evaluation times: High ``retrain_after`` (e.g., 8).

- Scenario -> ``n_workers``: The higher the number, the more configurations are evaluated in parallel. Recommendation:

  - High target function evaluation times: As many ``n_workers`` as cores.
  - Low target function evaluation times: Only one worker because the communication might take longer than evaluating
    on a single thread.