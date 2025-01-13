# Logging

Logging is a crucial part of the optimization, which should be customizable by the user. This page gives you the
overview how to customize the logging experience with SMAC.

## Level

The easiest way to change the logging behaviour is to change the level of the global logger. SMAC does this for you
if you specify the ``logging_level`` in any facade.

```python 
smac = Facade(
    ...
    logging_level=20,
    ...
)
```

The table shows you the specific levels:

| Name      | Level    |
|-----------|----------|
| 0         | SHOW ALL |
| 10        | DEBUG    |
| 20        | INFO     |
| 30        | WARNING  |
| 40        | ERROR    |
| 50        | CRITICAL |

## Standard Logging Files

By default, SMAC generates several files to document the optimization process. These files are stored in the directory structure `./output_directory/name/seed`, where name is replaced by a hash if no name is explicitly provided. This behavior can be customized through the [Scenario][smac.scenario] configuration, as shown in the example below:
```python
Scenario(
    configspace = some_configspace,
    name = 'experiment_name',
    output_directory = Path('some_directory'),
    ...
)
```
Notably, if an output already exists at `./some_directory/experiment_name/seed`, the behavior is determined by the overwrite parameter in the [facade's][smac/facade/abstract_facade] settings. This parameter specifies whether to continue the previous run (default) or start a new run.

The output is split into four different log files, and a copy of the utilized [Configuration Space of the ConfigSpace library](https://automl.github.io/ConfigSpace/latest/).

### intensifier.json
The [intensification][Intensification] is logged in `intensifier.json` and has the following structure:

```json
{
  "incumbent_ids": [
    65
  ],
  "rejected_config_ids": [
    1,
  ],
  "incumbents_changed": 2,
  "trajectory": [
    {
      "config_ids": [
        1
      ],
      "costs": [
        0.45706284046173096
      ],
      "trial": 1,
      "walltime": 0.029736042022705078
    },
    #...
  ],
  "state": {
    "tracker": {},
    "next_bracket": 0
  }
}
```

### optimization.json
The optimization process is portrayed in `optimization.json` with the following structure

```json
{
  "used_walltime": 184.87366724014282,
  "used_target_function_walltime": 20.229533672332764,
  "last_update": 1732703596.5609574,
  "finished": false
}
``` 
### runhistory.json
The runhistory.json in split into four parts. `stats`, `data`, `configs`, and `config_origins`.
`stats` contains overall broad stats on the different evaluated configurations:
```json
  "stats": {
    "submitted": 73,
    "finished": 73,
    "running": 0
  },
```

`data` contains a list of entries, one for each configuration.
```json
  "data": [
    {
      "config_id": 1,
      "instance": null,
      "seed": 209652396,
      "budget": 2.7777777777777777,
      "cost": 2147483647.0,
      "time": 0.0,
      "cpu_time": 0.0,
      "status": 0,
      "starttime": 0.0,
      "endtime": 0.0,
      "additional_info": {}
    },
    ...
  ]

```

`configs` is a human-readable dictionary of configurations, where the keys are the one-based `config_id`. It is important to note that in `runhistory.json`, the indexing is zero-based.
```json
  "configs": {
    "1": {
      "x": -2.3312147893012
    },
```

Lastly, `config_origins` specifies the source of a configuration, indicating whether it stems from the initial design or results from the maximization of an acquisition function.
```json
  "config_origins": {
    "1": "Initial Design: Sobol",
    ...
  }
```

### scenario.json
The ´scenario.json´ file contains the overall state of the [Scenario][smac.scenario] logged to a json file.

## Custom File

Sometimes, the user wants to disable or highlight specify modules. You can do that by passing a custom yaml
file to the facade instead.

```python 
smac = Facade(
    ...
    logging_level="path/to/your/logging.yaml",
    ...
)
```

The following file shows you how to display only error messages from the intensifier 
but keep the level of everything else on INFO:

```yaml
version: 1
disable_existing_loggers: false
formatters:
    simple:
        format: '[%(levelname)s][%(filename)s:%(lineno)d] %(message)s'
handlers:
    console:
        class: logging.StreamHandler
        level: INFO
        formatter: simple
        stream: ext://sys.stdout
loggers:
    smac.intensifier:
        level: ERROR
        handlers: [console]
root:
    level: INFO
    handlers: [console]
```
