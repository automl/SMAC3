from ConfigSpace import ConfigurationSpace

cs = ConfigurationSpace(
    {
        "a": (0.1, 1.5),
        "b": (2, 10),
        "c": ["cat", "dog", "mouse"],
    }
)

config = cs.sample_configuration(1)
print(config)

config_dict = config.get_dictionary()
config_dict["a"] = 10000

print(config)
