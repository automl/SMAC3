class NonNumericHyperparameterError(Exception):
    """Exception indicating that a non-numeric hyperparameter was used in a context where only numeric hyperparameters are allowed."""
    pass