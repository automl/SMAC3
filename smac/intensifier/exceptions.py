class SearchSpaceShrinkageWarning(Warning):
    """Warning raised if shrinking the Searchspace using SearchSpaceModifier.MultiFidelitySearchSpaceShrinker, results in a more 
        uniform distribution. Generally shrinking the search space should result in a more spiked distribution."""
    pass