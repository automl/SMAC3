

from benchmark.wrappers.wrapper import Wrapper


class Version140(Wrapper):
    def __init__(self, task) -> None:
        self._task = task
    
    def run(self) -> None:
        # Create facade
        # ...
