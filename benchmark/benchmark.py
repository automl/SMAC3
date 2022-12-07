from benchmark.datasets.dataset import Dataset
from benchmark.models.model import Model
from wrappers.v140 import Version140
from wrappers.v200a3 import Version200a3
from tasks import tasks
import socket


WRAPPERS = [Version140, Version200a3]


class Benchmark:
    """Selects the right wrapper (based on the environment), runs, and saves the benchmark."""

    def __init__(self) -> None:
        import smac

        try:
            version = smac.__version__
        except:
            pass

        try:
            version = smac.version
        except:
            pass

        if version is None:
            raise RuntimeError("Could not find version of SMAC.")

        for wrapper in WRAPPERS:
            if version == wrapper.version:
                self._wrapper = wrapper
                break

        if self._wrapper is None:
            raise RuntimeError(f"Could not find a wrapper for version {version}.")

        self._version = version

    def run(self) -> None:
        # Get name of the current computer (for comparison purposes)
        computer = socket.gethostname()

        for task in tasks:
            task_wrapper = self._wrapper(task)
            task_wrapper.run()

            # Now we collect some metrics
            # ...

        # Now we merge other versions to a combined report
        # ...
