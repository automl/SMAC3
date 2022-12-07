from benchmark.datasets.iris import IrisDataset
from benchmark.models.svm import SVMModel
from benchmark.tasks.task import Task


tasks = [
    Task(
        name="SVM-Iris",
        model=SVMModel(IrisDataset()),
        objectives=["1-accuracy"],
        n_trials=100,
        optimization_type="hyperparameter_optimization",
    ),
]
