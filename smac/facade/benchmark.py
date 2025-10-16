"""
Benchmark SMAC HPOFacade and MFFacade with retrain_after = 1 vs 8.
✅ Compatible with SMAC >= 2.3 (2025 API).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from ConfigSpace import ConfigurationSpace, Float
from hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade, Scenario
from multi_fidelity_facade import MultiFidelityFacade as MFFacade
from smac.intensifier.successive_halving import SuccessiveHalving


# -----------------------------
# 1️⃣ Objective Function (Branin)
# -----------------------------
def branin(cfg, seed=0, budget=None):
    x1, x2 = cfg["x1"], cfg["x2"]
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2) + s * (1 - t) * np.cos(x1) + s


# -----------------------------
# 2️⃣ Configuration Space
# -----------------------------
cs = ConfigurationSpace()
cs.add(Float("x1", (-5, 10)))
cs.add(Float("x2", (0, 15)))


# -----------------------------
# 3️⃣ Scenario
# -----------------------------
def make_scenario(name):
    return Scenario(
        configspace=cs,
        name=name,
        n_trials=40,
        output_directory=f"results/{name}",
        deterministic=True,
    )


# -----------------------------
# 4️⃣ Benchmark Runner
# -----------------------------
def run_benchmark():
    configs = [
        ("HPOFacade", HPOFacade, 1),
        ("HPOFacade", HPOFacade, 8),
        ("MFFacade", MFFacade, 1),
        ("MFFacade", MFFacade, 8),
    ]

    results = {}

    for label, Facade, retrain_after in configs:
        name = f"{label}_retrain{retrain_after}"
        print(f"\n🚀 Running {name} ...")

        scenario = make_scenario(name)

        # ✅ Use SuccessiveHalving (compatible intensifier)
        intensifier = SuccessiveHalving(scenario=scenario)
        intensifier.retrain_after = retrain_after

        start = time.time()
        smac = Facade(
            scenario=scenario,
            target_function=branin,
            intensifier=intensifier,
        )
        incumbent = smac.optimize()
        runtime = time.time() - start

        best_val = branin(incumbent)
        results[name] = (best_val, runtime)
        print(f"✅ {name}: best={best_val:.4f}, time={runtime:.2f}s")

    return results


# -----------------------------
# 5️⃣ Results Summary
# -----------------------------
def summarize(results):
    print("\n📊 Benchmark Summary:")
    print("{:<22} | {:<12} | {:<10}".format("Experiment", "Best Value", "Runtime (s)"))
    print("-" * 55)
    for name, (val, rt) in results.items():
        print(f"{name:<22} | {val:<12.4f} | {rt:<10.2f}")

    labels = list(results.keys())
    runtimes = [rt for _, rt in results.values()]
    values = [val for val, _ in results.values()]

    plt.figure(figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.barh(labels, runtimes)
    plt.xlabel("Runtime (seconds)")
    plt.title("⏱ Runtime Comparison")

    plt.subplot(1, 2, 2)
    plt.barh(labels, values)
    plt.xlabel("Best Objective Value")
    plt.title("🎯 Best Function Value")

    plt.tight_layout()
    plt.savefig("benchmark_summary.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    results = run_benchmark()
    summarize(results)
    print("\n📁 Results saved in 'results/' and 'benchmark_summary.png'")
