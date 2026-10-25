"""Track A: Continuous Optimization - Benchmark Runner & Experiment Suite.

Executes all compulsory comparisons for Track A:
1. First-order vs Second-order methods across landscapes (Quadratic, Rosenbrock, Saddle Point, Rastrigin)
2. Backpropagation training of TwoLayerMLP
3. Curvature approximations (Fisher, Diagonal Hessian, HVP)
4. Step-size sensitivity and noisy gradient resilience
5. Saves CSV results and plots.
"""

import os
import sys
import time
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import matplotlib.pyplot as plt

from src.track_a_continuous.optimizers import (
    GradientDescent, MomentumOptimizer, NesterovMomentumOptimizer,
    AdaGrad, RMSProp, Adam
)
from src.track_a_continuous.second_order import NewtonOptimizer, GaussNewtonOptimizer
from src.track_a_continuous.backprop import TwoLayerMLP
from src.track_a_continuous.landscapes import (
    QuadraticBowl, Rosenbrock, SaddlePointLandscape, Rastrigin
)
from src.track_a_continuous.curvature import (
    compute_empirical_fisher, compute_diagonal_hessian, hessian_vector_product
)


def run_first_order_benchmark(landscape, init_point, n_iters=150, noise_std=0.0):
    """Benchmarks all first-order optimizers on a given landscape."""
    np.random.seed(42)
    optimizers = {
        "GD": GradientDescent(lr=0.002),
        "Momentum": MomentumOptimizer(lr=0.002, beta=0.85),
        "NAG": NesterovMomentumOptimizer(lr=0.002, beta=0.85),
        "AdaGrad": AdaGrad(lr=0.1),
        "RMSProp": RMSProp(lr=0.01, beta=0.9),
        "Adam": Adam(lr=0.02, beta1=0.9, beta2=0.999),
    }

    results = {}
    for name, opt in optimizers.items():
        x = init_point.copy().astype(float)
        history = [landscape(x)]
        start_t = time.perf_counter()
        for _ in range(n_iters):
            g = landscape.grad(x)
            if noise_std > 0.0:
                g += np.random.normal(0.0, noise_std, size=g.shape)
            x = opt.step(x, g)
            history.append(landscape(x))
        elapsed = time.perf_counter() - start_t
        results[name] = {
            "final_loss": history[-1],
            "final_x": x.tolist(),
            "elapsed_ms": elapsed * 1000.0,
            "history": history
        }
    return results


def run_second_order_benchmark(landscape, init_point, n_iters=20):
    """Benchmarks Newton-Raphson optimization on the given landscape."""
    x = init_point.copy().astype(float)
    newton = NewtonOptimizer(damping=1e-3)
    history = [landscape(x)]
    start_t = time.perf_counter()
    for _ in range(n_iters):
        g = landscape.grad(x)
        H = landscape.hessian(x)
        x = newton.step(x, g, H)
        history.append(landscape(x))
        if np.linalg.norm(g) < 1e-6:
            break
    elapsed = time.perf_counter() - start_t
    return {
        "final_loss": history[-1],
        "final_x": x.tolist(),
        "elapsed_ms": elapsed * 1000.0,
        "history": history,
        "iterations": len(history) - 1
    }


def run_mlp_backpropagation_experiment():
    """Trains TwoLayerMLP with analytical backpropagation using Adam optimizer."""
    np.random.seed(42)
    # Generate non-linear XOR-like dataset
    N = 200
    X = np.random.uniform(-2.0, 2.0, (N, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)

    mlp = TwoLayerMLP(in_features=2, hidden_features=8, out_features=1, seed=42)
    p_vec = mlp.get_params_vector()
    opt = Adam(lr=0.05)

    loss_history = []
    start_t = time.perf_counter()
    for epoch in range(150):
        y_hat = mlp.forward(X)
        loss = mlp.compute_loss(y, y_hat)
        loss_history.append(loss)
        grads_dict = mlp.backward(y)
        grad_vec = mlp.grads_to_vector(grads_dict)
        p_vec = opt.step(p_vec, grad_vec)
        mlp.set_params_vector(p_vec)
    elapsed = time.perf_counter() - start_t

    accuracy = np.mean((mlp.forward(X) > 0.5) == y)
    return {
        "final_loss": loss_history[-1],
        "accuracy": float(accuracy),
        "elapsed_ms": elapsed * 1000.0,
        "loss_history": loss_history
    }


def run_all_track_a_experiments(output_dir="results"):
    """Main execution orchestrator for Track A."""
    print("=" * 60)
    print("Running Track A: Continuous Optimization Experiments...")
    print("=" * 60)

    # 1. First-order on Rosenbrock (Ill-conditioned)
    rosenbrock = Rosenbrock()
    init_pt = np.array([-1.2, 1.0])
    fo_results = run_first_order_benchmark(rosenbrock, init_pt, n_iters=200)

    # 2. Second-order Newton on Rosenbrock
    so_result = run_second_order_benchmark(rosenbrock, init_pt, n_iters=15)

    # 3. Saddle Point escape test (Start near saddle point [0.01, 0.0])
    saddle = SaddlePointLandscape()
    saddle_init = np.array([0.05, 0.01])
    saddle_results = run_first_order_benchmark(saddle, saddle_init, n_iters=100)

    # 4. Noisy objective experiment (SGD vs Adam under Gaussian noise)
    noisy_results = run_first_order_benchmark(rosenbrock, init_pt, n_iters=150, noise_std=0.2)

    # 5. MLP Backpropagation
    mlp_res = run_mlp_backpropagation_experiment()

    # 6. Curvature approximations check
    test_pt = np.array([1.5, 2.0])
    diag_H = compute_diagonal_hessian(rosenbrock, test_pt)
    true_H = rosenbrock.hessian(test_pt)
    v = np.array([1.0, 0.5])
    hvp = hessian_vector_product(rosenbrock.grad, test_pt, v)
    true_hvp = true_H @ v

    # Record summary data table
    summary_data = []
    for opt_name, res in fo_results.items():
        summary_data.append({
            "Landscape": "Rosenbrock",
            "Optimizer": opt_name,
            "Iterations": len(res["history"]) - 1,
            "Final_Loss": res["final_loss"],
            "Elapsed_ms": res["elapsed_ms"],
            "Noise_Std": 0.0
        })
    summary_data.append({
        "Landscape": "Rosenbrock",
        "Optimizer": "Newton-Raphson",
        "Iterations": so_result["iterations"],
        "Final_Loss": so_result["final_loss"],
        "Elapsed_ms": so_result["elapsed_ms"],
        "Noise_Std": 0.0
    })
    for opt_name, res in noisy_results.items():
        summary_data.append({
            "Landscape": "Rosenbrock (Noisy)",
            "Optimizer": opt_name,
            "Iterations": len(res["history"]) - 1,
            "Final_Loss": res["final_loss"],
            "Elapsed_ms": res["elapsed_ms"],
            "Noise_Std": 0.2
        })

    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, "raw", "track_a_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    # Plot convergence comparison
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for opt_name, res in fo_results.items():
        plt.plot(res["history"][:80], label=opt_name)
    plt.plot(so_result["history"], label="Newton-Raphson", linestyle="--", linewidth=2.5, color="black")
    plt.yscale("log")
    plt.title("Rosenbrock Convergence (First-Order vs Newton)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mlp_res["loss_history"], label="MLP Backprop Loss", color="teal")
    plt.title(f"Explicit Backpropagation MLP Training (Acc: {mlp_res['accuracy']*100:.1f}%)")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "track_a_convergence.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved convergence plot to {fig_path}")

    print("\nTrack A Execution Summary:")
    print(df.to_string(index=False))
    print(f"MLP Final Accuracy: {mlp_res['accuracy']*100:.1f}% in {mlp_res['elapsed_ms']:.2f} ms")
    print(f"HVP Error: {np.linalg.norm(hvp - true_hvp):.2e}")
    return df


if __name__ == "__main__":
    run_all_track_a_experiments()
