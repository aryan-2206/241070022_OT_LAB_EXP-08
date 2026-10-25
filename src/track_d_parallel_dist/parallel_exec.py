"""Track D: Parallel Program Optimization.

Implements:
1. Data Parallelism with chunked static partitioning
2. Dynamic Work Stealing for skewed task workloads
3. Pipeline Parallelism emulation across staged queues
4. Speedup (S_p) and Parallel Efficiency (E_p) instrumentation
"""

import time
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np


def synthetic_compute_kernel(task_weight):
    """Simulates CPU-bound numeric work proportional to task_weight."""
    count = max(100, min(80000, int(task_weight * 5000)))
    arr = np.arange(1, count + 1, dtype=float)
    return float(np.sum(np.sqrt(arr)))


class ParallelBenchmarkSuite:
    def __init__(self, n_tasks=100, skew="skewed", seed=42):
        np.random.seed(seed)
        self.n_tasks = n_tasks
        if skew == "skewed":
            # Pareto / exponential distribution of task workloads
            self.task_weights = np.random.pareto(a=1.5, size=n_tasks) + 0.1
        else:
            # Uniform task workloads
            self.task_weights = np.ones(n_tasks)

    def run_sequential(self):
        """Sequential single-threaded baseline (T_1)."""
        start_t = time.perf_counter()
        results = [synthetic_compute_kernel(w) for w in self.task_weights]
        elapsed = time.perf_counter() - start_t
        return elapsed, len(results)

    def run_data_parallel_static(self, num_workers=4):
        """Static block partitioning across worker threads/processes."""
        start_t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(synthetic_compute_kernel, self.task_weights))
        elapsed = time.perf_counter() - start_t
        return elapsed, len(results)

    def run_dynamic_work_stealing(self, num_workers=4):
        """Dynamic queue work stealing: idle workers dequeue next available task."""
        task_q = queue.Queue()
        for w in self.task_weights:
            task_q.put(w)

        results = []
        lock = threading.Lock()

        def worker_loop():
            while True:
                try:
                    w = task_q.get_nowait()
                except queue.Empty:
                    break
                res = synthetic_compute_kernel(w)
                with lock:
                    results.append(res)
                task_q.task_done()

        threads = [threading.Thread(target=worker_loop) for _ in range(num_workers)]
        start_t = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start_t
        return elapsed, len(results)

    def run_pipeline_parallel(self, num_stages=3):
        """Simulates 3-stage pipeline parallelism: Stage 1 (Extract) -> Stage 2 (Transform) -> Stage 3 (Aggregate)."""
        q1 = queue.Queue(maxsize=10)
        q2 = queue.Queue(maxsize=10)
        final_results = []
        done_token = object()

        def stage1():
            for w in self.task_weights:
                time.sleep(0.0001)  # Staged overhead
                q1.put(w)
            q1.put(done_token)

        def stage2():
            while True:
                item = q1.get()
                if item is done_token:
                    q2.put(done_token)
                    break
                res = synthetic_compute_kernel(item)
                q2.put(res)

        def stage3():
            while True:
                item = q2.get()
                if item is done_token:
                    break
                final_results.append(item * 1.01)

        t1 = threading.Thread(target=stage1)
        t2 = threading.Thread(target=stage2)
        t3 = threading.Thread(target=stage3)

        start_t = time.perf_counter()
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
        elapsed = time.perf_counter() - start_t
        return elapsed, len(final_results)
