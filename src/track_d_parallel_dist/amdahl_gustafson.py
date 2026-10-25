"""Track D: Amdahl's Law and Gustafson's Law Modeling.

Calculates:
1. Amdahl's Law (Strong Scaling):
   S_p = 1 / ((1 - p_parallel) + p_parallel / p) = 1 / (s + (1 - s) / p)
2. Gustafson's Law (Weak Scaling):
   S_p = p - s * (p - 1) = s + p * (1 - s)
"""

import numpy as np


class ScalingLaws:
    @staticmethod
    def amdahl_speedup(p, serial_fraction):
        """Theoretical speedup according to Amdahl's Law for fixed problem size."""
        s = serial_fraction
        p_par = 1.0 - s
        return 1.0 / (s + p_par / p)

    @staticmethod
    def gustafson_speedup(p, serial_fraction):
        """Theoretical scaled speedup according to Gustafson's Law for scaled problem size."""
        s = serial_fraction
        return p - s * (p - 1.0)

    @staticmethod
    def estimate_serial_fraction(p_values, measured_speedups):
        """Fits empirical speedups to Amdahl's formula using non-linear least squares."""
        from scipy.optimize import curve_fit

        def fit_fn(p, s):
            return 1.0 / (s + (1.0 - s) / p)

        popt, _ = curve_fit(fit_fn, p_values, measured_speedups, p0=[0.1], bounds=(0.0, 1.0))
        return float(popt[0])
