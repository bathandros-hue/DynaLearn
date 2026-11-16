import math
import unittest

import numpy as np

from dynalearn import simulate, identify, bootstrap_intervals


class TestDynamics(unittest.TestCase):
    def setUp(self):
        self.mass, self.damping, self.stiffness = 2.0, 1.2, 25.0
        self.data = simulate(self.mass, self.damping, self.stiffness,
                             5, 0.01, lambda t: math.sin(1.8 * t), x0=0.12)
        self.observations = {k: self.data[k] for k in
                             ("displacement", "velocity", "acceleration", "force")}

    def test_exact_recovery(self):
        fit = identify(self.mass, **self.observations)
        self.assertAlmostEqual(fit.damping_ns_m, self.damping, places=8)
        self.assertAlmostEqual(fit.stiffness_n_m, self.stiffness, places=8)
        self.assertGreater(fit.r_squared, 0.999999)

    def test_noisy_recovery_and_bootstrap(self):
        obs = dict(self.observations)
        rng = np.random.default_rng(5)
        obs["acceleration"] = obs["acceleration"] + rng.normal(0, 0.03, len(obs["force"]))
        fit = identify(self.mass, **obs)
        self.assertLess(abs(fit.stiffness_n_m - self.stiffness), 0.3)
        self.assertLess(abs(fit.damping_ns_m - self.damping), 0.2)
        interval = bootstrap_intervals(self.mass, **obs, resamples=100)
        self.assertLess(interval["stiffness_n_m"][0], interval["stiffness_n_m"][1])

    def test_unforced_energy_conservation(self):
        data = simulate(1, 0, 4, 8, 0.01, lambda t: 0, x0=1)
        energy = 0.5 * (data["velocity"] ** 2 + 4 * data["displacement"] ** 2)
        self.assertLess(np.max(abs(energy - energy[0])), 0.0001)

    def test_nonnegative_constraint(self):
        x = np.arange(1, 11, dtype=float)
        v = np.array([2, 4, 3, 8, 5, 12, 7, 16, 9, 20], dtype=float)
        f = np.zeros(len(x))
        a = (f + 3 * v - 2 * x) / 2
        fit = identify(2, x, v, a, f)
        self.assertGreaterEqual(fit.damping_ns_m, 0)
        self.assertGreaterEqual(fit.stiffness_n_m, 0)
        unconstrained = identify(2, x, v, a, f, nonnegative=False)
        self.assertLess(unconstrained.damping_ns_m, 0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            simulate(0, 1, 1, 2, 0.1, lambda t: 0)
        with self.assertRaises(ValueError):
            identify(1, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0])
        with self.assertRaises(ValueError):
            bootstrap_intervals(self.mass, **self.observations, resamples=5)


if __name__ == "__main__":
    unittest.main()
