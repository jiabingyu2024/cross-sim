"""Circuit-level regressions for the SONOS separate-source parasitic solver.

Run without optional pytest dependencies:
    uv run --locked python -m unittest discover -s tests -p test_separate_source_parasitics.py -v
"""
import unittest
import warnings
import os

import numpy as np

from simulator import CrossSimParameters
from simulator.backend import ComputeBackend
from simulator.circuits.array import NoninterleavedSeparateSourceArray


class SeparateSourceParasiticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ComputeBackend(use_gpu=os.environ.get('CROSSSIM_TEST_GPU') == '1').backend

    def host(self, value):
        return value.get() if hasattr(value, 'get') else value

    def test_high_bl_resistance_has_finite_kirchhoff_solution(self):
        """A passive BL has a solution even when fixed-point retries overflow."""
        params = CrossSimParameters()
        params.core.rows_max = 1152
        params.xbar.device.Rmin = 62500
        params.xbar.array.parasitics.Rp_col = 100
        params.xbar.array.parasitics.current_from_input = False
        params.xbar.array.parasitics.selected_rows = 'top'
        params.simulation.relaxation_gamma = .9
        params.simulation.Niters_max_parasitics = 100
        params.simulation.Verr_th_mvm = 2e-4
        params.simulation.hide_convergence_msg = True
        solver = NoninterleavedSeparateSourceArray(params)
        count = 144
        conductance = np.ones((1, count), dtype=np.float32)
        vector = np.ones(count, dtype=np.float32)
        vector[3] = 0  # A gated-off SONOS cell, as in bit-serial inference.

        # Independent dense Kirchhoff reference, including unused-row wiring.
        segments = np.full(count, 100 / 62500)
        segments[-1] *= 1152 - count + 1
        tails = np.cumsum(segments[::-1])[::-1]
        impedance = tails[np.maximum.outer(np.arange(count), np.arange(count))]
        active = (vector != 0).astype(float)
        currents = np.linalg.solve(np.eye(count) + active[:, None] * impedance, vector)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            actual = solver.iterative_solve(self.backend.asarray(conductance), self.backend.asarray(vector))
        actual = self.host(actual)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, [currents.sum()], rtol=2e-4, atol=1e-6)

    def test_batched_signed_and_gated_inputs_match_dense_circuit(self):
        rng = np.random.default_rng(37)
        for rp in [.01, 10, 100, 1000]:
            for placement in ['top', 'bottom']:
                with self.subTest(rp=rp, placement=placement):
                    p = CrossSimParameters()
                    p.core.rows_max = 1152
                    p.xbar.device.Rmin = 62500
                    p.xbar.array.parasitics.Rp_col = rp
                    p.xbar.array.parasitics.Rp_col_terminal = .2
                    p.xbar.array.parasitics.selected_rows = placement
                    p.simulation.Verr_th_mvm = 2e-4
                    p.simulation.hide_convergence_msg = True
                    n = 12
                    g = rng.uniform(0, 1, (2, n)).astype(np.float32)
                    v = rng.integers(-1, 2, (2, n, 3)).astype(np.float32)
                    expected = np.empty((2, 2, 3))  # batch, output, window
                    segments = np.full(n, rp / 62500)
                    if placement == 'top':
                        segments[-1] *= 1152 - n + 1
                    segments[-1] += .2 / 62500
                    tails = np.cumsum(segments[::-1])[::-1]
                    impedance = tails[np.maximum.outer(np.arange(n), np.arange(n))]
                    for batch in range(2):
                        for out in range(2):
                            for win in range(3):
                                active_g = g[out] * (v[batch, :, win] != 0)
                                a = np.eye(n) + active_g[:, None] * impedance
                                currents = np.linalg.solve(a, active_g * v[batch, :, win])
                                expected[batch, out, win] = currents.sum()
                    solver = NoninterleavedSeparateSourceArray(p)
                    actual = self.host(solver.iterative_solve(self.backend.asarray(g), self.backend.asarray(v)))
                    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=1e-5)

    def test_zero_gates_do_not_mutate_programmed_conductance(self):
        p = CrossSimParameters()
        p.xbar.array.parasitics.Rp_col = .1
        solver = NoninterleavedSeparateSourceArray(p)
        g = self.backend.ones((1, 4), dtype=np.float32)
        original = g.copy()
        solver.iterative_solve(g, self.backend.asarray([1, 0, 1, 0], dtype=np.float32))
        np.testing.assert_array_equal(self.host(g), self.host(original))

    def test_nonfinite_current_is_never_accepted(self):
        solver = NoninterleavedSeparateSourceArray(CrossSimParameters())
        for value in [np.inf, np.nan]:
            with self.assertRaises(RuntimeError):
                solver._post_process(self.backend.asarray([value]), input_dim=1)
            with self.assertRaises(RuntimeError):
                solver._error_metric(self.backend.asarray([[value]]))


if __name__ == '__main__':
    unittest.main()
