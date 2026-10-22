import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from flash.detection import detect, validate_change


class ValidateChangeTest(unittest.TestCase):
    def test_persistent_latency_increase_is_reported(self):
        values = [100.0] * 10 + [130.0] * 20

        result = validate_change(values, pos=10, probability=0.99)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["degradation"], 0.30)
        self.assertAlmostEqual(result["recovery_ratio"], 0.0)
        self.assertEqual(result["baseline_ms"], 100.0)
        self.assertEqual(result["degraded_ms"], 130.0)

    def test_transient_spike_that_returns_to_baseline_is_filtered(self):
        values = [100.0] * 10 + [1000.0] + [100.0] * 9

        result = validate_change(values, pos=10, probability=0.99)

        self.assertIsNone(result)

    def test_recovery_ratio_at_threshold_is_filtered(self):
        # Baseline=100, degraded mean=300, and post-change recent mean=150,
        # so recovery_ratio=(300-150)/(300-100)=0.75 exactly.
        values = [100.0] * 10 + [3150.0] + [150.0] * 19

        result = validate_change(values, pos=10, probability=0.99)

        self.assertIsNone(result)

    def test_five_percent_increase_is_below_twenty_percent_threshold(self):
        values = [100.0] * 10 + [105.0] * 20

        result = validate_change(values, pos=10, probability=0.99)

        self.assertIsNone(result)

    def test_rightmost_change_point_is_rejected_by_guard(self):
        values = [100.0] * 10 + [130.0] * 20

        result = validate_change(values, pos=len(values) - 1, probability=0.99)

        self.assertIsNone(result)

    def test_low_probability_is_rejected(self):
        values = [100.0] * 10 + [130.0] * 20

        result = validate_change(values, pos=10, probability=0.79)

        self.assertIsNone(result)

    def test_probability_at_configured_threshold_is_accepted(self):
        self.assertIsNotNone(validate_change([100.0]*10+[150.0]*20, 10, 0.8))


class DetectNoFaultTest(unittest.TestCase):
    def test_skips_exactly_three_zero_based_warmup_iterations(self):
        observed=[]
        def beast(values, **kwargs):
            observed.extend(values.tolist())
            return types.SimpleNamespace(trend=types.SimpleNamespace(cp=[], cpPr=[]))
        history=[{'iteration':i,'time_ms':1000.0+i} for i in range(33)]
        with patch.dict(sys.modules, {'Rbeast':types.SimpleNamespace(beast=beast)}):
            self.assertIsNone(detect(history))
        self.assertEqual(observed, [1000.0+i for i in range(3,33)])

    def test_constant_history_is_not_reported_as_a_fault(self):
        fake_rbeast = types.SimpleNamespace(
            beast=lambda *args, **kwargs: types.SimpleNamespace(
                trend=types.SimpleNamespace(cp=np.array([15.0]), cpPr=np.array([0.99]))
            )
        )
        history = [
            {"iteration": iteration, "time_ms": 100.0}
            for iteration in range(4, 34)
        ]

        with patch.dict(sys.modules, {"Rbeast": fake_rbeast}):
            result = detect(history)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
