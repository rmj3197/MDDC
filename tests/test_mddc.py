import unittest

import pandas as pd

from MDDC.MDDC import mddc
from MDDC.MDDC._mddc_basic import MDDCBoxplotResult, MDDCMonteCarloResult


class MDDCTestCase(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame(
            {"Drug1": [10, 5, 2, 0], "Drug2": [3, 0, 1, 0], "Drug3": [0, 7, 0, 4]},
            index=["AE1", "AE2", "AE3", "AE4"],
        )

    def check_mddc_result(self, result, result_type):
        self.assertIsInstance(result, result_type)
        expected_attrs = ["signal", "corr_signal_pval", "corr_signal_adj_pval"]
        if result_type is MDDCMonteCarloResult:
            expected_attrs.extend(
                ["pval", "fisher_signal", "corr_signal_pval", "corr_signal_adj_pval"]
            )
        for attr in expected_attrs:
            self.assertTrue(hasattr(result, attr))

    def test_mmdc_options(self):
        invalid_options = {
            "col_specific_cutoff": "some",
            "separate": "some",
            "if_col_corr": "some",
            "seed": "some",
        }
        for option, value in invalid_options.items():
            with self.assertRaises(TypeError):
                mddc(self.data, **{option: value})

    def test_mddc_with_valid_data_and_methods(self):
        methods = [
            {
                "method": "monte_carlo",
                "params": {"rep": 10000, "quantile": 0.95, "seed": 42},
            },
            {
                "method": "monte_carlo",
                "params": {
                    "col_specific_cutoff": False,
                    "rep": 10000,
                    "quantile": 0.95,
                    "seed": 42,
                    "chunk_size": 1,
                },
            },
            {
                "method": "monte_carlo",
                "params": {
                    "separate": False,
                    "col_specific_cutoff": False,
                    "rep": 10000,
                    "quantile": 0.95,
                    "seed": 42,
                },
            },
            {
                "method": "monte_carlo",
                "params": {
                    "if_col_corr": True,
                    "separate": False,
                    "col_specific_cutoff": False,
                    "rep": 10000,
                    "quantile": 0.95,
                    "seed": 42,
                },
            },
            {
                "method": "monte_carlo",
                "params": {
                    "if_col_corr": True,
                    "separate": False,
                    "col_specific_cutoff": False,
                    "rep": 10000,
                    "quantile": 0.95,
                    "seed": 42,
                },
                "data": self.data.values,
            },
            {"method": "boxplot", "params": {"col_specific_cutoff": False, "seed": 42}},
            {
                "method": "boxplot",
                "params": {"col_specific_cutoff": False, "separate": False, "seed": 42},
            },
            {
                "method": "boxplot",
                "params": {
                    "col_specific_cutoff": False,
                    "separate": False,
                    "if_col_corr": True,
                    "seed": 42,
                },
            },
            {
                "method": "boxplot",
                "params": {"col_specific_cutoff": False, "seed": 42},
                "data": self.data.values,
            },
            {
                "method": "boxplot",
                "params": {"col_specific_cutoff": True, "separate": False, "seed": 42},
            },
            {
                "method": "boxplot",
                "params": {"col_specific_cutoff": True, "separate": True, "seed": 42},
            },
        ]
        for method in methods:
            result = mddc(
                self.data if "data" not in method else method["data"],
                method=method["method"],
                **method["params"],
            )
            result_type = (
                MDDCMonteCarloResult
                if method["method"] == "monte_carlo"
                else MDDCBoxplotResult
            )
            self.check_mddc_result(result, result_type)

    def test_mddc_with_invalid_parameters(self):
        invalid_params = {
            "data": ([1, 2, 3, 4], "monte_carlo"),
            "method": "invalid_method",
            "rep": -100,
            "quantile": 1.5,
            "exclude_same_drug_class": "not_a_boolean",
            "n_jobs": "not_an_integer",
            "corr_lim": 1.5,
        }
        for param, value in invalid_params.items():
            with self.assertRaises(TypeError if param != "method" else ValueError):
                if param == "method":
                    mddc(self.data, method=value)
                else:
                    mddc(self.data, method="monte_carlo", **{param: value})


if __name__ == "__main__":
    unittest.main()
