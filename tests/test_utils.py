import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MDDC.utils import (
    generate_contin_table_with_clustered_AE,
    plot_heatmap,
    report_drug_AE_pairs,
)


class TestGenerateContinTableWithClusteredAE(unittest.TestCase):
    def setUp(self):
        self.contin_table_df = pd.DataFrame(
            [[5, 10], [15, 20]], index=["AE1", "AE2"], columns=["Drug1", "Drug2"]
        )
        self.contin_table_np = np.array([[5, 10], [15, 20]])
        self.signal_mat = np.array([[1, 1.5], [1.2, 1]])
        self.cluster_idx_list = [0, 1]
        self.cluster_idx_np = np.array([0, 1])

    def test_valid_input_dataframe(self):
        result = generate_contin_table_with_clustered_AE(
            self.contin_table_df,
            self.signal_mat,
            self.cluster_idx_list,
            n=5,
            rho=0.5,
        )
        self.assertEqual(len(result), 5)
        self.assertTrue(all(isinstance(table, pd.DataFrame) for table in result))

    def test_valid_input_numpy(self):
        result = generate_contin_table_with_clustered_AE(
            self.contin_table_np,
            self.signal_mat,
            self.cluster_idx_np,
            n=5,
            rho=0.5,
            seed=42,
        )
        self.assertEqual(len(result), 5)
        self.assertTrue(all(isinstance(table, np.ndarray) for table in result))

    def test_empty_contin_table(self):
        with self.assertRaises(ValueError):
            generate_contin_table_with_clustered_AE(
                np.array([]),
                self.signal_mat,
                self.cluster_idx_list,
                n=5,
                rho=0.5,
            )

    def test_empty_cluster_idx(self):
        with self.assertRaises(ValueError):
            generate_contin_table_with_clustered_AE(
                self.contin_table_np, self.signal_mat, [], n=5, rho=0.5
            )

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            generate_contin_table_with_clustered_AE(
                self.contin_table_np, self.signal_mat, [0], n=5, rho=0.5
            )

    def test_invalid_rho(self):
        for invalid_rho in [-0.5, 1.5]:
            with self.assertRaises(ValueError):
                generate_contin_table_with_clustered_AE(
                    self.contin_table_np,
                    self.signal_mat,
                    self.cluster_idx_np,
                    n=5,
                    rho=invalid_rho,
                )

    def test_invalid_options(self):
        for invalid_input in ["string"]:
            with self.assertRaises(TypeError):
                generate_contin_table_with_clustered_AE(
                    invalid_input,
                    self.signal_mat,
                    self.cluster_idx_np,
                    n=5,
                    rho=0.5,
                )
            with self.assertRaises(TypeError):
                generate_contin_table_with_clustered_AE(
                    self.contin_table_df,
                    self.signal_mat,
                    "cluster_idx_np",
                    n=5,
                    rho=0.5,
                )

class TestReportDrugAEPairs(unittest.TestCase):
    def setUp(self):
        self.contin_table = np.array([[5, 1, 0], [3, 0, 2], [0, 4, 6]])
        self.contin_table_signal = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 1]])
        self.contin_table_no_signal = np.zeros((3, 3))

        self.contin_table_df = pd.DataFrame(
            self.contin_table,
            index=["AE1", "AE2", "AE3"],
            columns=["Drug1", "Drug2", "Drug3"],
        )
        self.contin_table_signal_df = pd.DataFrame(
            self.contin_table_signal,
            index=["AE1", "AE2", "AE3"],
            columns=["Drug1", "Drug2", "Drug3"],
        )
        self.contin_table_no_signal_df = pd.DataFrame(
            self.contin_table_no_signal,
            index=["AE1", "AE2", "AE3"],
            columns=["Drug1", "Drug2", "Drug3"],
        )

    def test_report_drug_AE_pairs_with_numpy_arrays(self):
        result = report_drug_AE_pairs(self.contin_table, self.contin_table_signal)
        expected_columns = [
            "Drug",
            "AE",
            "Observed Count",
            "Expected Count",
            "Standard Pearson Residual",
        ]
        self.assertEqual(result.columns.tolist(), expected_columns)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.iloc[0]["Drug"], 0)  # First column index
        self.assertEqual(result.iloc[0]["AE"], 0)  # First row index
        self.assertEqual(result.iloc[0]["Observed Count"], 5)

    def test_report_drug_AE_pairs_with_pandas_dataframes(self):
        result = report_drug_AE_pairs(self.contin_table_df, self.contin_table_signal_df)
        expected_columns = [
            "Drug",
            "AE",
            "Observed Count",
            "Expected Count",
            "Standard Pearson Residual",
        ]
        self.assertEqual(result.columns.tolist(), expected_columns)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.iloc[0]["Drug"], "Drug1")
        self.assertEqual(result.iloc[0]["AE"], "AE1")
        self.assertEqual(result.iloc[0]["Observed Count"], 5)

    def test_report_drug_AE_pairs_with_no_signal(self):
        result = report_drug_AE_pairs(
            self.contin_table_df, self.contin_table_no_signal_df
        )
        expected_columns = [
            "Drug",
            "AE",
            "Observed Count",
            "Expected Count",
            "Standard Pearson Residual",
        ]
        self.assertEqual(result.columns.tolist(), expected_columns)
        self.assertEqual(len(result), 0)

    def test_dimension_mismatch(self):
        contin_table_signal_wrong = np.array([[1, 0], [1, 0], [0, 1]])
        with self.assertRaises(ValueError) as context:
            report_drug_AE_pairs(self.contin_table, contin_table_signal_wrong)
        self.assertIn(
            "dimensions of contin_table and contin_table_signal must be the same",
            str(context.exception),
        )

    def test_row_names_mismatch(self):
        contin_table_signal_wrong_df = self.contin_table_signal_df.copy()
        contin_table_signal_wrong_df.index = ["AE1", "AE2", "AE4"]
        with self.assertRaises(ValueError) as context:
            report_drug_AE_pairs(self.contin_table_df, contin_table_signal_wrong_df)
        self.assertIn(
            "row names of contin_table and contin_table_signal must match",
            str(context.exception),
        )

    def test_column_names_mismatch(self):
        contin_table_signal_wrong_df = self.contin_table_signal_df.copy()
        contin_table_signal_wrong_df.columns = ["Drug1", "Drug2", "Drug4"]
        with self.assertRaises(ValueError) as context:
            report_drug_AE_pairs(self.contin_table_df, contin_table_signal_wrong_df)
        self.assertIn(
            "column names of contin_table and contin_table_signal must match",
            str(context.exception),
        )

    def test_options(self):
        contin_table = "str"
        with self.assertRaises(TypeError):
            report_drug_AE_pairs(contin_table, self.contin_table_signal)
        with self.assertRaises(TypeError):
            report_drug_AE_pairs(
                self.contin_table_df, self.contin_table_signal_df, along_rows=1234
            )

class TestPlotHeatmap(unittest.TestCase):
    def test_plot_heatmap_with_numpy_array(self):
        data = np.random.rand(10, 10)
        fig = plot_heatmap(data)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_heatmap_with_pandas_dataframe(self):
        data = pd.DataFrame(
            np.random.rand(10, 10), columns=[f"col_{i}" for i in range(10)]
        )
        fig = plot_heatmap(data)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_heatmap_with_invalid_data_type(self):
        data = [1, 2, 3, 4, 5]
        with self.assertRaises(TypeError):
            plot_heatmap(data)

    def test_plot_heatmap_with_empty_data(self):
        data = pd.DataFrame()
        fig = plot_heatmap(data)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_heatmap_with_custom_size_cell(self):
        data = np.random.rand(10, 10)
        fig = plot_heatmap(data, size_cell=0.5)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_heatmap_with_additional_kwargs(self):
        data = np.random.rand(10, 10)
        fig = plot_heatmap(data, cmap="viridis")
        self.assertIsInstance(fig, plt.Figure)
