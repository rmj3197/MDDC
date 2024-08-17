import unittest

import numpy as np
import pandas as pd

from MDDC.datasets import (
    load_anaphylaxis_data,
    load_statin49_cluster_idx_data,
    load_statin49_data,
)


class TestLoadStatin49Data(unittest.TestCase):
    def test_default_behavior(self):
        # Test default behavior (as_dataframe=True, desc=False)
        data = load_statin49_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (50, 7))

    def test_with_description(self):
        # Test with desc=True and as_dataframe=True
        desc, data = load_statin49_data(desc=True)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (50, 7))

    def test_as_numpy_array(self):
        # Test with as_dataframe=False
        data = load_statin49_data(as_dataframe=False)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (50, 7))

    def test_with_description_as_numpy_array(self):
        # Test with desc=True and as_dataframe=False
        desc, data = load_statin49_data(desc=True, as_dataframe=False)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (50, 7))


class TestLoadStatin49ClusterIdxData(unittest.TestCase):
    def test_default_behavior(self):
        # Test default behavior (as_dataframe=True, desc=False)
        data = load_statin49_cluster_idx_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (50, 8))

    def test_with_description(self):
        # Test with desc=True and as_dataframe=True
        desc, data = load_statin49_cluster_idx_data(desc=True)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (50, 8))

    def test_as_numpy_array(self):
        # Test with as_dataframe=False
        data = load_statin49_cluster_idx_data(as_dataframe=False)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (50, 8))

    def test_with_description_as_numpy_array(self):
        # Test with desc=True and as_dataframe=False
        desc, data = load_statin49_cluster_idx_data(desc=True, as_dataframe=False)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (50, 8))


class TestLoadAnaphylaxisData(unittest.TestCase):
    def test_default_behavior(self):
        # Test default behavior (as_dataframe=True, desc=False)
        data = load_anaphylaxis_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (131, 4))

    def test_with_description(self):
        # Test with desc=True and as_dataframe=True
        desc, data = load_anaphylaxis_data(desc=True)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (131, 4))

    def test_as_numpy_array(self):
        # Test with as_dataframe=False
        data = load_anaphylaxis_data(as_dataframe=False)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (131, 4))

    def test_with_description_as_numpy_array(self):
        # Test with desc=True and as_dataframe=False
        desc, data = load_anaphylaxis_data(desc=True, as_dataframe=False)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (131, 4))
