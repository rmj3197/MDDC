import unittest

import numpy as np
import pandas as pd

from MDDC.datasets import (
    load_betablocker500_data,
    load_sedative1000_data,
    load_statin49_cluster_idx_data,
    load_statin49_data,
    load_statin101_data,
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

class TestLoadBetaBlocker500Data(unittest.TestCase):
    def test_default_behavior(self):
        # Test default behavior (as_dataframe=True, desc=False)
        data = load_betablocker500_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (501, 9))

    def test_with_description(self):
        # Test with desc=True and as_dataframe=True
        desc, data = load_betablocker500_data(desc=True)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (501, 9))

    def test_as_numpy_array(self):
        # Test with as_dataframe=False
        data = load_betablocker500_data(as_dataframe=False)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (501, 9))

    def test_with_description_as_numpy_array(self):
        # Test with desc=True and as_dataframe=False
        desc, data = load_betablocker500_data(desc=True, as_dataframe=False)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (501, 9))

class TestLoadSedative1000Data(unittest.TestCase):
    def test_default_behavior(self):
        # Test default behavior (as_dataframe=True, desc=False)
        data = load_sedative1000_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (1001, 11))

    def test_with_description(self):
        # Test with desc=True and as_dataframe=True
        desc, data = load_sedative1000_data(desc=True)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (1001, 11))

    def test_as_numpy_array(self):
        # Test with as_dataframe=False
        data = load_sedative1000_data(as_dataframe=False)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (1001, 11))

    def test_with_description_as_numpy_array(self):
        # Test with desc=True and as_dataframe=False
        desc, data = load_sedative1000_data(desc=True, as_dataframe=False)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (1001, 11))

class TestLoadStatin101Data(unittest.TestCase):
    def test_default_behavior(self):
        # Test default behavior (as_dataframe=True, desc=False)
        data = load_statin101_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (102, 5))

    def test_with_description(self):
        # Test with desc=True and as_dataframe=True
        desc, data = load_statin101_data(desc=True)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.shape, (102, 5))

    def test_as_numpy_array(self):
        # Test with as_dataframe=False
        data = load_statin101_data(as_dataframe=False)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (102, 5))

    def test_with_description_as_numpy_array(self):
        # Test with desc=True and as_dataframe=False
        desc, data = load_statin101_data(desc=True, as_dataframe=False)
        self.assertIsInstance(desc, str)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual(data.shape, (102, 5))