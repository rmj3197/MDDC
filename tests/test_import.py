"""
Checks if MDDC can be imported correctly along with submodules
"""

import unittest

import MDDC


class TestMDDCModule(unittest.TestCase):
    def test_version(self):
        from MDDC import __version__

        self.assertTrue(isinstance(__version__, str))

    def test_dir(self):
        import MDDC

        self.assertIsNotNone(dir(MDDC))

    def test_submodules_exist(self):
        from MDDC import submodules

        for submodule in submodules:
            self.assertIsNotNone(getattr(MDDC, submodule, None))
