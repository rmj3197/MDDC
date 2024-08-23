import unittest

import pandas as pd

from MDDC.MDDC import find_optimal_coef


class TestFindOptimalCoef(unittest.TestCase):
    def test_optimal_coef_equal_length(self):
        # Example contingency table
        contin_table = pd.DataFrame(
            {"A": [10, 20, 30], "B": [15, 25, 35], "C": [25, 30, 45]}
        )

        # Execute the function
        result1 = find_optimal_coef(contin_table, rep=100, seed=42)
        result2 = find_optimal_coef(
            contin_table, rep=100, exclude_small_count=False, seed=42
        )

        # Check that the number of columns in coef and FDR are equal
        self.assertEqual(
            len(result1.coef),
            len(result1.FDR),
            "The number of columns in coef and FDR should be equal.",
        )
        self.assertEqual(
            len(result1.coef),
            contin_table.shape[1],
            "The number of columns in coef and contin_table should be equal.",
        )

        self.assertEqual(
            len(result2.coef),
            len(result2.FDR),
            "The number of columns in coef and FDR should be equal.",
        )
        self.assertEqual(
            len(result2.coef),
            contin_table.shape[1],
            "The number of columns in coef and contin_table should be equal.",
        )

    def test_optimal_coef_non_empty(self):
        # Example contingency table
        contin_table = pd.DataFrame(
            {"A": [10, 0, 5], "B": [15, 0, 35], "C": [0, 30, 0]}
        )

        # Execute the function
        result = find_optimal_coef(contin_table, rep=100, seed=42)

        # Check that both coef and FDR are non-empty
        self.assertGreater(
            len(result.coef), 0, "Coefficient vector should not be empty."
        )
        self.assertGreater(len(result.FDR), 0, "FDR vector should not be empty.")


if __name__ == "__main__":
    unittest.main()
