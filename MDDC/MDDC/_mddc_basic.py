"""
Defines named tuples for organizing results in MDDC (Monte Carlo and Boxplot) analyses.

Named tuples are used to encapsulate the results of different MDDC analyses in a structured way, allowing
easy access to specific result fields by name.

1. `MDDCMonteCarloResult`:
    - Represents the result of a Monte Carlo simulation analysis in MDDC.
    - Fields:
      - `pval`: p-value of the test.
      - `signal`: computed signal value.
      - `fisher_signal`: signal value from Fisher's exact test.
      - `corr_signal_pval`: p-value of the correlation signal.
      - `corr_signal_adj_pval`: adjusted p-value of the correlation signal.

2. `MDDCMBoxplotResult`:
    - Represents the result of a boxplot analysis in MDDC.
    - Fields:
      - `signal`: computed signal value.
      - `corr_signal_pval`: p-value of the correlation signal.
      - `corr_signal_adj_pval`: adjusted p-value of the correlation signal.

These named tuples facilitate the organization and retrieval of analysis results, providing a clear and accessible structure for the data.
"""

from scipy._lib._bunch import _make_tuple_bunch

MDDCMonteCarloResult = _make_tuple_bunch(
    "MDDCMonteCarloResult",
    ["pval", "signal", "fisher_signal", "corr_signal_pval", "corr_signal_adj_pval"],
)

MDDCMBoxplotResult = _make_tuple_bunch(
    "MDDCMBoxplotResult", ["signal", "corr_signal_pval", "corr_signal_adj_pval"]
)
