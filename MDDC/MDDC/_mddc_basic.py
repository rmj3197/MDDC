"""
Defines named tuples for organizing results in MDDC (Monte Carlo and Boxplot) analyses.

Named tuples are used to encapsulate the results of different MDDC analyses in a structured way, allowing
easy access to specific result fields by name.

1. `MDDCMonteCarloResult`:
    - Represents the result of a Monte Carlo simulation analysis in MDDC.
    - Fields:
      - `pval`: unadjusted p-value from Step 2 of MDDC MC.
      - `pval_adj`: BH adjusted p-value of p-values obtained in Step 2 of MDDC MC.
      - `signal`: computed signal value when count > 5 using unadjusted p-values.
      - `fisher_signal`: computed signal value from Fisher's exact test using unadjusted p-values.
      - `signal_adj`: computed signal value when count > 5 using adjusted p-values.
      - `fisher_signal_adj`: computed signal value from Fisher's exact test using adjusted p-values.
      - `corr_signal_pval`: p-value of the correlation signal.
      - `corr_signal_adj_pval`: adjusted p-value of the correlation signal.

2. `MDDCBoxplotResult`:
    - Represents the result of a boxplot analysis in MDDC.
    - Fields:
      - `signal`: computed signal value from Step 2 of MDDC Boxplot.
      - `corr_signal_pval`: p-value of the correlation signal.
      - `corr_signal_adj_pval`: adjusted p-value of the correlation signal.

These named tuples facilitate the organization and retrieval of analysis results, providing a clear and accessible structure for the data.
"""

from scipy._lib._bunch import _make_tuple_bunch

MDDCMonteCarloResult = _make_tuple_bunch(
    "MDDCMonteCarloResult",
    [
        "pval",
        "pval_adj",
        "signal",
        "fisher_signal",
        "signal_adj",
        "fisher_signal_adj",
        "corr_signal_pval",
        "corr_signal_adj_pval",
    ],
)

MDDCBoxplotResult = _make_tuple_bunch(
    "MDDCBoxplotResult", ["signal", "corr_signal_pval", "corr_signal_adj_pval"]
)
