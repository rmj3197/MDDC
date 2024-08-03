from scipy._lib._bunch import _make_tuple_bunch

MDDCMonteCarloResult = _make_tuple_bunch(
    "MDDCMonteCarloResult",
    ["pval", "signal", "fisher_signal", "corr_signal_pval", "corr_signal_adj_pval"],
)

MDDCMBoxplotResult = _make_tuple_bunch(
    "MDDCMBoxplotResult", ["signal", "corr_signal_pval", "corr_signal_adj_pval"]
)
