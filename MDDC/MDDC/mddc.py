import numpy as np
import pandas as pd

from ._mddc_basic import MDDCMBoxplotResult, MDDCMonteCarloResult
from ._mddc_boxplot import _mddc_boxplot
from ._mddc_monte_carlo import _mddc_monte_carlo


def mddc(
    contin_table,
    method="monte_carlo",
    rep=10000,
    quantile=0.95,
    exclude_same_drug_class=True,
    col_specific_cutoff=True,
    separate=True,
    if_col_corr=False,
    corr_lim=0.8,
    r_cutoff=2.326348,
    n_jobs=-1,
    seed=None,
):
    # Check the type of contin_table
    if not (
        isinstance(contin_table, pd.DataFrame) or isinstance(contin_table, np.ndarray)
    ):
        raise TypeError("contin_table must be a pandas DataFrame or a numpy array.")

    # Check the type of method
    if method not in ["boxplot", "monte_carlo"]:
        raise ValueError("method must be either 'boxplot' or 'monte_carlo'.")

    # Check the type of rep
    if not isinstance(rep, int) or rep <= 0:
        raise TypeError("rep must be a positive integer.")

    # Check the type of quantile
    if not isinstance(quantile, (float, int)) or not (0 <= quantile <= 1):
        raise TypeError("quantile must be a float between 0 and 1.")

    # Check the type of exclude_same_drug_class
    if not isinstance(exclude_same_drug_class, bool):
        raise TypeError("exclude_same_drug_class must be a boolean.")

    # Check the type of col_specific_cutoff
    if not isinstance(col_specific_cutoff, bool):
        raise TypeError("col_specific_cutoff must be a boolean.")

    # Check the type of separate
    if not isinstance(separate, bool):
        raise TypeError("separate must be a boolean.")

    # Check the type of if_col_corr
    if not isinstance(if_col_corr, bool):
        raise TypeError("if_col_corr must be a boolean.")

    # Check the type of corr_lim
    if not isinstance(corr_lim, (float, int)) or not (0 <= corr_lim <= 1):
        raise TypeError("corr_lim must be a float between 0 and 1.")

    # Check the type of r_cutoff
    if not isinstance(r_cutoff, (float, int)):
        raise TypeError("r_cutoff must be a float or integer.")

    # Check the type of n_jobs
    if not isinstance(n_jobs, int):
        raise TypeError("n_jobs must be an integer.")

    # Check the type of seed
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be an integer or None.")

    if isinstance(contin_table, pd.DataFrame):
        contin_table_mat = contin_table.values
    else:
        contin_table_mat = contin_table

    if method == "monte_carlo":
        p_val_mat, signal_mat, second_signal_mat, r_pval, r_pval_adj = (
            _mddc_monte_carlo(
                contin_table_mat,
                rep,
                quantile,
                exclude_same_drug_class,
                col_specific_cutoff,
                separate,
                if_col_corr,
                corr_lim,
                r_cutoff,
                n_jobs,
                seed,
            )
        )

        if isinstance(contin_table, pd.DataFrame):
            row_names = list(contin_table.index)
            col_names = list(contin_table.columns)

            p_val_mat = pd.DataFrame(p_val_mat)
            p_val_mat.index = row_names
            p_val_mat.columns = col_names

            signal_mat = pd.DataFrame(signal_mat)
            signal_mat.index = row_names
            signal_mat.columns = col_names

            second_signal_mat = pd.DataFrame(second_signal_mat)
            second_signal_mat.index = row_names
            second_signal_mat.columns = col_names

            r_pval = pd.DataFrame(r_pval)
            r_pval.index = row_names
            r_pval.columns = col_names

            r_pval_adj = pd.DataFrame(r_pval_adj)
            r_pval_adj.index = row_names
            r_pval_adj.columns = col_names

            return MDDCMonteCarloResult(
                pval=p_val_mat,
                signal=signal_mat,
                fisher_signal=second_signal_mat,
                corr_signal_pval=r_pval,
                corr_signal_adj_pval=r_pval_adj,
            )

        else:
            return MDDCMonteCarloResult(
                pval=p_val_mat,
                signal=signal_mat,
                fisher_signal=second_signal_mat,
                corr_signal_pval=r_pval,
                corr_signal_adj_pval=r_pval_adj,
            )

    elif method == "boxplot":
        high_outlier, r_pval, r_pval_adj = _mddc_boxplot(
            contin_table_mat, col_specific_cutoff, separate, if_col_corr, corr_lim
        )

        if isinstance(contin_table, pd.DataFrame):
            row_names = list(contin_table.index)
            col_names = list(contin_table.columns)

            high_outlier = pd.DataFrame(high_outlier)
            high_outlier.index = row_names
            high_outlier.columns = col_names

            r_pval = pd.DataFrame(r_pval)
            r_pval.index = row_names
            r_pval.columns = col_names

            r_pval_adj = pd.DataFrame(r_pval_adj)
            r_pval_adj.index = row_names
            r_pval_adj.columns = col_names

            return MDDCMBoxplotResult(
                signal=high_outlier,
                corr_signal_pval=r_pval,
                corr_signal_adj_pval=r_pval_adj,
            )

        else:
            return MDDCMBoxplotResult(
                signal=high_outlier,
                corr_signal_pval=r_pval,
                corr_signal_adj_pval=r_pval_adj,
            )
