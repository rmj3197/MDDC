import numpy as np
import pandas as pd

from ._mddc_basic import MDDCBoxplotResult, MDDCMonteCarloResult
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
    coef=1.5,
    chunk_size=None,
    n_jobs=-1,
    seed=None,
):
    """
    Modified Detecting Deviating Cells (MDDC) algorithm for adverse event signal identification.

    This function implements the MDDC algorithm using either a Monte Carlo or Boxplot method for cutoff selection.
    The Monte Carlo or Boxplot method is used to estimate thresholds for identifying cells with high values of standardized Pearson residuals,

    Parameters
    ----------
    contin_table : pd.DataFrame or np.ndarray
        A contingency table of shape (I, J) where rows represent adverse events and columns represent drugs.
        If a DataFrame, it might have index and column names corresponding to the adverse events and drugs.

    method : str, optional, default="monte_carlo"
        Method for cutoff selection. Can be either "monte_carlo" or "boxplot".

    rep : int, optional, default=10000
        Number of Monte Carlo replications used for estimating thresholds. Utilized in Step 2 of the MDDC algorithm.
        Only used if method is "monte_carlo".

    quantile : float, optional, default=0.95
        The quantile of the null distribution obtained via Monte Carlo method to use as a threshold for identify cells with high value of the standardized Pearson residuals.
        Used in Step 2 of the MDDC algorithm. Only used if method is "monte_carlo".

    exclude_same_drug_class : bool, optional, default=True
        If True, excludes other drugs in the same class when constructing 2x2 tables for Fisher's exact test. Only used if method is "monte_carlo".

    col_specific_cutoff : bool, optional, default=True
        Apply Monte Carlo method to the standardized Pearson residuals of the entire table, or within each drug column.
        If True, applies the Monte Carlo/Boxplot method to residuals within each drug column. If False, applies it to the entire table.
        Utilized in Step 2 of the algorithm.

    separate : bool, optional, default=True
        Whether to separate the standardized Pearson residuals for the zero cells and non zero cells and apply Monte Carlo/Boxplot method separately or together.
        If True, separates zero and non-zero cells for cutoff application. If False, applies the cutoff method to all cells together. Utilized in Step 2 of MDDC algorithm.

    if_col_corr : bool, optional, default=False
        Whether to use column (drug) correlation or row (adverse event) correlation
        If True, uses drug correlation instead of adverse event correlation. Utilized in Step 3 of the MDDC algorithm.

    corr_lim : float, optional, default=0.8
        Correlation threshold used to select connected adverse events. Utilized in Step 3 of MDDC algorithm.

    coef : int, float, list, numpy.ndarray, default = 1.5
        Used only when `method` = `boxplot`.
        A numeric value or a list of numeric values. If a single numeric
        value is provided, it will be applied uniformly across all columns of the
        contingency table. If a list is provided, its length must match the number
        of columns in the contingency table, and each value will be used as the
        coefficient for the corresponding column.

    chunk_size : int, optional, default=None
        Useful in scenarios when the dimensions of the contingency table is large as well as the number of Monte Carlo replications. In such scenario the Monte Carlo samples
        need to be generated sequentially such that the memory footprint is manageable (or rather the generated samples fit into the RAM).

    n_jobs : int, optional, default=-1
        n_jobs specifies the maximum number of concurrently
        running workers. If 1 is given, no joblib parallelism
        is used at all, which is useful for debugging. For more
        information on joblib `n_jobs` refer to -
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.

    seed : int or None, optional, default=None
        Random seed for reproducibility.

    Returns
    -------
    result : namedtuple
        - If method is "monte_carlo" returns MDDCMonteCarloResult:
            * pval : numpy.ndarray, pandas.DataFrame
                p-values for each cell in the step 2 of the algorithm, calculated using the Monte Carlo method for cells with count greater than five, and Fisher's exact test for cells with count less than or equal to five.
            * signal : numpy.ndarray, pd.DataFrame
                Matrix indicating significant signals with count greater than five and identified in the step 2 by the Monte Carlo method. 1 indicates a signal, 0 indicates non-signal.
            * fisher_signal : numpy.ndarray, pd.DataFrame
                Matrix indicating signals with a count less than or equal to five and identified by Fisher's exact test. 1 indicates a signal, 0 indicates non-signal.
            * corr_signal_pval : numpy.ndarray, pd.DataFrame
                p-values for each cell in the contingency table in the step 5, when the :math:`r_{ij}` (residual) values are mapped back to the standard normal distribution.
            * corr_signal_adj_pval : numpy.ndarray, pd.DataFrame
                Benjamini-Hochberg adjusted p values for each cell in the step 5.

        - If method is "boxplot" returns MDDCBoxplotResult:
            * signal : numpy.ndarray, pd.DataFrame
                Matrix indicating significant signals with count greater than five and identified in the step 2 by the Monte Carlo method. 1 indicates a signal, 0 indicates non-signal.
            * corr_signal_pval : numpy.ndarray, pd.DataFrame
                p-values for each cell in the contingency table in the step 5, when the :math:`r_{ij}` (residual) values are mapped back to the standard normal distribution.
            * corr_signal_adj_pval : numpy.ndarray, pd.DataFrame
                Benjamini-Hochberg adjusted p values for each cell in the step 5.

    Notes
    ------
    This `chunk_size` option of the function is designed to be used in scenarios where the contingency table dimensions and the number of Monte Carlo replications are large. In such cases,
    the Monte Carlo samples need to be generated sequentially to ensure that the memory footprint remains manageable and the generated samples fit into the available RAM.
    """

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
    if not isinstance(quantile, float | int) or not (0 <= quantile <= 1):
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
    if not isinstance(corr_lim, float | int) or not (0 <= corr_lim <= 1):
        raise TypeError("corr_lim must be a float between 0 and 1.")

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
                chunk_size,
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
            contin_table_mat, col_specific_cutoff, separate, if_col_corr, corr_lim, coef
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

            return MDDCBoxplotResult(
                signal=high_outlier,
                corr_signal_pval=r_pval,
                corr_signal_adj_pval=r_pval_adj,
            )

        else:
            return MDDCBoxplotResult(
                signal=high_outlier,
                corr_signal_pval=r_pval,
                corr_signal_adj_pval=r_pval_adj,
            )
