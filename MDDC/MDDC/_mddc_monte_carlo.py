import numpy as np
import scipy
from joblib import Parallel, delayed
from matplotlib.cbook import boxplot_stats
from mddc_cpp_helper import getPVal, getZijMat, pearsonCorWithNA

from ._helper import (
    compute_fisher_exact,
    compute_whislo1,
    compute_whislo2,
    get_log_bootstrap_cutoff,
    get_log_bootstrap_cutoff_sequential,
    normalize_column,
    process_index,
)


def _mddc_monte_carlo(
    contin_table,
    rep=10000,
    quantile=0.95,
    exclude_same_drug_class=True,
    col_specific_cutoff=True,
    separate=True,
    if_col_corr=False,
    corr_lim=0.8,
    chunk_size=None,
    n_jobs=-1,
    seed=None,
):
    """
    Modified Detecting Deviating Cells (MDDC) algorithm for adverse event signal identification using Monte Carlo method for cutoff selection.

    This function implements the MDDC algorithm using the Monte Carlo method to determine cutoffs for identifying cells with high standardized Pearson residuals.

    For details on the algorithm please see :ref:`MDDC Algorithm <mddc_algorithm>`.

    Parameters:
    -----------
    contin_table : pd.DataFrame or np.ndarray
        A contingency table of shape (I, J) where rows represent adverse events and columns represent drugs.
        If a DataFrame, it might have index and column names corresponding to the adverse events and drugs.

    rep : int, optional, default=10000
        Number of Monte Carlo replications used for estimating thresholds. Utilized in Step 2 of the MDDC algorithm.

    quantile : float, optional, default=0.95
        The quantile of the null distribution obtained via Monte Carlo method to use as a threshold for identify cells with high value of the standardized Pearson residuals.
        Used in Step 2 of the MDDC algorithm.

    exclude_same_drug_class : bool, optional, default=True
        If True, excludes other drugs in the same class when constructing 2x2 tables for Fisher's exact test.

    col_specific_cutoff : bool, optional, default=True
        Apply Monte Carlo method to the standardized Pearson residuals of the entire table, or within each drug column.
        If True, applies the Monte Carlo method to residuals within each drug column. If False, applies it to the entire table.
        Utilized in Step 2 of the algorithm.

    separate : bool, optional, default=True
        Whether to separate the standardized Pearson residuals for the zero cells and non zero cells and apply MC method separately or together.
        If True, separates zero and non-zero cells for cutoff application. If False, applies the cutoff method to all cells together. Utilized in Step 2 of MDDC algorithm.

    if_col_corr : bool, optional, default=False
        Whether to use column (drug) correlation or row (adverse event) correlation
        If True, uses drug correlation instead of adverse event correlation. Utilized in Step 3 of the MDDC algorithm.

    corr_lim : float, optional, default=0.8
        Correlation threshold used to select connected adverse events. Utilized in Step 3 of MDDC algorithm.

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

    Returns:
    --------
    result : tuple
        A tuple with the following members:
        - 'pval': np.ndarray
            p-values for each cell in the step 2 of the algorithm, calculated using the Monte Carlo method for cells with count greater than five, and Fisher's exact test for cells with count less than or equal to five.
        - 'signal': np.ndarray
            Matrix indicating significant signals with count greater than five and identified in the step 2 by the Monte Carlo method. 1 indicates a signal, 0 indicates non-signal.
        - 'fisher_signal': np.ndarray
            Matrix indicating signals with a count less than or equal to five and identified by Fisher's exact test. 1 indicates a signal, 0 indicates non-signal.
        - 'corr_signal_pval': np.ndarray
            p-values for each cell in the contingency table in the step 5, when the :math:`r_{ij}` (residual) values are mapped back to the standard normal distribution.
        - 'corr_signal_adj_pval': np.ndarray
            Benjamini-Hochberg adjusted p values for each cell in the step 5.

    Notes
    ------
    This `chunk_size` option of the function function is designed to be used in scenarios where the contingency table dimensions and the number of Monte Carlo replications are large. In such cases,
    the Monte Carlo samples need to be generated sequentially to ensure that the memory footprint remains manageable and the generated samples fit into the available RAM.
    """

    if chunk_size is not None:
        c_univ_drug, null_dist_s = get_log_bootstrap_cutoff_sequential(
            contin_table, quantile, rep, chunk_size, seed
        )
    else:
        c_univ_drug, null_dist_s = get_log_bootstrap_cutoff(
            contin_table, quantile, rep, seed
        )

    z_ij_mat = getZijMat(contin_table, na=False)[0]
    log_Z_ij_mat = np.log(z_ij_mat)

    # Step 1: for each cell compute the standardized Pearson residual

    p_val_mat = np.array(
        list(
            map(
                lambda i: getPVal(log_Z_ij_mat[:, i], null_dist_s[:, i]),
                range(contin_table.shape[1]),
            )
        )
    ).T

    # Fisher Exact Test
    mask = (contin_table < 6) & (contin_table > 0)
    indices = np.where(mask)
    fisher_exact_test_vals = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_fisher_exact)(i, j, contin_table, exclude_same_drug_class)
        for i, j in zip(*indices, strict=False)
    )
    for (i, j), p_value in zip(
        zip(*indices, strict=False), fisher_exact_test_vals, strict=False
    ):
        p_val_mat[i, j] = p_value

    p_val_mat = np.nan_to_num(p_val_mat, nan=1)

    signal_mat = np.where((p_val_mat < (1 - quantile)) & (contin_table > 5), 1, 0)
    second_signal_mat = np.where(
        (p_val_mat < (1 - quantile)) & (contin_table < 6), 1, 0
    )

    res_all = z_ij_mat.flatten(order="F")
    # res_nonzero = z_ij_mat[contin_table != 0].flatten(order="F")
    res_zero = z_ij_mat[contin_table == 0].flatten(order="F")

    if col_specific_cutoff:
        if separate:
            zero_drug_cutoff = np.array(
                list(
                    map(
                        lambda a: compute_whislo1(z_ij_mat, contin_table, a),
                        range(contin_table.shape[1]),
                    )
                )
            )
        else:
            zero_drug_cutoff = np.apply_along_axis(compute_whislo2, 0, z_ij_mat)
    else:
        if separate:
            zero_drug_cutoff = np.repeat(
                boxplot_stats(res_zero)[0]["whislo"], contin_table.shape[1]
            )
        else:
            zero_drug_cutoff = np.repeat(
                boxplot_stats(res_all)[0]["whislo"], contin_table.shape[1]
            )

    # Step 2: apply univariate outlier detection to all the cells

    high_outlier = (z_ij_mat > np.exp(c_univ_drug)) * 1
    low_outlier = (z_ij_mat < -np.exp(c_univ_drug)) * 1
    zero_cell_outlier = ((z_ij_mat < zero_drug_cutoff) & (contin_table == 0)) * 1

    if_outlier_mat = ((high_outlier + low_outlier + zero_cell_outlier) != 0) * 1

    u_ij_mat = np.where((1 - if_outlier_mat), z_ij_mat, np.nan)

    # Step 3 & 4: consider the bivariate relations between AEs and predict values based on the connected AEs

    cor_u = pearsonCorWithNA(u_ij_mat, if_col_corr)

    iter_over = contin_table.shape[1] if if_col_corr else contin_table.shape[0]

    z_ij_hat_mat = np.full(contin_table.shape, fill_value=np.nan)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_index)(i, cor_u, corr_lim, contin_table, if_col_corr, u_ij_mat)
        for i in range(iter_over)
    )

    # Process the results to update the main variables
    for z_ij_hat, i, *_ in results:
        if len(z_ij_hat) != 0:  # Checks if z_ij_hat is not empty
            if if_col_corr:
                z_ij_hat_mat[:, i] = z_ij_hat
            else:
                z_ij_hat_mat[i, :] = z_ij_hat

    # Step 5: standardize the residuals within each drug column and flag outliers
    R_ij_mat = z_ij_mat - z_ij_hat_mat
    r_ij_mat = np.apply_along_axis(normalize_column, 0, R_ij_mat)
    r_pval = 1 - scipy.stats.norm.cdf(r_ij_mat)

    r_pval_adj = r_pval.copy().flatten()

    r_pval_nan_removed = r_pval_adj[~np.isnan(r_pval_adj)]
    bh_values = scipy.stats.false_discovery_control(r_pval_nan_removed, axis=None)
    r_adj_pval_nan_mask = ~np.isnan(r_pval_adj)
    r_pval_adj[r_adj_pval_nan_mask] = bh_values

    r_pval_adj = r_pval_adj.reshape(contin_table.shape)
    return (p_val_mat, signal_mat, second_signal_mat, r_pval, r_pval_adj)
