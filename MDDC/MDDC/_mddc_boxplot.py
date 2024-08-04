import numpy as np
import scipy
from mddc_cpp_helper import getZijMat, pearsonCorWithNA

from ._helper import (
    boxplot_stats,
    compute_whishi1,
    compute_whishi2,
    compute_whislo1,
    compute_whislo2,
    normalize_column,
)


def _mddc_boxplot(
    contin_table,
    col_specific_cutoff=True,
    separate=True,
    if_col_corr=False,
    corr_lim=0.8,
):
    """
    Modified Detecting Deviating Cells (MDDC) algorithm for adverse event signal identification with boxplot method for cutoff selection.

    This function implements the MDDC algorithm using the Boxplot method to determine cutoffs for identifying cells with high standardized Pearson residuals.

    For details on the algorithm please see :ref:`MDDC Algorithm <mddc_algorithm>`.

    Parameters
    ----------
    contin_table : pd.DataFrame or np.ndarray
        A contingency table of shape (I, J) where rows represent adverse events and columns represent drugs.
        If a DataFrame, it might have index and column names corresponding to the adverse events and drugs.

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
    
    Returns
    -------
    result : tuple
        A tuple with the following members:
        - 'signal': np.ndarray
            Matrix indicating significant signals with count greater than five and identified in the step 2 by the Monte Carlo method. 1 indicates a signal, 0 indicates non-signal.
        - 'corr_signal_pval': np.ndarray
            p-values for each cell in the contingency table in the step 5, when the :math:`r_{ij}` (residual) values are mapped back to the standard normal distribution.
        - 'corr_signal_adj_pval': np.ndarray
            Benjamini-Hochberg adjusted p values for each cell in the step 5.
    """

    z_ij_mat = getZijMat(contin_table, na=False)[0]
    res_all = z_ij_mat.flatten(order="F")
    res_nonzero = z_ij_mat[contin_table != 0].flatten(order="F")
    res_zero = z_ij_mat[contin_table == 0].flatten(order="F")

    if col_specific_cutoff:
        if separate:
            c_univ_drug = np.array(
                list(
                    map(
                        lambda a: compute_whishi1(z_ij_mat, contin_table, a),
                        range(contin_table.shape[1]),
                    )
                )
            )
            zero_drug_cutoff = np.array(
                list(
                    map(
                        lambda a: compute_whislo1(z_ij_mat, contin_table, a),
                        range(contin_table.shape[1]),
                    )
                )
            )
        else:
            c_univ_drug = np.apply_along_axis(compute_whishi2, 0, z_ij_mat)
            zero_drug_cutoff = np.apply_along_axis(compute_whislo2, 0, z_ij_mat)
    else:
        if separate:
            c_univ_drug = np.repeat(
                boxplot_stats(res_nonzero)[3], contin_table.shape[1]
            )

            zero_drug_cutoff = np.repeat(
                boxplot_stats(res_zero)[1], contin_table.shape[1]
            )
        else:
            c_univ_drug = np.repeat(boxplot_stats(res_all)[3], contin_table.shape[1])
            zero_drug_cutoff = np.repeat(
                boxplot_stats(res_all)[1], contin_table.shape[1]
            )

    # Step 2: apply univariate outlier detection to all the cells

    high_outlier = (z_ij_mat > c_univ_drug) * 1
    low_outlier = (z_ij_mat < -c_univ_drug) * 1
    zero_cell_outlier = ((z_ij_mat < zero_drug_cutoff) & (contin_table == 0)) * 1

    if_outlier_mat = ((high_outlier + low_outlier + zero_cell_outlier) != 0) * 1

    u_ij_mat = np.where((1 - if_outlier_mat), z_ij_mat, np.nan)

    # Step 3 & 4: consider the bivariate relations between AEs and predict values based on the connected AEs

    cor_u = pearsonCorWithNA(u_ij_mat, if_col_corr)

    iter_over = contin_table.shape[1] if if_col_corr else contin_table.shape[0]

    cor_list = []
    weight_list = []
    fitted_value_list = []
    coeff_list = []
    z_ij_hat_mat = np.full(contin_table.shape, fill_value=np.nan)

    for i in range(iter_over):
        idx = np.where(np.abs(cor_u[i, :]) >= corr_lim)[0]
        cor_list.append(idx[idx != i])

        weight = np.zeros_like(cor_u[i])
        weight[cor_list[i]] = np.abs(cor_u[i, cor_list[i]])
        weight_list.append(weight)

        if len(cor_list[i]) == 0:
            fitted_value_list.append(np.array([]))
            continue

        fitted_values = np.full(contin_table.shape, np.nan)
        if if_col_corr:
            for k in cor_list[i]:
                beta = scipy.stats.linregress(u_ij_mat[:, k], u_ij_mat[:, i])
                fit_values = u_ij_mat[:, k] * beta.slope + beta.intercept
                fitted_values[:, k] = fit_values
                coeff_list.append([beta.intercept, beta.slope])
        else:
            for k in cor_list[i]:
                var_x = u_ij_mat[k, :]
                var_y = u_ij_mat[i, :]
                mask = ~np.isnan(var_x) & ~np.isnan(var_y)
                beta = scipy.stats.linregress(var_x[mask], var_y[mask])
                fit_values = u_ij_mat[k, :] * beta.slope + beta.intercept
                fitted_values[k, :] = fit_values
                coeff_list.append([beta.intercept, beta.slope])

        nan_mask = np.isnan(fitted_values)
        weight_array = np.array(weight_list[i])
        if if_col_corr:
            wt_avg_weights = np.where(
                nan_mask,
                0,
                np.tile(weight_array.reshape(1, -1), contin_table.shape[0]).reshape(
                    contin_table.shape
                ),
            )
            z_ij_hat_mat[:, i] = np.ma.average(
                np.nan_to_num(fitted_values, 0), weights=wt_avg_weights, axis=1
            ).data
        else:
            wt_avg_weights = np.where(
                nan_mask,
                0,
                np.tile(weight_array.reshape(-1, 1), contin_table.shape[1]).reshape(
                    contin_table.shape
                ),
            )
            z_ij_hat_mat[i, :] = np.ma.average(
                np.nan_to_num(fitted_values, 0), weights=wt_avg_weights, axis=0
            ).data

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

    return (high_outlier, r_pval, r_pval_adj)
