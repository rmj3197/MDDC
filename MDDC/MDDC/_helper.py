"""
`_helper.py` contains a number of helper functions that are used in MDDC.
"""

import numpy as np
import scipy
from mddc_cpp_helper import getFisherExactTestTable, getZijMat


def apply_func(row, n, m):
    """
    Applies the `getZijMat` function to a reshaped version of the input row.

    This function takes a row vector, reshapes it into an n by m matrix, and then applies the `getZijMat` function to it.
    The function returns the first element of the result from `getZijMat`.

    Parameters
    -----------
    row : numpy.ndarray
        The input row vector to be reshaped and processed.
    n : int
        The number of rows for reshaping the input row vector.
    m : int
        The number of columns for reshaping the input row vector.

    Returns
    -------
    result : numpy.ndarray
        Returns the standardized Pearson residuals Zij for the selected row.
    """
    return getZijMat(row.reshape(n, m), True)[0]


def max_log_col(matrix):
    """
    Computes the maximum of the logarithm of each column in the input matrix, ignoring NaNs.

    This function takes a matrix, computes the natural logarithm of each element,
    and then returns the maximum value for each column, while ignoring NaNs.

    Parameters
    ----------
    matrix : numpy.ndarray
        A 2D array or matrix containing numerical values.

    Returns
    -------
    result : numpy.ndarray
        A 1D array containing the maximum of the logarithm of each column, ignoring NaNs.
    """
    return np.nanmax(np.log(matrix), axis=0)


def get_log_bootstrap_cutoff(
    contin_table,
    quantile=0.95,
    rep=3,
    seed=None,
):
    """
    Computes the bootstrap cutoffs for the maximum logarithm of bootstrap samples
    based on a contingency table.

    This function generates bootstrap samples from a contingency table, applies
    the `getZijMat` function to each sample, computes the logarithm of each element,
    and then returns the quantile cutoff for the maximum log values across all
    bootstrap samples.

    Parameters
    ----------
    contin_table : numpy.ndarray
        A 2D contingency table containing numerical values.
    quantile : float, optional
        The quantile value to be computed for the cutoff. Default is 0.95.
    rep : int, optional
        The number of bootstrap replications. Default is 3.
    seed : int or None, optional
        A seed for the random number generator to ensure reproducibility. Default is None.

    Returns
    -------
    cutoffs : numpy.ndarray
        The quantile cutoff values for the maximum log values across bootstrap samples.
    max_list : numpy.ndarray
        The maximum log values across all bootstrap samples.
    """

    z_ij_mat, n_dot_dot, p_i_dot, p_dot_j = getZijMat(contin_table)

    n, m = contin_table.shape

    p_i_dot = p_i_dot.reshape(contin_table.shape[0], 1)
    p_dot_j = p_dot_j.reshape(1, contin_table.shape[1])
    p_mat = p_i_dot * p_dot_j

    generator = np.random.RandomState(seed)
    sim_tables = generator.multinomial(n=n_dot_dot, pvals=p_mat.flatten(), size=rep)

    z_ij_mat_list = np.apply_along_axis(
        lambda row: apply_func(row, n, m), 1, sim_tables
    )

    max_list = np.apply_along_axis(max_log_col, 1, z_ij_mat_list)

    cutoffs = np.quantile(max_list, quantile, axis=0)

    return (cutoffs, max_list)


def fivenum(x, na_rm=True):
    """
    Computes Tukey's five-number summary for a given array. This is based on `fivenum` function in `R` base package `stats`.

    This function calculates the minimum, lower-hinge, median, upper-hinge, maximum.
    and maximum values of the input array, handling NaN values according to the `na_rm` parameter.

    Parameters
    ----------
    x : numpy.ndarray
        A 1D array or sequence of numerical values.
    na_rm : bool, optional
        If True, NaN values are removed before computation. If False,
        and any NaNs are present in the input array, the result will be an array of NaNs.
        Default is True.

    Returns
    -------
    summary : numpy.ndarray
        A 1D array containing the five-number summary: [minimum, lower-hinge, median, upper-hinge, maximum].
        If the input array is empty or contains only NaNs and `na_rm` is False,
        the result will be an array of NaNs.
    """

    x = np.asarray(x)

    if np.any(np.isnan(x)):
        if na_rm:
            x = x[~np.isnan(x)]
        else:
            return np.full(5, np.nan)

    x = np.sort(x)
    n = len(x)

    if n == 0:
        return np.full(5, np.nan)
    else:
        n4 = (n + 3) // 2 / 2
        d = np.array([1, n4, (n + 1) / 2, n + 1 - n4, n], dtype=float) - 1
        return 0.5 * (x[np.floor(d).astype(int)] + x[np.ceil(d).astype(int)])


def boxplot_stats(x, coef=1.5, na_rm=True):
    """
    Computes boxplot statistics for a given array. This is based on `boxplot.stats` function in `R` base package `grDevices`.

    Parameters:
    ----------
    x : numpy.ndarray
        A 1D array or sequence of numerical values.
    coef : float, optional
        The coefficient used to determine the length of the whiskers. Default is 1.5.
    na_rm : bool, optional
        If True, NaN values are removed before computation. If False,
        and any NaNs are present in the input array, the result will be an array of NaNs.
        Default is True.

    Returns:
    -------
    stats : numpy.ndarray
        A 1D array containing the boxplot statistics: [extreme of the lower whisker, the lower `hinge`,
        the median, the upper `hinge` and the extreme of the upper whisker].
        If the input array is empty or contains only NaNs and `na_rm` is False,
        the result will be an array of NaNs.
    """
    x = np.asarray(x)
    if coef < 0:
        raise ValueError("coef must not be negative")

    if na_rm:
        x = x[~np.isnan(x)]

    if x.size == 0:
        return np.array([np.nan] * 5)

    stats = fivenum(x)
    iqr = stats[3] - stats[1]
    low_lim = stats[1] - coef * iqr
    upper_lim = stats[3] + coef * iqr

    low_whisker = np.min(x[x > low_lim])
    upper_whisker = np.max(x[x < upper_lim])

    stats[1] = low_whisker
    stats[3] = upper_whisker
    return stats


def compute_fisher_exact(i, j, contin_table, exclude_same_drug_class):
    """
    Computes the p-value of Fisher's Exact Test for a specific contingency table subset.

    This function calculates the p-value of Fisher's Exact Test for a 2x2 contingency table
    derived from the input contingency table based on indices `i` and `j`. The function uses
    the `exclude_same_drug_class` parameter to determine how to construct the 2x2 table.

    Parameters:
    ----------
    i : int
        The index of the first category or group in the contingency table.
    j : int
        The index of the second category or group in the contingency table.
    contin_table : numpy.ndarray
        A 2D array or matrix representing the contingency table from which the 2x2 table will be derived.
    exclude_same_drug_class : bool
        A flag indicating whether to exclude the same drug class when creating the 2x2 table.

    Returns:
    -------
    p_value : float
        The p-value of Fisher's Exact Test for the 2x2 contingency table.
    """

    tabl = getFisherExactTestTable(contin_table, i, j, exclude_same_drug_class)
    return scipy.stats.fisher_exact(tabl)[1]


def normalize_column(a):
    """
    Normalizes a 1D array or column by standardizing it to have a mean of 0 and a standard deviation of 1.

    This function computes the z-score normalization of the input array, which involves subtracting
    the mean and dividing by the standard deviation of the array. NaN values are ignored in the
    calculations of the mean and standard deviation.

    Parameters:
    ----------
    a : numpy.ndarray
        A 1D array or sequence of numerical values to be normalized.

    Returns:
    -------
    normalized_a : numpy.ndarray
        A 1D array with the same shape as the input array, where each value has been normalized
        to have a mean of 0 and a standard deviation of 1.
    """
    mean_a = np.nanmean(a)
    sd_a = np.nanstd(a, ddof=1)
    return (a - mean_a) / sd_a


def compute_whishi1(z_ij_mat, contin_table, a):
    """
    Computes the upper whisker value from the boxplot statistics of a specific column in `z_ij_mat`.

    This function extracts the values from the `z_ij_mat` matrix corresponding to a specified column in the `contin_table`
    where the values are non-zero. It then computes the boxplot statistics for these extracted values and returns the
    upper whisker value.

    Parameters:
    ----------
    z_ij_mat : numpy.ndarray
        A 2D array or matrix from which the column values are extracted for boxplot statistics computation.
    contin_table : numpy.ndarray
        A 2D array or matrix containing the indices to determine which rows of `z_ij_mat` to consider.
    a : int
        The index of the column in `contin_table` to be used for filtering and extracting values from `z_ij_mat`.

    Returns:
    -------
    upper_whisker : float
        The upper whisker value from the boxplot statistics of the extracted values.
    """

    return boxplot_stats(z_ij_mat[contin_table[:, a] != 0, a])[3]


def compute_whishi2(vec):
    """
    Computes the upper whisker value from the boxplot statistics of the given vector.

    This function calculates the boxplot statistics for the input vector and returns the upper whisker value.
    The upper whisker is defined as the maximum value within 1.5 times the interquartile range (IQR) above the upper hinge (Q3) in boxplot statistics.

    Parameters:
    ----------
    vec : numpy.ndarray
        A 1D array or sequence of numerical values for which the boxplot statistics are to be computed.

    Returns:
    -------
    upper_whisker : float
        The upper whisker value from the boxplot statistics of the input vector.
    """
    return boxplot_stats(vec)[3]


def compute_whislo1(z_ij_mat, contin_table, a):
    """
    Computes the lower whisker value from the boxplot statistics of a specific column in `z_ij_mat`.

    This function extracts values from the `z_ij_mat` matrix corresponding to a specified column in the `contin_table`
    where the values are zero. It then calculates the boxplot statistics for these extracted values and returns the
    lower whisker value.

    Parameters:
    ----------
    z_ij_mat : numpy.ndarray
        A 2D array or matrix from which the column values are extracted for boxplot statistics computation.
    contin_table : numpy.ndarray
        A 2D array or matrix containing the indices to determine which rows of `z_ij_mat` to consider.
    a : int
        The index of the column in `contin_table` to be used for filtering and extracting values from `z_ij_mat`.

    Returns:
    -------
    lower_whisker : float
        The lower whisker value from the boxplot statistics of the extracted values.
    """
    return boxplot_stats(z_ij_mat[contin_table[:, a] == 0, a])[1]


def compute_whislo2(vec):
    """
    Computes the lower whisker value from the boxplot statistics of the given vector.

    This function calculates the boxplot statistics for the input vector and returns the lower whisker value.
    The lower whisker is defined as the minimum value within 1.5 times the interquartile range (IQR) below the lower hinge (Q1) in boxplot statistics.

    Parameters:
    ----------
    vec : numpy.ndarray
        A 1D array or sequence of numerical values for which the boxplot statistics are to be computed.

    Returns:
    -------
    lower_whisker : float
        The lower whisker value from the boxplot statistics of the input vector.
    """
    return boxplot_stats(vec)[1]
