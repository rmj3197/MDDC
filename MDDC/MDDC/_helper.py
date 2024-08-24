"""
`_helper.py` contains a number of helper functions that are used in MDDC.
"""

import numpy as np
import scipy
from mddc_cpp_helper import getFisherExactTestTable, getZijMat


def apply_func(row, n, m, na=True):
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
    na : bool
        whether NaN should be returned for cells where count is less
        than 6.

    Returns
    -------
    result : numpy.ndarray
        Returns the standardized Pearson residuals Zij for the selected row.
    """
    return getZijMat(row.reshape(n, m), na)[0]


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


def _process_chunk(chunk_rep, n, m, n_dot_dot, p_mat, seed, iter):
    """
    Process a chunk of data by generating multinomial samples and applying
    transformation functions to compute the maximum log column for each sample.

    Parameters
    ----------
    chunk_rep : int
        The number of replications or samples to generate in this chunk.

    n : int
        The number of rows in the contingency table.

    m : int
        The number of columns in the contingency table.

    n_dot_dot : int
        The total number of observations across all cells in the contingency table.

    p_mat : np.ndarray
        A matrix of probabilities for the multinomial distribution. The matrix
        should have shape `(n, m)` and represent the probability for each cell.

    seed : int or None
        A seed value for random number generation. If `None`, the seed is not set.
        The seed is modified by the `iter` value to ensure different sequences
        in different iterations.

    iter : int
        The current iteration number, used to modify the seed for random number
        generation.

    Returns
    -------
    np.ndarray
        A 1D array of maximum log column values, computed from the generated
        multinomial samples and their corresponding `Z_ij` matrices.
    """
    if seed is not None:
        seed = seed * iter
    generator = np.random.RandomState(seed)
    sim_tables_chunk = generator.multinomial(
        n=n_dot_dot, pvals=p_mat.flatten(), size=chunk_rep
    )
    z_ij_mat_list_chunk = np.apply_along_axis(
        lambda row: apply_func(row, n, m), 1, sim_tables_chunk
    )
    return np.apply_along_axis(max_log_col, 1, z_ij_mat_list_chunk)


def get_log_bootstrap_cutoff_sequential(
    contin_table,
    quantile=0.95,
    rep=3,
    chunk_size=1,
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

    chunks = [chunk_size] * (rep // chunk_size) + [rep % chunk_size]
    chunks = [chunk for chunk in chunks if chunk > 0]

    max_list = [
        _process_chunk(chunk, n, m, n_dot_dot, p_mat, seed, i)
        for i, chunk in enumerate(chunks)
    ]
    max_list = np.concatenate(max_list, axis=0)
    cutoffs = np.quantile(max_list, quantile, axis=0)

    return (cutoffs, max_list)


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

    low_whisker = np.min(x[x >= low_lim])
    upper_whisker = np.max(x[x <= upper_lim])

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


def compute_whishi1(z_ij_mat, contin_table, coef, a):
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
    coef : float
        Coefficient used for computing boxplot statistics.
    a : int
        The index of the column in `contin_table` to be used for filtering and extracting values from `z_ij_mat`.

    Returns:
    -------
    upper_whisker : float
        The upper whisker value from the boxplot statistics of the extracted values.
    """

    return boxplot_stats(z_ij_mat[contin_table[:, a] != 0, a], coef=coef)[3]


def compute_whishi2(vec, coef):
    """
    Computes the upper whisker value from the boxplot statistics of the given vector.

    This function calculates the boxplot statistics for the input vector and returns the upper whisker value.
    The upper whisker is defined as the maximum value within 1.5 times the interquartile range (IQR) above the upper hinge (Q3) in boxplot statistics.

    Parameters:
    ----------
    vec : numpy.ndarray
        A 1D array or sequence of numerical values for which the boxplot statistics are to be computed.
    coef : float
        Coefficient used for computing boxplot statistics.

    Returns:
    -------
    upper_whisker : float
        The upper whisker value from the boxplot statistics of the input vector.
    """
    return boxplot_stats(vec, coef=coef)[3]


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


# no cover: start
def process_index(i, cor_u, corr_lim, contin_table, if_col_corr, u_ij_mat):
    """
    Process the correlation of a specific index with others and fit a linear regression model
    to predict the values based on correlated variables.

    Parameters:
    -----------
    i : int
        The index of the variable to process.

    cor_u : numpy.ndarray
        The correlation matrix with shape (n, n), where `n` is the number of variables.

    corr_lim : float
        The threshold for correlation. Variables with absolute correlation greater than or equal
        to this value with `i` are considered for linear regression.

    contin_table : numpy.ndarray
        The contingency table or data matrix with shape (m, n), where `m` is the number of samples
        and `n` is the number of variables.

    if_col_corr : bool
        Flag indicating whether to treat `u_ij_mat` as columns (True) or rows (False) for
        correlation.

    u_ij_mat : numpy.ndarray
        The matrix containing the variables to be used in the regression, with shape (m, n)
        or (n, m) depending on `if_col_corr`.

    Returns:
    --------
    z_ij_hat : numpy.ndarray
        The fitted values averaged and weighted across correlated variables, with shape (m,) if
        `if_col_corr` is True, or (n,) if False.

    i : int
        The index of the processed variable.

    cor_list : list
        A list containing indices of variables that are highly correlated with the variable at
        index `i`.

    weight_list : list
        A list containing the weights based on the absolute correlations for the selected variables.

    fitted_value_list : list
        A list of numpy arrays containing the fitted values from the linear regression for each
        correlated variable.

    coeff_list : list
        A list of lists, each containing the intercept and slope of the linear regression model
        fitted for each correlated variable.

    Notes
    -----
    - If no variables are found to be highly correlated with the variable at index `i`,
      the function returns empty arrays for `z_ij_hat` and `fitted_value_list`, and `cor_list`
      and `weight_list` will contain empty entries.
    - The function handles missing values (`NaN`) by excluding them from the linear regression
      calculation and setting the corresponding fitted values to `NaN`.
    - The fitted values are weighted and averaged across all correlated variables to obtain the
      final predicted values, `z_ij_hat`.
    """
    cor_list = []
    weight_list = []
    fitted_value_list = []
    coeff_list = []

    idx = np.where(np.abs(cor_u[i, :]) >= corr_lim)[0]
    cor_list.append(idx[idx != i])

    weight = np.zeros_like(cor_u[i])
    weight[cor_list[0]] = np.abs(cor_u[i, cor_list[0]])
    weight_list.append(weight)

    if len(cor_list[0]) == 0:
        fitted_value_list.append(np.array([]))
        return (
            np.array([]),
            np.array([]),
            cor_list,
            weight_list,
            fitted_value_list,
            coeff_list,
        )

    fitted_values = np.full(contin_table.shape, np.nan)
    if if_col_corr:
        for k in cor_list[0]:
            beta = scipy.stats.linregress(u_ij_mat[:, k], u_ij_mat[:, i])
            fit_values = u_ij_mat[:, k] * beta.slope + beta.intercept
            fitted_values[:, k] = fit_values
            coeff_list.append([beta.intercept, beta.slope])
    else:
        for k in cor_list[0]:
            var_x = u_ij_mat[k, :]
            var_y = u_ij_mat[i, :]
            mask = ~np.isnan(var_x) & ~np.isnan(var_y)
            beta = scipy.stats.linregress(var_x[mask], var_y[mask])
            fit_values = u_ij_mat[k, :] * beta.slope + beta.intercept
            fitted_values[k, :] = fit_values
            coeff_list.append([beta.intercept, beta.slope])

    nan_mask = np.isnan(fitted_values)
    weight_array = np.array(weight_list[0])
    if if_col_corr:
        any_all_nan = np.all(nan_mask, axis=1)
        wt_avg_weights = np.where(
            nan_mask,
            0,
            np.tile(weight_array.reshape(1, -1), contin_table.shape[0]).reshape(
                contin_table.shape
            ),
        )
        z_ij_hat = np.ma.average(
            np.nan_to_num(fitted_values, 0), weights=wt_avg_weights, axis=1
        ).data
        z_ij_hat[any_all_nan] = np.nan
    else:
        any_all_nan = np.all(nan_mask, axis=0)
        wt_avg_weights = np.where(
            nan_mask,
            0,
            np.tile(weight_array.reshape(-1, 1), contin_table.shape[1]).reshape(
                contin_table.shape
            ),
        )
        z_ij_hat = np.ma.average(
            np.nan_to_num(fitted_values, 0), weights=wt_avg_weights, axis=0
        ).data
        z_ij_hat[any_all_nan] = np.nan
    return z_ij_hat, i, cor_list, weight_list, fitted_value_list, coeff_list


# no cover: stop


def get_boxplot_outliers(dat, c_j):
    """
    Identifies outliers in the data based on the boxplot method.

    This function computes the interquartile range (IQR) and uses it to identify outliers
    as values that exceed a threshold defined by the IQR and a scaling factor `c_j`.

    Parameters:
    -----------
    dat : numpy.ndarray
        The data array from which to detect outliers. Can contain NaN values.
    c_j : float
        The scaling factor to adjust the outlier threshold.

    Returns:
    --------
    outliers : numpy.ndarray
        A boolean array where True indicates an outlier in the corresponding position.
    """
    q3 = np.nanquantile(dat, 0.75)
    q1 = np.nanquantile(dat, 0.25)
    outliers = dat > (q3 + c_j * (q3 - q1))
    return outliers


def compute_fdr(res_list, c_j, j):
    """
    Computes the mean number of outliers for a given component in a 3D array.

    This function applies the `get_boxplot_outliers` function across the rows of the
    specified component of the `res_list` array and calculates the mean number of outliers
    across the rows.

    Parameters:
    -----------
    res_list : numpy.ndarray
        A 3D array containing the result values, where outliers will be computed along the rows.
    c_j : float
        The scaling factor passed to `get_boxplot_outliers` for determining the outlier threshold.
    j : int
        The index of the third dimension in `res_list` that specifies which component to analyze.

    Returns:
    --------
    mean_outliers : float
        The mean number of outliers across the rows of the specified component.
    """
    outliers = np.apply_along_axis(
        lambda x: get_boxplot_outliers(x, c_j), axis=1, arr=res_list[:, :, j]
    )
    sum_outliers = np.sum(outliers, axis=1)
    mean_outliers = np.mean(sum_outliers)
    return mean_outliers


def compute_fdr_all(res_list, c):
    """
    Computes the mean number of outliers for a 3D array.

    This function calculates the average number of outliers across a number of
    datasets.

    Parameters:
    -----------
    res_list : numpy.ndarray
        A 3D array containing the result values, where outliers will be computed along the rows.
    c : float
        The scaling factor passed to `get_boxplot_outliers` for determining the outlier threshold.

    Returns:
    --------
    mean_outliers : float
        The mean number of outliers across the rows of the specified component.
    """
    outliers = np.apply_along_axis(
        lambda x: get_boxplot_outliers(x, c),
        axis=1,
        arr=res_list.reshape(res_list.shape[0], res_list.shape[1] * res_list.shape[2]),
    )
    sum_outliers = np.sum(outliers, axis=1)
    mean_outliers = np.mean(sum_outliers)
    return mean_outliers
