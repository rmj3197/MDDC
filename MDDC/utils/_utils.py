"""
`utils.py` contains additional utility functions made available to a user.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mddc_cpp_helper import getEijMat, getZijMat


def _block_diagonal(*matrices):
    """
    Create a block diagonal matrix from a list of 2D NumPy arrays.

    This function constructs a block diagonal matrix where the input matrices
    are placed along the diagonal and zeros are filled in the off-diagonal blocks.

    Parameters
    ----------
    *matrices : tuple of numpy.ndarray
        A variable number of 2D NumPy arrays to be combined into a block diagonal matrix.
        Each matrix should be a 2D array. The matrices are placed along the diagonal
        in the order they are provided.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array representing the block diagonal matrix. The resulting matrix
        will have a shape equal to the sum of the row and column dimensions of the input
        matrices.
    """

    rows = sum(matrix.shape[0] for matrix in matrices)
    cols = sum(matrix.shape[1] for matrix in matrices)

    result = np.zeros((rows, cols))

    current_row = 0
    current_col = 0
    for matrix in matrices:
        r, c = matrix.shape
        result[current_row : current_row + r, current_col : current_col + c] = matrix
        current_row += r
        current_col += c

    return result


def _generate_simulated_zijmat(
    n_rows,
    n_columns,
    signal_mat,
    cov_matrix,
    p_i_dot,
    p_dot_j,
    e_ij_mat,
    rep,
    seed,
):
    """
    Generate a simulated contingency table matrix with cluster-specific adverse event correlations.

    This function creates a simulated contingency table matrix (`z_ij_mat`) where the correlations of adverse events
    within each cluster are accounted for. The simulation incorporates the correlation structure specified by `rho`
    and adjusts the matrix according to the input `signal_mat` and `e_ij_mat`.

    Parameters:
    -----------
    n_rows : int
        Number of rows in the simulated table.

    n_columns : int
        Number of columns in the simulated table.

    signal_mat : numpy.ndarray
        A matrix with the same dimensions as `contin_table`, where entries represent the signal strength associated
        with each element in the contingency table.

    cluster_idx : numpy.ndarray
        An array indicating the cluster index for each row in `contin_table`. Each row in the matrix is assigned to
        a specific cluster.

    count_dict : dict
        A dictionary where keys are cluster indices and values are the number of elements in each cluster.

    rho : float
        The correlation coefficient for adverse events within each cluster. A value between 0 and 1 indicates the
        strength of correlation, with 0 meaning no correlation and 1 meaning perfect correlation.

    p_i_dot : numpy.ndarray
        The row marginal probabilities of `e_ij_mat`, computed as the sum of `e_ij_mat` along columns, with axis
        kept.

    p_dot_j : numpy.ndarray
        The column marginal probabilities of `e_ij_mat`, computed as the sum of `e_ij_mat` along rows, with axis kept.

    e_ij_mat : numpy.ndarray
        A matrix representing expected values used in the simulation.

    rep : int
        The replication index for generating random numbers. This index is combined with `seed` to initialize the
        random number generator.

    seed : int or None
        The random seed for reproducibility of the simulation. If None, a random seed is used.

    Returns:
    --------
    simulated contingency table : numpy.ndarray
        The simulated contingency table matrix. The values are rounded and any negative values are set to 0.
    """

    if seed is not None:
        generator = np.random.RandomState(seed * rep)
    else:
        generator = np.random.RandomState(None)

    z_ij_mat = generator.multivariate_normal(
        mean=np.zeros(n_rows), cov=cov_matrix, size=n_columns
    ).T
    new_contin_table = np.round(
        z_ij_mat * np.sqrt(e_ij_mat * signal_mat * ((1 - p_i_dot) @ (1 - p_dot_j)))
        + e_ij_mat * signal_mat
    )
    new_contin_table[new_contin_table <= 0] = 0
    return new_contin_table


def generate_contin_table_with_clustered_AE(
    row_marginal,
    column_marginal,
    signal_mat,
    contin_table=None,
    cluster_idx=None,
    n=100,
    rho=None,
    n_jobs=-1,
    seed=None,
):
    """
    Generate simulated contingency tables with optional incorporation of adverse event correlation within clusters.

    This function generates multiple simulated contingency tables based on the input row and column marginals,
    or `contin_table`, signal strength matrix (`signal_mat`), and cluster indices (`cluster_idx`).
    It incorporates adverse event correlation within each cluster according to the specified correlation
    parameter (`rho`).

    Parameters:
    -----------
    row_marginal : list, np.ndarray, None
        Marginal sums for the rows of the contingency table.

    column_marginal : list, np.ndarray, None
        Marginal sums for the columns of the contingency table.

    signal_mat : numpy.ndarray, pandas.DataFrame
        A data matrix of the same dimensions as `contin_table`, where entries represent signal strength. Values
        should be greater than or equal to 1, where 1 indicates no signal and values greater than 1 indicate the
        presence of a signal.

    contin_table : numpy.ndarray, pandas.DataFrame, default=None
        A data matrix representing an I x J contingency table with row (adverse event) and column (drug) names.
        The row and column marginals of this table are used to generate the simulated data. It is advisable to
        check the input contingency table using the function `check_and_fix_contin_table()` before using this function.

    cluster_idx : numpy.ndarray, list, pd.DataFrame
        An array indicating the cluster index for each row in the `contin_table`. Clusters can be represented by
        names or numerical indices.

    n : int, optional, default=100
        The number of simulated contingency tables to generate.

    rho : float, optional, numpy.ndarray, default=None
        - If a float or int, `rho` represents the correlation value to be used between elements within each cluster
          specified by `cluster_idx`.
        - If a numpy.ndarray, `rho` must be a square matrix with dimensions equal to the number of rows in `contin_table`.
          In this case the `cluster_idx` is not used.
        - If None, a covariance matrix is generated based on the correlation coefficients of `contin_table`.

    n_jobs : int, optional, default=-1
        n_jobs specifies the maximum number of concurrently
        running workers. If 1 is given, no joblib parallelism
        is used at all, which is useful for debugging. For more
        information on joblib `n_jobs` refer to -
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.

    seed : int, optional, default=None
        Random seed for reproducibility of the simulation.

    Returns:
    --------
    simulated tables : list of numpy.ndarray
        A list containing the simulated contingency tables.
    """
    is_dataframe = False
    if isinstance(rho, (int, float)):
        if not (0 <= rho <= 1):
            raise ValueError("The value of `rho` must lie between [0,1]")
        if cluster_idx is not None:
            if not isinstance(cluster_idx, (list, np.ndarray, pd.DataFrame)):
                raise TypeError("cluster_idx must be a list or numpy array.")
            else:
                if (pd.DataFrame(cluster_idx).shape[1] == 1) or (
                    pd.DataFrame(cluster_idx).shape[0] == 1
                ):
                    cluster_idx = pd.DataFrame(cluster_idx).values.flatten()
                else:
                    raise ValueError(
                        "The pandas DataFrame is not of the correct format. \
                            Expected a dataframe with dimensions (n,1) or (1,n)"
                    )

                if contin_table is not None:
                    if not isinstance(contin_table, (pd.DataFrame, np.ndarray)):
                        raise TypeError(
                            "contin_table must be a pandas DataFrame or numpy array."
                        )

                    if contin_table.shape[0] != len(cluster_idx):
                        raise ValueError(
                            "The length of `cluster_idx` should be same as rows of `contin_table`."
                        )
                else:
                    if len(cluster_idx) != len(row_marginal):
                        raise ValueError(
                            "The length of `cluster_idx` should be same as length of `row_marginal`."
                        )

                groups, group_counts = np.unique(cluster_idx, return_counts=True)
                count_dict = dict(zip(groups, group_counts, strict=False))
                cov_matrices = {
                    group: np.full((count_dict[group], count_dict[group]), rho)
                    for group in groups
                }
                for group in groups:
                    np.fill_diagonal(cov_matrices[group], 1)
                cov_matrix = _block_diagonal(*cov_matrices.values())
        else:
            raise ValueError(
                "User provided `rho` but the `cluster_idx` is not provided."
            )
    elif isinstance(rho, np.ndarray):
        if rho.shape != (contin_table.shape[0],) * 2:
            raise ValueError(
                "Please check the shape of the input matrix `rho`. It should be I x I matrix where \
                                I is the number of rows in the contingency table."
            )
        else:
            cov_matrix = rho
    elif rho is None:
        if cluster_idx is not None:
            raise ValueError(
                "User provided the `cluster_idx` but `rho` has not been provided.\
                If user is unable to provide `rho`, then please set `cluster_idx`=None"
            )
        else:
            if contin_table is not None:
                cov_matrix = np.corrcoef(contin_table)
            else:
                raise ValueError(
                    "rho cannot be estimated if no `contin_table` is provided."
                )
    else:
        raise ValueError(
            "The rho must be None, a float or a numpy matrix of dimension I x I matrix where \
                            I is the number of rows in the contingency table."
        )

    if contin_table is not None:
        if not isinstance(contin_table, (pd.DataFrame, np.ndarray)):
            raise TypeError("contin_table must be a pandas DataFrame or numpy array.")

        if pd.DataFrame(contin_table).empty:
            raise ValueError("The `contin_table` cannot be empty")

        if isinstance(contin_table, pd.DataFrame):
            is_dataframe = True
            row_names = list(contin_table.index)
            column_names = list(contin_table.columns)
            contin_table = contin_table.values

        e_ij_mat = getEijMat(contin_table)
        n_i_dot = e_ij_mat.sum(axis=1, keepdims=True)
        n_dot_j = e_ij_mat.sum(axis=0, keepdims=True)
        n_dot_dot = e_ij_mat.sum()

        n_rows = contin_table.shape[0]
        n_columns = contin_table.shape[1]

    elif (
        (row_marginal is not None)
        and (column_marginal is not None)
        and (contin_table is None)
    ):
        if np.sum(row_marginal) == np.sum(column_marginal):
            pass
        else:
            raise AssertionError(
                "The sum of row and column \
                marginals do not match."
            )
        n_i_dot = np.array(row_marginal).reshape(-1, 1)
        n_dot_j = np.array(column_marginal).reshape(1, -1)
        n_dot_dot = np.sum(n_i_dot)
        e_ij_mat = (n_i_dot @ n_dot_j) / n_dot_dot

        n_rows = len(row_marginal)
        n_columns = len(column_marginal)
    else:
        if ((row_marginal is None) or (column_marginal is None)) and (
            contin_table is None
        ):
            raise ValueError(
                "`row_marginal` or `column_marginal` cannot be \
                None when `contin_table` is also None."
            )

    p_i_dot = n_i_dot / n_dot_dot
    p_dot_j = n_dot_j / n_dot_dot

    simulated_samples = Parallel(n_jobs=n_jobs)(
        delayed(_generate_simulated_zijmat)(
            n_rows,
            n_columns,
            signal_mat,
            cov_matrix,
            p_i_dot,
            p_dot_j,
            e_ij_mat,
            i,
            seed,
        )
        for i in range(n)
    )

    if is_dataframe:
        simulated_samples = list(
            map(
                lambda sample: pd.DataFrame(
                    sample, columns=column_names, index=row_names
                ),
                simulated_samples,
            )
        )

    return simulated_samples


def report_drug_AE_pairs(
    contin_table, contin_table_signal, along_rows="AE", along_columns="Drug"
):
    """
    Report potential adverse events for drugs based on the contingency table.

    This function analyzes the provided contingency table and signal matrix to identify potential adverse events
    associated with each drug. It computes the observed counts, expected counts, and standardized Pearson residuals
    for each (drug, adverse event) pair.

    Parameters:
    -----------
    contin_table : numpy.ndarray, pandas.DataFrame
        A data matrix representing an I x J contingency table with rows corresponding to adverse events and columns
        corresponding to drugs. The row and column names of this matrix are used in the analysis. It is advisable
        to check the input contingency table using the function `check_and_fix_contin_table()` before using this
        function.

    contin_table_signal : numpy.ndarray, pandas.DataFrame
        A data matrix of the same dimensions as `contin_table`, with entries of either 1 (indicating a signal) or
        0 (indicating no signal). This matrix should have the same row and column names as `contin_table` and can
        be obtained using the function `MDDC.MDDC.mddc()`.

    along_rows : str, optional, default = "AE"
        Specifies the content along the rows of the `contin_table` (e.g. AE or Drug).

    along_columns : str, optional, default = "Drug"
        Specifies the content along the columns of the `contin_table` (e.g. AE or Drug).

    Returns:
    --------
    Identified Drug-AE pairs : pandas.DataFrame
        A DataFrame with five columns:
            - `Drug` : str, The name of the drug. In case the `contin_table_signal` is a numpy.ndarray the `Drug` represents the column index.
            - `AE` : str, The potential adverse event associated with the drug. n case the `contin_table_signal` is a numpy.ndarray the `AE` represents the row index.
            - `Observed Count` : int, The observed count of the (drug, adverse event) pair.
            - `Expected Count` : float, The expected count of the (drug, adverse event) pair.
            - `Standard Pearson Residual` : float, The value of the standardized Pearson residual for the (drug, adverse event) pair.
    """
    if not (
        isinstance(contin_table, (np.ndarray, pd.DataFrame))
        and isinstance(contin_table_signal, (np.ndarray, pd.DataFrame))
    ):
        raise TypeError("Both inputs must be data matrices.")

    # Check if the dimensions match
    if contin_table.shape != contin_table_signal.shape:
        raise ValueError(
            "The dimensions of contin_table and contin_table_signal must be the same."
        )

    # Check if the row and column names match
    if isinstance(contin_table, pd.DataFrame) and isinstance(
        contin_table_signal, pd.DataFrame
    ):
        if not np.array_equal(contin_table.index, contin_table_signal.index):
            raise ValueError(
                "The row names of contin_table and contin_table_signal must match."
            )

        if not np.array_equal(contin_table.columns, contin_table_signal.columns):
            raise ValueError(
                "The column names of contin_table and contin_table_signal must match."
            )

    if not (isinstance(along_rows, str) and isinstance(along_columns, str)):
        raise TypeError("The `along_rows` and `along_columns` values must be string.")

    if isinstance(contin_table_signal, pd.DataFrame):
        row_names = list(contin_table_signal.index)
        column_names = list(contin_table_signal.columns)
        contin_table_signal = contin_table_signal.values
    else:
        row_names = range(contin_table.shape[0])
        column_names = range(contin_table_signal.shape[1])

    if isinstance(contin_table, pd.DataFrame):
        contin_table = contin_table.values

    contin_table_signal = np.where(
        np.isnan(contin_table_signal), 0, contin_table_signal
    )

    mat_expected_count = np.round(getEijMat(contin_table), 4)
    mat_std_res = np.round(getZijMat(contin_table)[0], 4)

    pairs = []

    for j in range(contin_table_signal.shape[1]):
        for i in range(contin_table_signal.shape[0]):
            if contin_table_signal[i, j] == 1 and contin_table[i, j] != 0:
                pairs.append(
                    [
                        column_names[j],
                        row_names[i],
                        contin_table[i, j],
                        mat_expected_count[i, j],
                        mat_std_res[i, j],
                    ]
                )

    if len(pairs) > 0:
        # Combine list of pairs into a DataFrame
        pairs_df = pd.DataFrame(
            pairs,
            columns=[
                along_columns,
                along_rows,
                "Observed Count",
                "Expected Count",
                "Standard Pearson Residual",
            ],
        )
        return pairs_df
    else:
        warnings.warn(
            "Empty DataFrame returned as no signals are found.",
            category=RuntimeWarning,
            stacklevel=1,
        )
        return pd.DataFrame(
            columns=[
                along_columns,
                along_rows,
                "Observed Count",
                "Expected Count",
                "Standard Pearson Residual",
            ]
        )


def plot_heatmap(data, size_cell=0.20, **kwargs):
    """
    Plot a heatmap of the specified attribute from the `mddc_result`.

    This function generates a heatmap for a given attribute of `mddc_result`, which should be a named tuple with
    fields that include the attribute specified by the `plot` parameter. The heatmap is visualized using a color
    mesh plot, with optional customization through additional keyword arguments.

    Parameters:
    -----------
    data : numpy.ndarray, pandas.DataFrame
        Data which should be plotted as a heatmap.

    size_cell : float, optional, default=0.20
        The size of each cell in the heatmap, which affects the dimensions of the resulting plot.

    \\**kwargs
        Additional keyword arguments to be passed to `ax.pcolormesh()` for customizing the heatmap appearance.

    Returns
    -------
    matplotlib.figure.Figure
        The generated heatmap figure.

    Raises:
    -------
    ValueError
        If the specified `plot` attribute is not found in `mddc_result`.

    Notes:
    ------
    - The function automatically adjusts the figure size based on the dimensions of the heatmap data and the specified
      `size_cell`.
    - The x and y axis labels are set according to the columns and index of the heatmap data.
    - The x-axis tick labels are rotated 90 degrees for better readability.
    """

    if not isinstance(data, (np.ndarray, pd.DataFrame)):
        raise TypeError("Data must be a numpy array or a pandas DataFrame.")

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    num_rows, num_cols = data.shape
    fig_width = max(8, num_cols * size_cell)
    fig_height = max(6, num_rows * size_cell)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    c = ax.pcolormesh(data, **kwargs)
    fig.colorbar(c, ax=ax)

    ax.set_xticks(np.arange(0.5, len(data.columns), 1))
    ax.set_yticks(np.arange(0.5, len(data.index), 1))

    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.index)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.close(fig)
    return fig


def get_expected_count(contin_table):
    """
    Calculate the expected count matrix for a given contingency table.

    This function takes a contingency table as input and calculates
    the expected count matrix based on the marginal sums of the table.
    The calculation is based on the assumption of independence between
    rows and columns in the contingency table.

    Parameters
    ----------
    contin_table : numpy.ndarray or pandas.DataFrame
        A contingency table representing the observed counts for
        different categories.

    Returns
    -------
    expected counts : numpy.ndarray or pandas.DataFrame
        A matrix of expected counts under the assumption of independence
        between rows and columns.

    Notes
    -----
    The expected counts are calculated using the formula:
        E[i, j] = (row_sum[i] * col_sum[j]) / total_sum
    where `row_sum` is the sum of the counts for each row,
    `col_sum` is the sum of the counts for each column,
    and `total_sum` is the grand total of the table.
    """
    if not (isinstance(contin_table, (np.ndarray, pd.DataFrame))):
        raise TypeError("Inputs must be numpy.ndarray or pandas.DataFrame")

    if isinstance(contin_table, pd.DataFrame):
        row_names = list(contin_table.index)
        column_names = list(contin_table.columns)
        contin_table = contin_table.values

        e_ij_mat = pd.DataFrame(getEijMat(contin_table))
        e_ij_mat.columns = column_names
        e_ij_mat.index = row_names

        return e_ij_mat

    else:
        row_names = range(contin_table.shape[0])
        column_names = range(contin_table.shape[1])

        return getEijMat(contin_table)


def get_std_pearson_res(contin_table):
    """
    Compute the standardized Pearson residuals for a given contingency table.

    This function calculates the standardized Pearson residuals from a
    contingency table, which measures the deviation of observed counts
    from expected counts under the assumption of independence between
    the rows and columns.

    If the input is a pandas DataFrame, the row and column names are
    retained in the resulting DataFrame of standardized Pearson residuals.
    Otherwise, the function returns a numpy array of residuals.

    Parameters
    ----------
    contin_table : numpy.ndarray or pandas.DataFrame
        A contingency table representing the observed counts for different
        categories. The table must be a 2D array-like structure (numpy array
        or pandas DataFrame).

    Returns
    -------
    standardized pearson residuals : numpy.ndarray or pandas.DataFrame
        The standardized Pearson residuals matrix. If the input is a
        pandas DataFrame, the output will also be a pandas DataFrame
        with the same row and column labels. Otherwise, the result
        is a numpy array.

    Raises
    ------
    TypeError
        If the input is not a numpy.ndarray or pandas.DataFrame.

    Notes
    -----
    The standardized Pearson residuals are calculated as:
        Z[i, j] = (O[i, j] - E[i, j]) / sqrt(E[i, j])
    where `O[i, j]` is the observed count and `E[i, j]` is the expected
    count under the independence model.
    """
    if not (isinstance(contin_table, (np.ndarray, pd.DataFrame))):
        raise TypeError("Inputs must be numpy.ndarray or pandas.DataFrame")

    if isinstance(contin_table, pd.DataFrame):
        row_names = list(contin_table.index)
        column_names = list(contin_table.columns)
        contin_table = contin_table.values

        z_ij_mat = pd.DataFrame(getZijMat(contin_table, False)[0])
        z_ij_mat.columns = column_names
        z_ij_mat.index = row_names

        return z_ij_mat

    else:
        row_names = range(contin_table.shape[0])
        column_names = range(contin_table.shape[1])

        return getZijMat(contin_table, False)[0]
