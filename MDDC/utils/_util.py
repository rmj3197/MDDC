"""
`utils.py` contains additional utility functions made available to a user. 
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mddc_cpp_helper import getEijMat, getZijMat


def _generate_simulated_zijmat(
    contin_table,
    signal_mat,
    cluster_idx,
    count_dict,
    rho,
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

    Parameters
    ----------
    contin_table : numpy.ndarray
        A data matrix representing the original contingency table. This matrix provides the dimensions and structure
        for the simulated output.

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

    Returns
    -------
    simulated contingency table : numpy.ndarray
        The simulated contingency table matrix. The values are rounded and any negative values are set to 0.
    """

    if seed is not None:
        generator = np.random.RandomState(seed * rep)
    else:
        generator = np.random.RandomState(None)

    z_ij_mat = np.zeros(contin_table.shape)

    for group in count_dict.keys():
        if count_dict[group] > 1:
            cov = np.full((count_dict[group], count_dict[group]), rho)
            np.fill_diagonal(cov, 1)
            z_ij_mat[np.where(cluster_idx == group), :] = generator.multivariate_normal(
                mean=np.zeros(count_dict[group]), cov=cov, size=contin_table.shape[1]
            ).T
        else:
            z_ij_mat[np.where(cluster_idx == group), :] = generator.randn(
                contin_table.shape[1]
            )
    new_contin_table = np.round(
        z_ij_mat * np.sqrt(e_ij_mat * signal_mat * ((1 - p_i_dot) @ (1 - p_dot_j)))
        + e_ij_mat * signal_mat
    )
    new_contin_table[new_contin_table < 0] = 0
    return new_contin_table


def generate_contin_table_with_clustered_AE(
    contin_table, signal_mat, cluster_idx, n=100, rho=0.5, n_jobs=-1, seed=None
):
    """
    Generate simulated contingency tables with optional incorporation of adverse event correlation within clusters.

    This function generates multiple simulated contingency tables based on the input data matrix (`contin_table`),
    signal strength matrix (`signal_mat`), and cluster indices (`cluster_idx`). It incorporates adverse event
    correlation within each cluster according to the specified correlation parameter (`rho`).

    Parameters
    ----------
    contin_table : numpy.ndarray, pandas.DataFrame
        A data matrix representing an I x J contingency table with row (adverse event) and column (drug) names.
        The row and column marginals of this table are used to generate the simulated data. It is advisable to
        check the input contingency table using the function `check_and_fix_contin_table()` before using this function.

    signal_mat : numpy.ndarray, pandas.DataFrame
        A data matrix of the same dimensions as `contin_table`, where entries represent signal strength. Values
        should be greater than or equal to 1, where 1 indicates no signal and values greater than 1 indicate the
        presence of a signal.

    cluster_idx : numpy.ndarray, list, pd.DataFrame
        An array indicating the cluster index for each row in the `contin_table`. Clusters can be represented by
        names or numerical indices.

    n : int, optional, default=100
        The number of simulated contingency tables to generate.

    rho : float, optional, default=0.5
        The correlation coefficient for adverse events within each cluster. A value between 0 and 1 indicates the
        strength of correlation, with 0 meaning no correlation and 1 meaning perfect correlation.

    n_jobs : int, optional, default=-1
        n_jobs specifies the maximum number of concurrently
        running workers. If 1 is given, no joblib parallelism
        is used at all, which is useful for debugging. For more
        information on joblib `n_jobs` refer to -
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html.

    seed : int, optional, default=None
        Random seed for reproducibility of the simulation.

    Returns
    -------
    simulated tables : list of numpy.ndarray
        A list containing the simulated contingency tables.
    """
    if not isinstance(contin_table, (pd.DataFrame, np.ndarray)):
        raise TypeError("contin_table must be a pandas DataFrame or numpy array.")

    if not isinstance(cluster_idx, (list, np.ndarray, pd.DataFrame)):
        raise TypeError("cluster_idx must be a list or numpy array.")

    if isinstance(cluster_idx, pd.DataFrame):
        if (cluster_idx.shape[1]) or (cluster_idx.shape[1]):
            cluster_idx = cluster_idx.values.flatten()
        else:
            raise ValueError(
                "The pandas DataFrame is not of the correct format. Expected a dataframe with dimensions (n,1) or (1,n)"
            )

    is_dataframe = False
    if isinstance(contin_table, pd.DataFrame):
        is_dataframe = True
        row_names = list(contin_table.index)
        column_names = list(contin_table.columns)
        contin_table = contin_table.values

    e_ij_mat = getEijMat(contin_table)
    n_i_dot = e_ij_mat.sum(axis=1, keepdims=True)
    n_dot_j = e_ij_mat.sum(axis=0, keepdims=True)
    n_dot_dot = e_ij_mat.sum()

    p_i_dot = n_i_dot / n_dot_dot
    p_dot_j = n_dot_j / n_dot_dot

    groups, group_counts = np.unique(cluster_idx, return_counts=True)
    count_dict = dict(zip(groups, group_counts))

    simulated_samples = Parallel(n_jobs=n_jobs)(
        delayed(_generate_simulated_zijmat)(
            contin_table,
            signal_mat,
            cluster_idx,
            count_dict,
            rho,
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


def report_drug_AE_pairs(contin_table, contin_table_signal):
    """
    Report potential adverse events for drugs based on the contingency table.

    This function analyzes the provided contingency table and signal matrix to identify potential adverse events
    associated with each drug. It computes the observed counts, expected counts, and standardized Pearson residuals
    for each (drug, adverse event) pair.

    Parameters
    ----------
    contin_table : numpy.ndarray, pandas..DataFrame
        A data matrix representing an I x J contingency table with rows corresponding to adverse events and columns
        corresponding to drugs. The row and column names of this matrix are used in the analysis. It is advisable
        to check the input contingency table using the function `check_and_fix_contin_table()` before using this
        function.

    contin_table_signal : numpy.ndarray, pandas.DataFrame
        A data matrix of the same dimensions as `contin_table`, with entries of either 1 (indicating a signal) or
        0 (indicating no signal). This matrix should have the same row and column names as `contin_table` and can
        be obtained using the function `MDDC.MDDC.mddc()`.

    Returns
    -------
    Identified Drug-AE pairs : pandas.DataFrame
        A DataFrame with five columns:
            - `Drug` : str, The name of the drug.
            - `AE` : str, The potential adverse event associated with the drug.
            - `Observed Count` : int, The observed count of the (drug, adverse event) pair.
            - `Expected Count` : float, The expected count of the (drug, adverse event) pair.
            - `Standard Pearson Residual` : float, The value of the standardized Pearson residual for the (drug, adverse event) pair.
    """
    if not (
        isinstance(contin_table, (np.ndarray, pd.DataFrame))
        and isinstance(contin_table_signal, (np.ndarray, pd.DataFrame))
    ):
        raise ValueError("Both inputs must be data matrices.")

    # Check if the dimensions match
    if contin_table.shape != contin_table_signal.shape:
        raise ValueError(
            "The dimensions of contin_table and contin_table_signal must be the same."
        )

    # Check if the row and column names match
    if not np.array_equal(contin_table.index, contin_table_signal.index):
        raise ValueError(
            "The row names of contin_table and contin_table_signal must match."
        )

    if not np.array_equal(contin_table.columns, contin_table_signal.columns):
        raise ValueError(
            "The column names of contin_table and contin_table_signal must match."
        )

    if isinstance(contin_table_signal, pd.DataFrame):
        row_names = list(contin_table_signal.index)
        column_names = list(contin_table_signal.columns)
        contin_table_signal = contin_table_signal.values

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
                "Drug",
                "AE",
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
                "Drug",
                "AE",
                "Observed Count",
                "Expected Count",
                "Standard Pearson Residual",
            ]
        )


def plot_heatmap(mddc_result, plot="signal", size_cell=0.20, **kwargs):
    """
    Plot a heatmap of the specified attribute from the `mddc_result`.

    This function generates a heatmap for a given attribute of `mddc_result`, which should be a named tuple with
    fields that include the attribute specified by the `plot` parameter. The heatmap is visualized using a color
    mesh plot, with optional customization through additional keyword arguments.

    Parameters
    ----------
    mddc_result : namedtuple
        A named tuple object containing various fields. The field specified by the `plot` parameter is used for
        generating the heatmap. Ensure that `mddc_result` contains the specified field.

    plot : str, optional, default="signal"
        The name of the attribute in `mddc_result` to be plotted. This attribute should be a numpy.ndarray or pandas.DataFrame.

    size_cell : float, optional, default=0.20
        The size of each cell in the heatmap, which affects the dimensions of the resulting plot.

    \**kwargs
        Additional keyword arguments to be passed to `ax.pcolormesh()` for customizing the heatmap appearance.

    Returns
    -------
    matplotlib.figure.Figure
        The generated heatmap figure.

    Raises
    ------
    ValueError
        If the specified `plot` attribute is not found in `mddc_result`.

    Notes
    -----
    - The function automatically adjusts the figure size based on the dimensions of the heatmap data and the specified
      `size_cell`.
    - The x and y axis labels are set according to the columns and index of the heatmap data.
    - The x-axis tick labels are rotated 90 degrees for better readability.
    """

    if plot not in mddc_result._fields:
        raise ValueError(
            f"{mddc_result.__class__.__name__} does not contain attribute {plot}. Please check both `mddc_result` and `plot` arguments."
        )
    else:
        plot_obj = mddc_result._asdict()[plot]

    if isinstance(plot_obj, np.ndarray):
        plot_obj = pd.DataFrame(plot_obj)

    num_rows, num_cols = plot_obj.shape
    fig_width = max(8, num_cols * size_cell)
    fig_height = max(6, num_rows * size_cell)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    c = ax.pcolormesh(plot_obj, **kwargs)
    fig.colorbar(c, ax=ax)

    ax.set_xticks(np.arange(0.5, len(plot_obj.columns), 1))
    ax.set_yticks(np.arange(0.5, len(plot_obj.index), 1))

    ax.set_xticklabels(plot_obj.columns)
    ax.set_yticklabels(plot_obj.index)

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.close(fig)
    return fig
