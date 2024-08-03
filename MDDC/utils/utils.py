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
    if seed is not None:
        generator = np.random.RandomState(seed * rep)
    else:
        generator = np.random.RandomState(None)

    z_ij_mat = np.full_like(contin_table, fill_value=np.nan)

    for group in count_dict.keys():
        if count_dict[group] > 1:
            cov = np.full((count_dict[group], count_dict[group]), rho)
            np.fill_diagonal(cov, 1)
            z_ij_mat[np.where(cluster_idx == group), :] = generator.multivariate_normal(
                mean=np.zeros(count_dict[group]), cov=cov, size=contin_table.shape[0]
            ).T

        else:
            z_ij_mat[np.where(cluster_idx == group), :] = generator.randn(
                contin_table.shape[0]
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
    e_ij_mat = getEijMat(contin_table)
    p_i_dot = e_ij_mat.sum(axis=1, keepdims=True)
    p_dot_j = e_ij_mat.sum(axis=0, keepdims=True)

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
    return simulated_samples


def report_drug_AE_pairs(contin_table, contin_table_signal):
    if not (
        isinstance(contin_table, (np.ndarray, pd.DataFrame))
        and isinstance(contin_table_signal, (np.ndarray, pd.DataFrame))
    ):
        raise ValueError("Both inputs must be data matrices.")

    if isinstance(contin_table, np.ndarray):
        contin_table = pd.DataFrame(contin_table)

    if isinstance(contin_table_signal, np.ndarray):
        contin_table_signal = pd.DataFrame(contin_table_signal)

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

    contin_table_signal = np.where(
        np.isnan(contin_table_signal), 0, contin_table_signal
    )

    mat_expected_count = np.round(getEijMat(contin_table), 4)
    mat_std_res = np.round(getZijMat(contin_table), 4)

    pairs = []

    for j in range(contin_table_signal.shape[1]):
        for i in range(contin_table_signal.shape[0]):
            if contin_table_signal[i, j] == 1 and contin_table[i, j] != 0:
                pairs.append(
                    [
                        list(contin_table_signal.columns)[j],
                        list(contin_table_signal.index)[i],
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
