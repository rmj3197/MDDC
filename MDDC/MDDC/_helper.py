import numpy as np
import scipy

# from matplotlib.cbook import boxplot_stats

from mddc_cpp_helper import getZijMat, getFisherExactTestTable


def apply_func(row, n, m):
    return getZijMat(row.reshape(n, m), True)[0]


def max_log_col(matrix):
    return np.nanmax(np.log(matrix), axis=0)


def get_log_bootstrap_cutoff(
    contin_table,
    quantile=0.95,
    rep=3,
    seed=None,
):
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
    tabl = getFisherExactTestTable(contin_table, i, j, exclude_same_drug_class)
    return scipy.stats.fisher_exact(tabl)[1]


def normalize_column(a):
    mean_a = np.nanmean(a)
    sd_a = np.nanstd(a, ddof=1)
    return (a - mean_a) / sd_a


def compute_whishi1(z_ij_mat, contin_table, a):
    return boxplot_stats(z_ij_mat[contin_table[:, a] != 0, a])[3]


def compute_whishi2(vec):
    return boxplot_stats(vec)[3]


def compute_whislo1(z_ij_mat, contin_table, a):
    return boxplot_stats(z_ij_mat[contin_table[:, a] == 0, a])[1]


def compute_whislo2(vec):
    return boxplot_stats(vec)[1]
