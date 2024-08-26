from collections import namedtuple

import numpy as np
import pandas as pd
from mddc_cpp_helper import getEijMat

from ._helper import apply_func, compute_fdr, compute_fdr_all


def find_optimal_coef(
    contin_table,
    rep=1000,
    target_fdr=0.05,
    grid=0.1,
    exclude_small_count=True,
    col_specific_cutoff=True,
    seed=None,
):
    """
    Find Adaptive Boxplot Coefficient `coef` via Grid Search. The algorithm
    can be found at :ref:`Algorithms <optimalc_alg>`

    This function performs a grid search to determine the optimal adaptive
    boxplot coefficient `coef` for each column of a contingency table, ensuring
    that the target false discovery rate (FDR) is met.

    Parameters:
    -----------
    contin_table : numpy.ndarray
        A matrix representing the I x J contingency table.

    rep : int, optional
        The number of simulated tables under the assumption of independence
        between rows and columns. Default is 1000.

    target_fdr : float, optional
        The desired level of false discovery rate (FDR). Default is 0.05.

    grid : float, optional
        The size of the grid added to the default value of `coef = 1.5`
        as suggested by Tukey. Default is 0.1.

    exclude_small_count : bool, optional
        Whether to exclude cells with counts smaller than or equal to five
        when computing boxplot statistics. Default is True.

    col_specific_cutoff : bool, optional
        If `True`, then a single value of the coefficient is returned for the
        entire dataset, else when `False` specific values corresponding to each
        of the columns are returned.

    Returns:
    --------
    OptimalCoef : namedtuple
        A namedtuple with the following elements:

        - 'coef': numpy.ndarray
            A numeric array containing the optimal coefficient `coef`
            for each column of the input contingency table.

        - 'FDR': numpy.ndarray
            A numeric array with the corresponding false discovery rate (FDR)
            for each column.

    Examples:
    ---------
    >>> # Example using a simulated contingency table
    >>> import numpy as np
    >>> contin_table = np.random.randint(0, 100, size=(10, 5))
    >>> find_optimal_coef(contin_table)
    """
    if isinstance(contin_table, pd.DataFrame):
        contin_table = contin_table.values

    n, m = contin_table.shape
    expected_counts = getEijMat(contin_table)
    generator = np.random.RandomState(seed)
    p_mat = expected_counts.flatten() / np.sum(expected_counts)
    sim_tables = generator.multinomial(
        n=np.sum(contin_table, dtype=int), pvals=p_mat, size=rep
    )

    if exclude_small_count is False:
        res_list = np.apply_along_axis(
            lambda row: apply_func(row, n, m, False), 1, sim_tables
        )
        sim_tables = sim_tables.reshape(rep, n, m)
        mask = sim_tables == 0
        res_list[mask] = np.nan
    else:
        res_list = np.apply_along_axis(
            lambda row: apply_func(row, n, m, True), 1, sim_tables
        )

    if col_specific_cutoff:
        c_vec = np.repeat(1.5, m)
        fdr_vec = np.ones(m)

        for i in range(m):
            while fdr_vec[i] > target_fdr:
                c_vec[i] += grid
                fdr_vec[i] = compute_fdr(res_list, c_vec[i], i)

        OptimalCoef = namedtuple("OptimalCoef", ["coef", "FDR"])
        oc = OptimalCoef(c_vec, fdr_vec)

    else:
        c = 1.5
        fdr = 1
        while fdr > target_fdr:
            c += grid
            fdr = compute_fdr_all(res_list, c)

        OptimalCoef = namedtuple("OptimalCoef", ["coef", "FDR"])
        oc = OptimalCoef(c, fdr)

    return oc
