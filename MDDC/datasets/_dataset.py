import pandas as pd
import numpy as np
from importlib import resources


def load_statin49_data(desc=False, as_dataframe=True):
    """
    A 50 by 7 data matrix of a contingency table processed from FDA Adverse Event Reporting System (FAERS)
    database from the third quarter of 2014 to the fourth quarter of 2020.

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    desc : boolean, optional
        If set to `True`, the function will return the description along with the data.
        If set to `False`, the description will not be included. Defaults to False.

    as_dataframe : boolean, optional
        Determines whether the function should return the data as a pandas DataFrame (Trues)
        or as a numpy array (False). Defaults to True.

    Returns
    -------
    data : pandas.DataFrame, if `as_dataframe` is True
        Dataframe of the data with shape (n_samples, n_features + class).

    (desc, data) : tuple, if `desc` is True.
        If `as_dataframe` is True, a tuple containing the description and a pandas.DataFrame with shape (50, 7) is returned.
        If `as_dataframe` is False, a numpy.ndarray with shape (50, 7) is returned along with the description.

    Data Source
    ------------
    https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-latest-quarterly-data-files

    Examples
    --------
    >>> from MDDC.datasets import load_statin49_data
    >>> statin49 = load_statin49_data()
    """
    
    if desc:
        desc_file = resources.files("MDDC.datasets").joinpath("data/statin49.rst")
        fdescr = desc_file.read_text()

    data = pd.read_csv(
        str(resources.files("MDDC.datasets").joinpath("data/statin49.csv"))
    )
    data.index = data[data.columns[0]].tolist()
    data.drop(columns=data.columns[0], inplace=True)

    if as_dataframe:
        if desc:
            return (fdescr, data)
        else:
            return data
    else:
        if desc:
            return (fdescr, data.values)
        else:
            return data.values
