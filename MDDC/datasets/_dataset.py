from importlib import resources

import pandas as pd


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


def load_statin49_cluster_idx_data(desc=False, as_dataframe=True):
    """
    A 50 by 7 data matrix of a contingency table processed from FDA Adverse Event Reporting System (FAERS)
    database from the third quarter of 2014 to the fourth quarter of 2020.

    The 49 rows represent 49 important adverse events associated with the statin class, with the final row aggregating the remaining 5,990 adverse events.
    The 49 AEs are classified into three clusters:

    1. AEs associated with signs and symptoms of muscle injury,
    2. AEs associated with laboratory tests for muscle injury, and
    3. AEs associated with kidney injury and its laboratory diagnosis and treatment.

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
    >>> from MDDC.datasets import load_statin49_cluster_idx_data
    >>> statin49_with_cluster_idx = load_statin49_cluster_idx_data()
    """

    if desc:
        desc_file = resources.files("MDDC.datasets").joinpath(
            "data/statin49_with_cluster_idx.rst"
        )
        fdescr = desc_file.read_text()

    data = pd.read_csv(
        str(
            resources.files("MDDC.datasets").joinpath(
                "data/statin49_with_cluster_idx.csv"
            )
        )
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

def load_statin101_data(desc=False, as_dataframe=True):
    """
    A 102 by 5 data matrix of a contingency table processed from FDA Adverse Event Reporting System (FAERS)
    database from the first quarter of 2021 to the fourth quarter of 2023.

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
        If `as_dataframe` is True, a tuple containing the description and a pandas.DataFrame with shape (102, 5) is returned.
        If `as_dataframe` is False, a numpy.ndarray with shape (102, 5) is returned along with the description.

    Data Source
    ------------
    
    The quarterly files can be found at https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html . 


    Examples
    --------
    >>> from MDDC.datasets import load_statin101_data
    >>> statin49 = load_statin101_data()
    """

    if desc:
        desc_file = resources.files("MDDC.datasets").joinpath("data/statin101.rst")
        fdescr = desc_file.read_text()

    data = pd.read_csv(
        str(resources.files("MDDC.datasets").joinpath("data/statin101.csv"))
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
        
        
def load_betablocker500_data(desc=False, as_dataframe=True):
    """
    A 501 by 9 data matrix of a contingency table processed from FDA Adverse Event Reporting System (FAERS)
    database from the first quarter of 2021 to the fourth quarter of 2023.

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
        If `as_dataframe` is True, a tuple containing the description and a pandas.DataFrame with shape (501, 9) is returned.
        If `as_dataframe` is False, a numpy.ndarray with shape (501, 9) is returned along with the description.

    Data Source
    ------------
    
    The quarterly files can be found at https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html . 


    Examples
    --------
    >>> from MDDC.datasets import load_betablocker500_data
    >>> statin49 = load_betablocker500_data()
    """

    if desc:
        desc_file = resources.files("MDDC.datasets").joinpath("data/betablocker500.rst")
        fdescr = desc_file.read_text()

    data = pd.read_csv(
        str(resources.files("MDDC.datasets").joinpath("data/betablocker500.csv"))
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
        
def load_sedative1000_data(desc=False, as_dataframe=True):
    """
    A 1001 by 11 data matrix of a contingency table processed from FDA Adverse Event Reporting System (FAERS)
    database from the first quarter of 2021 to the fourth quarter of 2023.

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
        If `as_dataframe` is True, a tuple containing the description and a pandas.DataFrame with shape (1001, 11) is returned.
        If `as_dataframe` is False, a numpy.ndarray with shape (1001, 11) is returned along with the description.

    Data Source
    ------------
    
    The quarterly files can be found at https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html . 


    Examples
    --------
    >>> from MDDC.datasets import load_sedative1000_data
    >>> statin49 = load_sedative1000_data()
    """

    if desc:
        desc_file = resources.files("MDDC.datasets").joinpath("data/sedative1000.rst")
        fdescr = desc_file.read_text()

    data = pd.read_csv(
        str(resources.files("MDDC.datasets").joinpath("data/sedative1000.csv"))
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