import pandas as pd
import numpy as np
from importlib import resources


def load_wireless_data(desc=False, as_dataframe=True):
    data = pd.read_csv(
        str(
            resources.files("MDDC.datasets").joinpath("data/statin49.csv")
        )
    )
    
    data.index = data[data.columns[0]].tolist()
    data.drop(columns = data.columns[0], inplace = True)
    
    print(data)
    return None


load_wireless_data()