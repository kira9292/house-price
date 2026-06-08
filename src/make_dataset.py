""" Data generator for modeling """

import pandas as pd
from sklearn.datasets import fetch_openml
from loguru import logger

def load_data(dataset_name: str, columns_to_lower: bool = False) -> pd.DataFrame:
    """

    Args:
        dataset_name:
        columns_to_lower:

    Returns:
        pd.DataFrame:

    """
    if (dataset_name is None) or (dataset_name == ""):
        raise ValueError("Dataset name cannot be empty")
    
    data_house = fetch_openml(name='house_prices', return_X_y=False, target_column=None)
    data = data_house['data']

    if columns_to_lower and dataset_name == "house_prices":
        data.columns = data.columns.str.lower()
        data = data.assign(building_age=lambda dfr: dfr.yrsold.astype(float) - dfr.yearbuilt.astype(float),
                           remodel_age=lambda dfr: dfr.yrsold.astype(float) - dfr.yearremodadd.astype(float),
                           garage_age=lambda dfr: dfr.yrsold.astype(float) - dfr.garageyrblt.astype(float)
                           )

    logger.info(f"Loaded {dataset_name} dataset")
    logger.info(f"dataset description : {data_house['DESCR']}")
    logger.info(f"Data Shape: {data.shape}")
    return data
