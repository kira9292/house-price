"""Settings"""
from os import environ
from pathlib import Path

# Home directory

if "COLAB_JUPYTER_IP" in environ:  # colab running
  HOME_DIR = Path("drive", "MyDrive", "house_price")
else:  # localhost
  HOME_DIR = Path.cwd().parent

# data
DATA_DIR = Path(HOME_DIR, "data")
DATA_DIR_INPUT = Path(DATA_DIR, "input")
DATA_DIR_OUTPUT = Path(DATA_DIR, "output")

# models
MODEL_DIR = Path(HOME_DIR, "models")
MODEL_NAME = "model_house_price.dill"  # add on prefix the execution date (YYYYMMDD_{MODEL_NAME})

# reports: graphs, html, ...
REPORT_DIR = Path(HOME_DIR, "reports")

# Source de code
SRC_DIR = Path(HOME_DIR, "src")

# Notebook
NOTEBOOK_DIR = Path(HOME_DIR, "notebooks")

TIMEZONE = "UTC"

MODEL_PARAMS = {
    "TEST_SIZE": 0.25,
    "MIN_COMPLETION_RATE": 0.5,  # min completion rate
    "TARGET": "saleprice",
    "MIN_PPS": 0.10,  # Minimal value for Predictive Power Score (PPS)
    "FEATURES": ['bsmtfinsf1',  # Type 1 finished square feet
                 'bsmtunfsf',  # Type 2 finished square feet
                 'condition2',  # Proximity to various conditions (if more than one is present)
                 'exterqual',  # Evaluates the quality of the material on the exterior
                 'foundation',  # Type of foundation
                 'garagecars',  # Size of garage in car capacity
                 'garagetype',  # Garage location
                 'heating',  # Type of heating (chauffage)
                 'heatingqc',  # Heating quality and condition
                 'housestyle',  # Style of dwelling (type de maison)
                 'lotarea',  # Lot size in square feet
                 'masvnrarea',  # Masonry veneer area in square feet
                 'masvnrtype',  # Masonry veneer type
                 'miscfeature',  # Miscellaneous feature not covered in other categories
                 'mssubclass',  # Identifies the type of dwelling involved in the sale.
                 'overallqual',  # Rates the overall material and finish of the house
                 "saletype",  # Type of sale
                 'street',  # Type of road access to property
                 'totalbsmtsf',  # Total square feet of basement area
                 "building_age",  # Building age: yrsold - yearbuilt
                 "remodel_age",  # Remodel age (same as building_age if no remodeling): yrsold - yearremodadd
                 ],

}

# random state
SEED = 43