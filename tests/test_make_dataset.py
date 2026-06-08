"""Unit tests for src/make_dataset.py"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_fake_openml_result():
    """Build a minimal fake OpenML result for house_prices."""
    df = pd.DataFrame({
        "LotArea": [8450.0, 9600.0],
        "YearBuilt": ["2003", "1976"],
        "YearRemodAdd": ["2003", "1976"],
        "GarageYrBlt": ["2003", "1980"],
        "YrSold": ["2008", "2007"],
        "SalePrice": [208500.0, 181500.0],
    })
    mock_result = MagicMock()
    mock_result.__getitem__ = lambda self, key: {
        "data": df,
        "DESCR": "Ames Housing dataset.",
    }[key]
    return mock_result


@patch("src.make_dataset.fetch_openml")
def test_load_data_returns_dataframe(mock_fetch):
    """load_data should return a DataFrame."""
    mock_fetch.return_value = _make_fake_openml_result()
    from src.make_dataset import load_data

    result = load_data(dataset_name="house_prices")
    assert isinstance(result, pd.DataFrame)


@patch("src.make_dataset.fetch_openml")
def test_load_data_lowercase_columns(mock_fetch):
    """columns_to_lower=True should lowercase all column names."""
    mock_fetch.return_value = _make_fake_openml_result()
    from src.make_dataset import load_data

    result = load_data(dataset_name="house_prices", columns_to_lower=True)
    assert all(col == col.lower() for col in result.columns)


@patch("src.make_dataset.fetch_openml")
def test_load_data_feature_engineering(mock_fetch):
    """columns_to_lower=True should add building_age, remodel_age, garage_age."""
    mock_fetch.return_value = _make_fake_openml_result()
    from src.make_dataset import load_data

    result = load_data(dataset_name="house_prices", columns_to_lower=True)
    assert "building_age" in result.columns
    assert "remodel_age" in result.columns
    assert "garage_age" in result.columns


def test_load_data_raises_on_empty_name():
    """load_data should raise ValueError when dataset_name is empty."""
    from src.make_dataset import load_data

    with pytest.raises(ValueError):
        load_data(dataset_name="")


def test_load_data_raises_on_none_name():
    """load_data should raise ValueError when dataset_name is None."""
    from src.make_dataset import load_data

    with pytest.raises(ValueError):
        load_data(dataset_name=None)
