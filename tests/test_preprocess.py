# tests/test_preprocess.py
import pytest
import pandas as pd
from src.preprocess import load_dataset, preprocess_dataset

def test_load_dataset():
    # Test loading a valid dataset
    data = load_dataset("data/compas-scores-two-years.csv")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

    # Test loading a non-existent dataset
    with pytest.raises(FileNotFoundError):
        load_dataset("data/non_existent.csv")

def test_preprocess_dataset():
    data = pd.DataFrame({
        'age': [25, 30, 22],
        'age_cat': ['Young', 'Adult', 'Young'],
        'race': ['White', 'Black', 'Asian'],
        'sex': ['Male', 'Female', 'Male'],
        'juv_fel_count': [0, 1, 0],
        'juv_misd_count': [0, 0, 1],
        'juv_other_count': [0, 0, 0],
        'priors_count': [0, 2, 1],
        'decile_score': [5, 7, 3],
        'c_charge_degree': ['M', 'F', 'M'],
        'is_recid': [0, 1, 0],
        'is_violent_recid': [0, 0, 0],
        'two_year_recid': [0, 1, 0],
    })
    relevant_columns = [
        'age',
        'age_cat',
        'race',
        'sex',
        'juv_fel_count',
        'juv_misd_count',
        'juv_other_count',
        'priors_count',
        'decile_score',
        'c_charge_degree',
        'is_recid',
        'is_violent_recid',
        'two_year_recid',
    ]
    target_column = 'two_year_recid'
    protected_attribute = 'race'

    (X_train, X_test, y_train, y_test), processed_data = preprocess_dataset(
        data,
        relevant_cols=relevant_columns,
        target_column=target_column,
        protected_attribute=protected_attribute
    )

    assert not X_train.empty
    assert not X_test.empty
    assert 'race_Black' in processed_data.columns
    assert 'race_White' in processed_data.columns
    assert 'race_Asian' not in processed_data.columns  # Because drop_first=True
