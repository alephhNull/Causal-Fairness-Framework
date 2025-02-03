import pandas as pd
from sklearn.linear_model import LinearRegression


def repair_features(data, protected_columns, features_to_repair):
    """
    Repair the specified features by removing the influence of the protected attributes.

    For each feature in features_to_repair, regress the feature on the protected_columns and replace the feature with the residual.

    Parameters:
    - data: pd.DataFrame, the dataset.
    - protected_columns: list of str, columns corresponding to protected attributes.
    - features_to_repair: list of str, feature columns to repair.

    Returns:
    - repaired_data: pd.DataFrame, the dataset with repaired features.
    """
    repaired_data = data.copy()
    for feature in features_to_repair:
        print(f"Repairing feature '{feature}'...")
        X = repaired_data[protected_columns]
        y = repaired_data[feature]
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        # Replace the feature with its residual (i.e., the part orthogonal to the protected attributes)
        repaired_data[feature] = y - y_pred
        print(f"Feature '{feature}' repaired (residual computed).")
    return repaired_data


if __name__ == "__main__":
    # Example usage:
    import pandas as pd

    df = pd.DataFrame({
        'race_African_American': [1, 0, 1, 0],
        'race_Asian': [0, 1, 0, 1],
        'priors_count': [3, 5, 2, 4],
        'decile_score': [7, 6, 8, 5],
        'two_year_recid': [1, 0, 1, 0]
    })
    protected_columns = ['race_African_American', 'race_Asian']
    features_to_repair = ['priors_count', 'decile_score']
    repaired_df = repair_features(df, protected_columns, features_to_repair)
    print(repaired_df)
