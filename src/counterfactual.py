import pandas as pd
import torch

def generate_counterfactuals(X_test, model_nn, scaler, feature_cols, y_test):
    X_cf_test = X_test.copy()
    # X_cf_test["sex_Male"] = 1 - X_cf_test["sex_Male"]

    X_cf_test_scaled = scaler.transform(X_cf_test[feature_cols])
    X_cf_test_tensor = torch.tensor(X_cf_test_scaled, dtype=torch.float32)
    y_cf_pred = model_nn(X_cf_test_tensor).detach().numpy()

    df_results = X_test.copy()
    df_results["income_original"] = y_test.values
    df_results["income_repaired"] = y_cf_pred.round()

    return df_results
