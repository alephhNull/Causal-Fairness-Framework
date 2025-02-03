import numpy as np

def compute_fairness_metrics(X_test, y_test, title="Fairness Analysis"):
    mean_income_female = y_test[X_test["sex_Male"] == 0].mean()
    mean_income_male = y_test[X_test["sex_Male"] == 1].mean()
    dp_diff = mean_income_male - mean_income_female
    corr = np.corrcoef(X_test["sex_Male"], y_test)[0, 1]

    print(f"\n=== {title} ===")
    print(f"Mean Income for Females: {mean_income_female:.4f}")
    print(f"Mean Income for Males: {mean_income_male:.4f}")
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Correlation (Sensitive vs Income): {corr:.4f}")
