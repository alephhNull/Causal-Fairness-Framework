import numpy as np


def compute_counterfactual_equalized_odds(y_true, y_pred_original, y_pred_cf):
    disparities = []
    for y in [0, 1]:
        # Indices where true label = y
        idx = (y_true == y)
        # Subset predictions and sensitive attribute for this class
        y_pred_orig_sub = y_pred_original[idx]
        y_pred_cf_sub = y_pred_cf[idx]

        # Calculate disparity for this class
        orig_rate = y_pred_orig_sub.mean()
        cf_rate = y_pred_cf_sub.mean()
        print(orig_rate, cf_rate)
        disparity = abs(orig_rate - cf_rate)
        disparities.append(disparity)

    # CEO difference is the maximum disparity across classes
    ceo_diff = max(disparities)
    print(f"Counterfactual Equalized Odds Difference: {ceo_diff:.4f}")


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
