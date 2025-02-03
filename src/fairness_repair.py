import pandas as pd
from dowhy import CausalModel


def repair_data(data, inadmissible, outcome, admissible, graph):
    """
    Repair the dataset by removing the effect of inadmissible (protected) attributes on the outcome.

    For each protected attribute, we estimate its effect on the outcome using a regression-based causal estimator.
    We then adjust the outcome by subtracting the product of the estimated effect and the value of the protected attribute.
    """
    repaired_data = data.copy()

    for var in inadmissible:
        print(f"Repairing outcome to remove influence of {var} on {outcome}...")

        model = CausalModel(
            data=data,
            treatment=var,
            outcome=outcome,
            graph=graph
        )
        estimand = model.identify_effect()
        causal_estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
        effect = causal_estimate.value
        print(f"Estimated effect of {var} on {outcome}: {effect}")

        # Adjust the outcome: only for rows where the protected attribute is 1 (since it is one-hot encoded)
        repaired_data[outcome] = repaired_data[outcome] - data[var] * effect

    print("Fairness data repair complete.")
    return repaired_data
