import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dowhy import CausalModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import load_dataset, preprocess_dataset  # Assume these are implemented


# Define Neural Network (MLP) Model for Income Prediction
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # For binary outcome
        )

    def forward(self, x):
        return self.network(x)


def estimate_causal_effect(data, causal_graph, treatment, outcome, method="backdoor.linear_regression"):
    """
    Build a DoWhy causal model using the specified graph and estimate the causal effect.
    """
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=causal_graph
    )
    identified_estimand = model.identify_effect()
    effect = model.estimate_effect(identified_estimand, method_name=method)
    return effect


def compute_fairness_metrics(features_df, target_series, sensitive_attr):
    """Compute group means and correlation between sensitive attribute and target."""
    # For group means, assume target_series is numeric (or binary as 0/1)
    group0 = target_series[features_df[sensitive_attr] == 0].mean()
    group1 = target_series[features_df[sensitive_attr] == 1].mean()
    diff = group1 - group0
    corr = np.corrcoef(features_df[sensitive_attr], target_series)[0, 1]
    return group0, group1, diff, corr


def main():
    # ---------------------------
    # Step 0: Load and Preprocess Data
    # ---------------------------
    file_path = '../data/adult.csv'
    relevant_columns = [
        'age', 'workclass', 'fnlwgt', 'education.num', 'marital.status',
        'occupation', 'relationship', 'race', 'sex', 'capital.gain',
        'capital.loss', 'hours.per.week', 'native.country', 'income'
    ]
    target_column = 'income'

    data = load_dataset(file_path)
    # Convert income to binary (1 if >50K, 0 otherwise)
    data['income'] = (data['income'] == '>50K').astype(int)

    # Assume preprocess_dataset splits the data and returns:
    # (X_train, X_test, y_train, y_test) and the full processed DataFrame.
    (X_train, X_test, y_train, y_test), processed_data = preprocess_dataset(
        data, relevant_cols=relevant_columns, target_column=target_column
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")

    # For fairness evaluation, we need the sensitive attribute.
    # Assume that after preprocessing, the column 'sex_Male' (1 for Male, 0 for Female) exists.

    # ---------------------------
    # Step 1: Estimate Causal Effect (Before Repair)
    # ---------------------------
    # Define the causal graph (as a DOT string)
    causal_graph = """
    digraph {
        sex_Male -> income;
        sex_Male -> education_num;
        education_num -> income;
        hours_per_week -> income;
        age -> income;
    }
    """

    effect_before = estimate_causal_effect(
        data=processed_data,
        causal_graph=causal_graph,
        treatment="sex_Male",
        outcome="income",
        method="backdoor.linear_regression"
    )

    print("\n=== Causal Effect BEFORE Repair ===")
    print(f"Estimated Total Effect of Gender on Income: {effect_before.value:.4f}")

    # Compute demographic parity on the test set (original)
    # Since X_test does not contain the target 'income', we use y_test.
    group0_orig, group1_orig, dp_diff_orig, corr_orig = compute_fairness_metrics(X_test, y_test, 'sex_Male')
    print("\n=== BEFORE REPAIR: Fairness Metrics on Test Set ===")
    print(f"Mean Income for Females: {group0_orig:.4f}")
    print(f"Mean Income for Males: {group1_orig:.4f}")
    print(f"Demographic Parity Difference: {dp_diff_orig:.4f}")
    print(f"Correlation (Sensitive vs Income): {corr_orig:.4f}")

    # ---------------------------
    # Step 2: Train a Neural Network Model for Counterfactual Predictions
    # ---------------------------
    feature_cols = ['education_num', 'hours_per_week', 'age', 'sex_Male']
    target_col = 'income'

    # Split data for neural network training from processed_data.
    # (You can either use X_train, y_train or the processed_data directly.)
    scaler = StandardScaler()
    X_train_nn = scaler.fit_transform(X_train[feature_cols])
    X_test_nn = scaler.transform(X_test[feature_cols])

    # Convert training and test sets to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_nn, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_nn, dtype=torch.float32)

    model_nn = MLPClassifier(input_dim=X_train_nn.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_nn.parameters(), lr=0.01)

    # Train the neural network for a sufficient number of epochs
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model_nn(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/500], Loss: {loss.item():.4f}")

    # Evaluate on test set
    y_pred = model_nn(X_test_tensor).detach().numpy().round()
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy (Neural Network): {accuracy:.4f}")

    # ---------------------------
    # Step 3: Counterfactual Repair using the Neural Network
    # ---------------------------
    # For counterfactual fairness, we flip the sensitive attribute in the test set.
    X_cf_test = X_test.copy()
    X_cf_test['sex_Male'] = 1 - X_cf_test['sex_Male']  # Flip: if originally 0 then 1, if 1 then 0.

    X_cf_test_nn = scaler.transform(X_cf_test[feature_cols])
    X_cf_test_tensor = torch.tensor(X_cf_test_nn, dtype=torch.float32)

    # Predict counterfactual outcomes (i.e., counterfactual income predictions)
    y_cf_pred = model_nn(X_cf_test_tensor).detach().numpy()
    print(y_cf_pred.shape)

    # ---------------------------
    # Step 4: Build a Results DataFrame for the Test Set
    # ---------------------------
    # Since X_test does not include the income column, we create a new DataFrame:
    df_results = X_test.copy()
    df_results["income_original"] = y_test.values  # True outcomes
    df_results["income_repaired"] = y_cf_pred.round()  # Counterfactual outcomes

    # ---------------------------
    # Step 5: Evaluate Causal Effect and Fairness on Repaired Data
    # ---------------------------
    # We now create a new processed DataFrame that uses the repaired income.
    # One way to check the reduction in the gender effect is to run the causal effect estimation on the repaired data.
    # First, add the 'income_repaired' column to the processed_data (for evaluation).
    processed_data_repaired = processed_data.copy()
    processed_data_repaired.loc[X_test.index, "income"] = df_results["income_repaired"].values

    effect_after = estimate_causal_effect(
        data=processed_data_repaired,
        causal_graph=causal_graph,
        treatment="sex_Male",
        outcome="income",
        method="backdoor.linear_regression"
    )

    print("\n=== Causal Effect AFTER Repair ===")
    print(f"Estimated Total Effect of Gender on Income: {effect_after.value:.4f}")

    # Also, compute fairness metrics on the test set using df_results:
    group0_repaired, group1_repaired, dp_diff_repaired, corr_repaired = compute_fairness_metrics(df_results, df_results[
        "income_repaired"], 'sex_Male')
    print("\n=== AFTER REPAIR: Fairness Metrics on Test Set ===")
    print(f"Mean Income for Females (Repaired): {group0_repaired:.4f}")
    print(f"Mean Income for Males (Repaired): {group1_repaired:.4f}")
    print(f"Demographic Parity Difference (Repaired): {dp_diff_repaired:.4f}")
    print(f"Correlation (Sensitive vs Repaired Income): {corr_repaired:.4f}")

    # ---------------------------
    # Step 6: Plot Income Distributions Before and After Repair (Test Set)
    # ---------------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(y_test.astype(int), bins=10, alpha=0.5, label="Original Income (Test)")
    plt.hist(y_pred.astype(int), bins=10, alpha=0.5, label="Predicted Income (Test)")
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.title("Original vs Predicted Income Distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(df_results[df_results['sex_Male'] == 0]['income_repaired'].astype(int), bins=10, alpha=0.5,
             label="Female (Repaired)")
    plt.hist(df_results[df_results['sex_Male'] == 1]['income_repaired'].astype(int), bins=10, alpha=0.5,
             label="Male (Repaired)")
    plt.xlabel("Repaired Income")
    plt.ylabel("Frequency")
    plt.title("Repaired Income Distribution")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
