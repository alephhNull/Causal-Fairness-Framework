import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_repair_effect(X_test, y_true, y_pred_after, sensitive_col="sex_Male"):
    """
    Plots the average outcome (or predicted positive rate) by sensitive group
    before (using y_true) and after repair (using y_pred_after).

    Args:
        X_test (pd.DataFrame): Test features including the sensitive attribute.
        y_true (array-like): Original true outcomes (before repair).
        y_pred_after (array-like): Repaired outcomes (from the adversarial framework).
        sensitive_col (str): Column name in X_test for the sensitive attribute.
    """
    # Create a DataFrame with the sensitive attribute, true outcomes, and repaired outcomes
    df = X_test.copy()
    df = df[[sensitive_col]].copy()  # keep only the sensitive attribute
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_after"] = pd.Series(y_pred_after).values

    # Compute average outcomes per group
    group_means = df.groupby(sensitive_col)[["y_true", "y_pred_after"]].mean().reset_index()

    # Reshape the DataFrame for plotting
    group_means_melted = group_means.melt(id_vars=sensitive_col,
                                          var_name='Outcome Type',
                                          value_name='Average Outcome')

    # Plot using Seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sensitive_col, y='Average Outcome', hue='Outcome Type', data=group_means_melted)
    plt.title("Average Outcome by Sensitive Group: Before vs. After Repair")
    plt.xlabel(sensitive_col)
    plt.ylabel("Average Outcome (Rate)")
    plt.ylim(0, 1)  # Assuming outcomes are probabilities or binary values
    plt.legend(title="Outcome")
    plt.show()
