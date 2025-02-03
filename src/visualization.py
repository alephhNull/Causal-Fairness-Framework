import matplotlib.pyplot as plt

def plot_income_distributions(y_test, y_pred, df_results):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(y_test.astype(int), bins=10, alpha=0.5, label="Original Income (Test)")
    plt.hist(y_pred.astype(int), bins=10, alpha=0.5, label="Predicted Income (NN Test)")
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.title("Original vs Predicted Income Distribution")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(df_results["income_repaired"].astype(int), bins=10, alpha=0.5, label="Repaired Income")
    plt.xlabel("Income")
    plt.ylabel("Frequency")
    plt.title("Repaired Income Distribution")
    plt.legend()

    plt.show()
