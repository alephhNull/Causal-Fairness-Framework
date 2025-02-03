import warnings

from sklearn.metrics import accuracy_score
from preprocess import load_dataset, preprocess_dataset
from model import MLPClassifier, predict_nn, train_nn_adversarial
from causal_analysis import estimate_causal_effect, creat_causal_model
from fairness_metrics import compute_fairness_metrics
from visualization import plot_repair_effect


def main():
    # Load and preprocess data
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)

    file_path = '../data/adult.csv'
    relevant_columns = [
        'age', 'workclass', 'fnlwgt', 'education.num', 'marital.status',
        'occupation', 'relationship', 'race', 'sex', 'capital.gain',
        'capital.loss', 'hours.per.week', 'native.country', 'income'
    ]
    target_column = 'income'

    data = load_dataset(file_path)
    data['income'] = (data['income'] == '>50K').astype(int)
    (X_train, X_test, y_train, y_test), processed_data = preprocess_dataset(
        data, relevant_cols=relevant_columns, target_column=target_column
    )

    print(f"Training Data Shape: {X_train.shape}")
    print(f"Testing Data Shape: {X_test.shape}")

    # Fairness Metrics Before Repair
    compute_fairness_metrics(X_test, y_test, title="BEFORE REPAIR")

    # Causal Effect Estimation Before Repair
    causal_graph = """
    digraph {
        sex_Male -> income;
        sex_Male -> education_num;
        education_num -> income;
        hours_per_week -> income;
        sex_Male -> hours_per_week;
        age -> income;
    }
    """

    processed_data_test = processed_data.loc[X_test.index, :].copy()
    causal_model = creat_causal_model(processed_data_test, causal_graph, "sex_Male", "income")
    causal_model.view_model()
    effect_before = estimate_causal_effect(causal_model)
    print(f"\n=== Causal Effect BEFORE Repair ===")
    print(f"Estimated Total Effect of Gender on Income: {effect_before.value:.4f}")

    # Train Neural Network
    feature_cols = ['education_num', 'hours_per_week', 'age']
    s_train = X_train['sex_Male']
    model_nn, scaler = train_nn_adversarial(X_train, y_train, s_train, feature_cols, n_epochs=500, lambda_=10)

    # Evaluate Model
    y_pred = predict_nn(model_nn, X_test, feature_cols, scaler)
    print(f"\nTest Accuracy (NN): {accuracy_score(y_test, y_pred):.4f}")

    # Counterfactual Repair
    # df_results = generate_counterfactuals(X_test, model_nn, scaler, feature_cols, y_test)

    # Fairness Metrics After Repair
    compute_fairness_metrics(X_test, y_pred.squeeze(), title="AFTER REPAIR")

    # Causal Effect Estimation After Repair
    processed_data_test["income"] = y_pred

    causal_model_after = creat_causal_model(processed_data_test, causal_graph, "sex_Male", "income")
    effect_after = estimate_causal_effect(causal_model_after)
    print(f"\n=== Causal Effect AFTER Repair ===")
    print(f"Estimated Total Effect of Gender on Income: {effect_after.value:.4f}")

    plot_repair_effect(X_test, y_test, y_pred.squeeze())


if __name__ == "__main__":
    main()
