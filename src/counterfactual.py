from src.model import predict_nn


def generate_counterfactuals(X_test, model, scaler, feature_cols, sensitive_attr):
    """
    Generate counterfactual data by flipping the sensitive attribute
    and adjusting downstream features using causal assumptions.
    """
    X_cf = X_test.copy()
    X_cf[sensitive_attr] = 1 - X_cf[sensitive_attr]

    # Predict on counterfactual data
    y_pred_cf = predict_nn(model, X_cf, feature_cols, scaler)

    # Return counterfactual predictions
    return X_cf, y_pred_cf