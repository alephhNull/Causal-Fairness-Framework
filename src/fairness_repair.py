from model import predict_nn


def repair_data(model, scaler, original_df, processed_data, feature_cols, outcome):
    y_repaired = predict_nn(model, processed_data, feature_cols, scaler)
    repaired_data = original_df.copy()
    repaired_data[outcome] = y_repaired
    repaired_data.to_csv('../data/repaired.csv', index=False)
    return repaired_data
