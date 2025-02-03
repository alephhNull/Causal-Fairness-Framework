import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(file_path):
    """
    Load a CSV dataset.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}.")
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        raise


def preprocess_dataset(data, relevant_cols, target_column, test_size=0.3, random_state=42):
    """
    Preprocess the dataset by selecting relevant columns, encoding categorical variables, and splitting data.
    """
    # Select relevant columns and drop missing values
    processed_data = data[relevant_cols].dropna()
    print("Selected relevant columns and dropped missing values.")

    # One-hot encode categorical variables
    categorical_columns = processed_data.select_dtypes(include=['object']).columns
    processed_data = pd.get_dummies(processed_data, columns=categorical_columns, drop_first=True)
    print("Encoded categorical variables using one-hot encoding.")

    # Replace spaces and hyphens with underscores in column names
    processed_data.columns = processed_data.columns.str.replace('.', '_').str.replace('-', '_')
    print("Replaced spaces and hyphens with underscores in column names.")

    # Split features and target
    X = processed_data.drop(columns=[target_column])
    y = processed_data[target_column]
    print(f"Split data into features (X) and target (y): {target_column}.")

    # (We perform a train/test split for completeness, though for feature repair we use all processed data.)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Data split into train and test sets with test size = {test_size}.")

    return (X_train, X_test, y_train, y_test), processed_data


if __name__ == "__main__":
    file_path = '../data/adult.csv'
    relevant_columns = [
        'age',
        'workclass',
        'fnlwgt',
        'education.num',
        'marital.status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital.gain',
        'capital.loss',
        'hours.per.week',
        'native.country',
        'income'
    ]
    target_column = 'income'

    data = load_dataset(file_path)
    data['income'] = data['income'] == '>50K'
    (_, _, _, _), processed_data = preprocess_dataset(
        data, relevant_cols=relevant_columns, target_column=target_column
    )
    print("\nPreprocessing completed successfully.")
