# Causal Fairness through Data Repair: Framework Implementation

## Step 1: Project Setup

### Project Structure
# causal_fairness_framework/
# ├── data/
# │   ├── adult.csv
# │   ├── compas.csv
# ├── src/
# │   ├── preprocess.py
# │   ├── causal_model.py
# │   ├── fairness_repair.py
# │   ├── metrics.py
# │   ├── visualize.py
# ├── notebooks/
# │   ├── exploratory_analysis.ipynb
# │   ├── fairness_demo.ipynb
# ├── tests/
# │   ├── test_fairness_repair.py
# │   ├── test_metrics.py
# ├── requirements.txt
# ├── README.md

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Requirements
# pandas, numpy, networkx, scikit-learn, matplotlib, seaborn, dowhy


## Step 2: Data Loading and Preprocessing

def load_dataset(file_path):
    """Load a CSV dataset."""
    return pd.read_csv(file_path)


def preprocess_dataset(data, target_column, protected_attribute):
    """Preprocess the dataset by encoding categorical variables and splitting data."""
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.3, random_state=42), data


## Step 3: Building the Causal Model
from dowhy import CausalModel

def create_causal_model(data, treatment, outcome):
    """Define the causal graph and initialize the CausalModel."""
    causal_graph = """graph[
        Treatment -> Outcome;
        Treatment -> Confounder;
        Confounder -> Outcome;
    ]"""
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=causal_graph
    )
    return model


## Step 4: Data Repair for Fairness

def repair_data(data, treatment, outcome, admissible, inadmissible):
    """Repair the data by removing effects of inadmissible variables on the outcome."""
    model = create_causal_model(data, treatment, outcome)
    identified_estimand = model.identify_effect(
        proceed_when_unidentifiable=True,
        method_name="backdoor",
        admissible_set=admissible
    )

    repaired_data = model.do("do", variable=outcome, value=None)
    return repaired_data


## Step 5: Metrics for Fairness
from sklearn.metrics import f1_score


def demographic_parity(data, protected_attribute, outcome):
    """Calculate demographic parity."""
    groups = data.groupby(protected_attribute)
    rates = groups[outcome].mean()
    return rates


## Step 6: Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_distributions(data, column, title):
    """Visualize distribution of a feature."""
    sns.histplot(data[column], kde=True)
    plt.title(title)
    plt.show()


## Step 7: Main Script
if __name__ == "__main__":
    # Load data
    file_path = "data/adult.csv"
    target_column = "income"
    protected_attribute = "race"

    # Preprocess dataset
    (X_train, X_test, y_train, y_test), data = preprocess_dataset(
        load_dataset(file_path), target_column, protected_attribute
    )

    # Repair data for fairness
    repaired_data = repair_data(data, treatment="race", outcome="income",
                                admissible=["education", "age"], inadmissible=["race"])

    # Evaluate metrics
    parity = demographic_parity(repaired_data, "race", "income")
    print(f"Demographic Parity: {parity}")

    # Visualize
    visualize_distributions(repaired_data, "income", "Repaired Income Distribution")
