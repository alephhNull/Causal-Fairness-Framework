# Fairness in Machine Learning through Causal Data Repair

## Overview
This project implements a framework for **fairness-aware machine learning** using **causal inference techniques**. It aims to preprocess datasets by removing **unwanted causal effects** of protected attributes (such as gender or race) to ensure **interventional fairness**. The methodology is based on **causal effect estimation** and **adversarial learning** to mitigate bias.

## Project Goals
- **Preprocess biased datasets** and remove discriminatory patterns.
- **Estimate causal effects** of protected attributes on outcomes.
- **Apply adversarial training** to reduce dependence on protected attributes.
- **Compare fairness metrics before and after repair** using causal fairness definitions.
- **Provide visualizations** of bias reduction through data repair.

## Dataset Used
This project demonstrates bias mitigation using the **Adult Census Income Dataset**, which predicts whether an individual's income exceeds $50K per year based on demographic and work-related factors.

## Key Techniques Used
- **Causal Graphs** for understanding relationships in the data.
- **Structural Causal Models (SCM)** to estimate interventional fairness.
- **Adversarial Neural Networks** to debias machine learning models.
- **Counterfactual Analysis** to repair dataset features before training.
- **Fairness Metrics** such as demographic parity and equalized odds.

## Repository Structure
```
├── data/                # Dataset folder
│   ├── adult.csv        # Raw dataset
│
├── src/                 # Source code
│   ├── preprocess.py    # Data preprocessing & feature encoding
│   ├── model.py         # Neural network model with adversarial training
│   ├── fairness_metrics.py  # Computes fairness metrics before/after repair
│   ├── causal_analysis.py   # Causal effect estimation & graph creation
│   ├── visualization.py     # Visualization utilities for fairness analysis
│   ├── interventional_fairness_repair.py  # Feature repair & counterfactual adjustments
│
├── notebooks/           # Jupyter notebooks for demonstration
│   ├── fairness_demo.ipynb  # Main interactive notebook
│
├── README.md            # Project documentation
```

## Installation
To run this project locally, clone the repository and install dependencies:
```bash
git clone https://github.com/YOUR_USERNAME/Fairness-ML-Causal-Repair.git
cd Fairness-ML-Causal-Repair
pip install -r requirements.txt
```

## Usage
Run the main script to process data, repair features, and evaluate fairness:
```bash
python src/main.py
```
Alternatively, open the **Jupyter Notebook** for an interactive demonstration:
```bash
jupyter notebook notebooks/fairness_demo.ipynb
```
