"""
analysis.py
===============

Usage:
    python analysis.py

Dependencies:
    - pandas
    - scikit‑learn
    - numpy
    - matplotlib

Outputs:
    - decision_tree_results.csv: tabular metrics for each dataset variant
    - decision_tree_f1_chart.png: bar chart of F1 scores
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


def load_data(path: str) -> pd.DataFrame:
    """Load the water potability dataset from CSV file."""
    return pd.read_csv(path)


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values using the mean of each column."""
    df_imputed = df.copy()
    for col in df_imputed.columns:
        if col == 'Potability':
            continue
        if df_imputed[col].isnull().any():
            df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
    return df_imputed


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features (pH binning and composite interaction terms)"""
    engineered = df.copy()
    # pH binning: 0 = acidic, 1 = neutral, 2 = basic
    engineered['pH_bin'] = pd.cut(
        engineered['ph'],
        bins=[-np.inf, 6.5, 8.5, np.inf],
        labels=[0, 1, 2]
    )
    # Composite features
    engineered['solids_sulfate'] = engineered['Solids'] * engineered['Sulfate']
    # Avoid division by zero by adding a small epsilon
    engineered['sulfate_hardness'] = engineered['Sulfate'] / (engineered['Hardness'] + 1e-5)
    engineered['solids_orgcarb'] = engineered['Solids'] / (engineered['Organic_carbon'] + 1e-5)
    return engineered


def select_features_by_correlation(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Select features whose absolute correlation with the target is at least median."""
    corr = df.corr()[target].abs()
    threshold = corr.median()
    selected = corr[corr >= threshold].index.tolist()
    if target in selected:
        selected.remove(target)
    return df[selected]


def evaluate_decision_tree(X: pd.DataFrame, y: pd.Series, dataset_name: str) -> dict:
    """Train and evaluate a decision tree on the given features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return {
        'Dataset': dataset_name,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }


def main():
    data_path = os.path.join('data', 'water_potability.csv')
    results_csv = 'decision_tree_results.csv'
    chart_png = 'decision_tree_f1_chart.png'

    df_raw = load_data(data_path)
    y = df_raw['Potability']
    # Impute missing values
    df_imputed = impute_missing(df_raw)

    results = []

    # Dataset 1: converted (imputed raw features)
    X1 = df_imputed.drop('Potability', axis=1)
    results.append(evaluate_decision_tree(X1, y, 'converted'))

    # Dataset 2: normalized (Min Max scaling)
    scaler = MinMaxScaler()
    X_norm = df_imputed.drop('Potability', axis=1).copy()
    X_norm = pd.DataFrame(
        scaler.fit_transform(X_norm), columns=X_norm.columns
    )
    results.append(evaluate_decision_tree(X_norm, y, 'normalized'))

    # Dataset 3: features (engineered features added)
    df_feat = add_feature_engineering(df_imputed)
    df_feat_imputed = impute_missing(df_feat)
    X3 = df_feat_imputed.drop('Potability', axis=1)
    results.append(evaluate_decision_tree(X3, y, 'features'))

    # Dataset 4: selected_converted (feature selection on imputed raw)
    df_selected_conv = df_imputed.copy()
    X4 = select_features_by_correlation(df_selected_conv, 'Potability')
    results.append(evaluate_decision_tree(X4, y, 'selected_converted'))

    # Dataset 5: selected_features (feature selection on engineered dataset)
    df_selected_feat = df_feat_imputed.copy()
    X5 = select_features_by_correlation(df_selected_feat, 'Potability')
    results.append(evaluate_decision_tree(X5, y, 'selected_features'))


    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_csv, index=False)

    # Create bar chart for F1 score
    plt.figure()
    plt.bar(results_df['Dataset'], results_df['F1 Score'])
    plt.title('F1 Score by Dataset Variant (Decision Tree)')
    plt.xlabel('Dataset Variant')
    plt.ylabel('F1 Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(chart_png)

    print('Results saved to', results_csv)
    print('Chart saved to', chart_png)


if __name__ == '__main__':
    main()
