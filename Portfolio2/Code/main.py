"""
Portfolio 2 ML pipeline for COS40007.

Usage:
```
python main.py \
  --boning_csv /path/to/Boning.csv \
  --slicing_csv /path/to/Slicing.csv \
  --student_digit 0 \
  --output_dir ./outputs
```

The output directory will contain the feature table as
features.csv and a results table results.csv summarising model
performance.
"""
import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# if warnings:

def get_sensor_columns(student_digit: int) -> Tuple[str, str]:
    """Return the names of the two sensors (body positions) to use.
    """
    mapping = {
        0: ("Neck", "Head"),
        1: ("Right Shoulder", "Left Shoulder"),
        2: ("Right Upper Arm", "Left Upper Arm"),
        3: ("Right Forearm", "Left Forearm"),
        4: ("Right Hand", "Left Hand"),
        5: ("Right Upper Leg", "Left Upper Leg"),
        6: ("Right Lower Leg", "Left Lower Leg"),
        7: ("Right Foot", "Left Foot"),
        8: ("Right Toe", "Left Toe"),
        9: ("L5", "T12"),
    }
    if student_digit not in mapping:
        raise ValueError(f"Invalid digit {student_digit}. Must be in 0–9.")
    return mapping[student_digit]


def compute_rms(vals: List[np.ndarray]) -> np.ndarray:
    """Compute root mean square of a list of arrays along axis=0.

    The RMS of n arrays is defined as sqrt(mean(square(arrays), axis=0)).
    """
    squares = np.array([v ** 2 for v in vals])
    return np.sqrt(np.mean(squares, axis=0))


def create_composite_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute composite signals for a given sensor prefix.

    Given a prefix , this function computes six new columns
    using the x/y/z accelerations:

    """
    x = df[f"{prefix} x"].to_numpy()
    y = df[f"{prefix} y"].to_numpy()
    z = df[f"{prefix} z"].to_numpy()
    # RMS combinations
    rms_xy = np.sqrt((x**2 + y**2) / 2)
    rms_yz = np.sqrt((y**2 + z**2) / 2)
    rms_zx = np.sqrt((z**2 + x**2) / 2)
    rms_xyz = np.sqrt((x**2 + y**2 + z**2) / 3)
    # Roll and pitch
    roll = 180.0 * np.arctan2(y, np.sqrt(x**2 + z**2)) / math.pi
    pitch = 180.0 * np.arctan2(x, np.sqrt(y**2 + z**2)) / math.pi
    return pd.DataFrame({
        f"{prefix}_rms_xy": rms_xy,
        f"{prefix}_rms_yz": rms_yz,
        f"{prefix}_rms_zx": rms_zx,
        f"{prefix}_rms_xyz": rms_xyz,
        f"{prefix}_roll": roll,
        f"{prefix}_pitch": pitch,
    })


def load_and_prepare_data(boning_csv: str, slicing_csv: str, student_digit: int) -> pd.DataFrame:
    """Load the boning and slicing data, select relevant columns and build composite features.

    The result is a DataFrame with 20 columns: frame, 6 raw axes
    (3 per sensor), twelve composite columns (6 per sensor) and the
    class label 0 for boning, 1 for slicing.
    """
    sensor1, sensor2 = get_sensor_columns(student_digit)
    # Columns to extract (frame + sensor axes)
    raw_cols = ['Frame',
                f'{sensor1} x', f'{sensor1} y', f'{sensor1} z',
                f'{sensor2} x', f'{sensor2} y', f'{sensor2} z']
    # Load each dataset
    boning = pd.read_csv(boning_csv, usecols=raw_cols, engine='c')
    boning['class'] = 0
    slicing = pd.read_csv(slicing_csv, usecols=raw_cols, engine='c')
    slicing['class'] = 1
    combined = pd.concat([boning, slicing], ignore_index=True)
    # Create composite features for each sensor
    comp1 = create_composite_columns(combined, sensor1)
    comp2 = create_composite_columns(combined, sensor2)
    # Merge composites and select final columns
    final = pd.concat([combined[['Frame',
                                 f'{sensor1} x', f'{sensor1} y', f'{sensor1} z',
                                 f'{sensor2} x', f'{sensor2} y', f'{sensor2} z',
                                 'class']], comp1, comp2], axis=1)
    return final


def compute_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per‑minute features from the dynamic columns.

    The function groups the data in non‑overlapping windows of 60 frames
    (i.e., floor(index/60)) and computes six statistical features for each
    of the 18 dynamic columns: mean, std, min, max, AUC and number of peaks.
    """
    # Identify feature columns (exclude frame and class)
    feature_cols = df.columns.difference(['Frame', 'class'])
    df = df.reset_index(drop=True)
    df['minute_id'] = df.index // 60
    aggregated_rows = []
    for minute_id, group in df.groupby('minute_id'):
        # Skip incomplete minute at the end if necessary
        if len(group) < 60:
            continue
        feat_row: Dict[str, float] = {}
        for col in feature_cols:
            values = group[col].values
            # Mean
            feat_row[f'{col}_mean'] = float(np.mean(values))
            # Standard deviation
            feat_row[f'{col}_std'] = float(np.std(values, ddof=0))
            # Min and max
            feat_row[f'{col}_min'] = float(np.min(values))
            feat_row[f'{col}_max'] = float(np.max(values))
            # Area under curve via trapezoidal rule
            feat_row[f'{col}_auc'] = float(np.trapz(values))
            # Number of peaks: a simple local maxima count
            if len(values) >= 3:
                peaks = np.sum((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))
            else:
                peaks = 0
            feat_row[f'{col}_peaks'] = float(peaks)
        # Class label for this minute (assumes constant label within window)
        feat_row['class'] = int(group['class'].iloc[0])
        aggregated_rows.append(feat_row)
    return pd.DataFrame(aggregated_rows)


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Train and evaluate multiple classifiers and return a summary table.

    Models evaluated:
        1. Baseline SVM (RBF kernel) with standardisation.
        2. SVM with hyperparameter tuning via grid search.
        3. SVM with feature selection (SelectKBest) and tuning.
        4. SVM with PCA (10 components) and tuning.
        5. SGDClassifier (log loss).
        6. RandomForestClassifier.
        7. MLPClassifier.

    Both train/test accuracy and mean 10‑fold cross‑validation accuracy are
    reported.
    """
    results = []
    # Prepare a single train/test split for all models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Baseline SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf'))
    ])
    pipeline.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    
    results.append({
        'Model': 'SVM baseline',
        'Test accuracy': test_acc,
        'CV accuracy': cv_scores.mean(),
    })

    # SVM with hyperparameter tuning
    param_grid = {
        'svc__kernel': ['rbf', 'linear'],
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto'],
    }
    grid = GridSearchCV(Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC())
    ]), param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    test_acc = accuracy_score(y_test, best_model.predict(X_test))
    results.append({
        'Model': 'SVM tuned',
        'Test accuracy': test_acc,
        'CV accuracy': grid.best_score_,
    })

    # SVM with feature selection (SelectKBest) and tuning
    pipeline_fs = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(f_classif, k=min(10, X_train.shape[1]))),
        ('svc', SVC())
    ])
    grid_fs = GridSearchCV(
        pipeline_fs,
        param_grid={
            'svc__kernel': ['rbf', 'linear'],
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto'],
        },
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_fs.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, grid_fs.best_estimator_.predict(X_test))
    results.append({
        'Model': 'SVM tuned + SelectKBest',
        'Test accuracy': test_acc,
        'CV accuracy': grid_fs.best_score_,
    })

    # SVM with PCA (10 components) and tuning
    pipeline_pca = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=min(10, X_train.shape[1]))),
        ('svc', SVC())
    ])
    grid_pca = GridSearchCV(
        pipeline_pca,
        param_grid={
            'svc__kernel': ['rbf', 'linear'],
            'svc__C': [0.1, 1, 10],
            'svc__gamma': ['scale', 'auto'],
        },
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_pca.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, grid_pca.best_estimator_.predict(X_test))
    results.append({
        'Model': 'SVM tuned + PCA',
        'Test accuracy': test_acc,
        'CV accuracy': grid_pca.best_score_,
    })

    # SGDClassifier
    sgd_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('sgd', SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42))
    ])
    sgd_pipe.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, sgd_pipe.predict(X_test))
    cv_scores = cross_val_score(sgd_pipe, X_train, y_train, cv=cv, scoring='accuracy')
    results.append({
        'Model': 'SGD',
        'Test accuracy': test_acc,
        'CV accuracy': cv_scores.mean(),
    })

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    results.append({
        'Model': 'Random Forest',
        'Test accuracy': test_acc,
        'CV accuracy': cv_scores.mean(),
    })

    # MLPClassifier
    mlp_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42))
    ])
    mlp_pipe.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, mlp_pipe.predict(X_test))
    cv_scores = cross_val_score(mlp_pipe, X_train, y_train, cv=cv, scoring='accuracy')
    results.append({
        'Model': 'MLP',
        'Test accuracy': test_acc,
        'CV accuracy': cv_scores.mean(),
    })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Train portfolio 2 ML models.")
    parser.add_argument('--boning_csv', type=str, required=True, help='Path to Boning.csv')
    parser.add_argument('--slicing_csv', type=str, required=True, help='Path to Slicing.csv')
    parser.add_argument('--student_digit', type=int, required=True, help='Last digit of student number (0–9)')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save aggregated data and results')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Load and prepare raw data with composites
    print('Loading and preparing data...')
    prepared = load_and_prepare_data(args.boning_csv, args.slicing_csv, args.student_digit)
    
    # Aggregate features per minute
    print('Computing minute‑level features...')
    feature_df = compute_window_features(prepared)
    
    # Save aggregated data
    features_path = os.path.join(args.output_dir, 'features.csv')
    feature_df.to_csv(features_path, index=False)
    print(f'Saved feature data to {features_path}')
    
    # Training and evaluation
    print('Training models...')
    X = feature_df.drop(columns=['class'])
    y = feature_df['class']
    results_df = evaluate_models(X, y)
    results_path = os.path.join(args.output_dir, 'results.csv')
    results_df.to_csv(results_path, index=False)
    print(f'Saved results to {results_path}')
    print('Summary of results:')
    print(results_df)



if __name__ == '__main__':
    main()