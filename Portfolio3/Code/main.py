"""
Vegemite consistency portfolio 3

Steps
- Load vegemite.csv and shuffle (seeded).
- Create a 1,000 row hold-out with class counts 333/333/334 (0/1/2).
- Drop constant columns; add PV SP delta features ("Delta <base>").
- Treat low-cardinality numerics and objects as categorical (one-hot); scale the rest.
- Oversample training split; train multiple classifiers; save confusion matrices + metrics.
- Evaluate all models on the hold-out; persist the best validation model.
- Fit a shallow set-point only tree and export text rules for the report.

Usage:
    python main.py --data vegemite.csv --out ../outputs
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

# LOGGING + CLI STUFF ---
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Vegemite consistency modelling.")
    ap.add_argument("--data", type=Path, default=Path("vegemite.csv"),
                    help="Path to vegemite.csv (default: ./vegemite.csv)")
    ap.add_argument("--out", type=Path, default=Path("../outputs"),
                    help="Output directory (default: ../outputs)")
    return ap.parse_args()


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# DATA LOADING / PREP ---
def load_and_shuffle(path: Path) -> pd.DataFrame:
    """Load CSV and shuffle with a fixed seed for reproducibility."""
    df = pd.read_csv(path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def take_holdout(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Take out 1,000 samples with near-equal class distribution:
    333 (class 0), 333 (class 1), 334 (class 2).
    """
    sample_sizes = {0: 333, 1: 333, 2: 334}
    holdout_parts = []
    for cls, n in sample_sizes.items():
        subset = df[df["Class"] == cls]
        if len(subset) < n:
            raise ValueError(f"Not enough rows for class {cls} to build hold-out.")
        holdout_parts.append(subset.sample(n=n, random_state=42))
    holdout = pd.concat(holdout_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    remaining = df.drop(holdout.index).reset_index(drop=True)
    return holdout, remaining


def remove_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns that have only a single unique value."""
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    return df.drop(columns=constant_cols), constant_cols


def generate_composite_features(df: pd.DataFrame, out_dir: Path | None = None) -> pd.DataFrame:
    """
    Create delta features for SP/PV pairs: Delta <base> = PV - SP.
    SP columns end with 'SP'; PV columns end with 'PV'.
    """
    df_new = df.copy()
    sp_cols = [c for c in df_new.columns if c.endswith("SP")]
    pv_cols = [c for c in df_new.columns if c.endswith("PV")]

    def base_name(col: str, suffix: str) -> str:
        return col[: -len(suffix)].rstrip()

    sp_bases = {base_name(c, "SP"): c for c in sp_cols}
    pv_bases = {base_name(c, "PV"): c for c in pv_cols}

    pairs: List[Tuple[str, str, str]] = []
    for base, sp_col in sp_bases.items():
        if base in pv_bases:
            pv_col = pv_bases[base]
            new_col = f"Delta {base}"
            df_new[new_col] = df_new[pv_col] - df_new[sp_col]
            pairs.append((base, sp_col, pv_col))

    logging.info("Delta features created for %d SP/PV pairs.", len(pairs))
    if out_dir is not None:
        pd.DataFrame(pairs, columns=["Base", "SP_Column", "PV_Column"]).to_csv(
            out_dir / "sp_pv_pairs.csv", index=False
        )
    return df_new


def identify_categorical_columns(df: pd.DataFrame, threshold: int = 10) -> List[str]:
    """
    Numeric columns with < threshold unique values are treated as categorical.
    Object dtype columns are also categorical.
    """
    cat_cols: List[str] = []
    for col in df.columns:
        if col == "Class":
            continue
        series = df[col]
        if series.dtype == object:
            cat_cols.append(col)
        else:
            if series.nunique() < threshold:
                cat_cols.append(col)
    return cat_cols


def prepare_datasets(df: pd.DataFrame, out_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Pipeline:
    - drop constant cols
    - add delta features
    - split X/y
    - identify categorical vs numeric
    """
    df_clean, constant_cols = remove_constant_columns(df)
    if constant_cols:
        logging.info("Dropped constant columns: %s", constant_cols)

    df_features = generate_composite_features(df_clean, out_dir=out_dir)
    y = df_features["Class"]
    X = df_features.drop(columns=["Class"])

    cat_cols = identify_categorical_columns(df_features)
    num_cols = [c for c in X.columns if c not in cat_cols]

    if out_dir is not None:
        pd.Series(cat_cols, name="categorical_columns").to_csv(out_dir / "categorical_columns.csv", index=False)
        pd.Series(num_cols, name="numeric_columns").to_csv(out_dir / "numeric_columns.csv", index=False)

    return X, y, num_cols, cat_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    """Standardise numeric vars and one-hot encode categoricals."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def oversample_training_set(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Simple random oversampling to the size of the largest class.
    """
    df_train = X.copy()
    df_train["Class"] = y.values
    class_counts = df_train["Class"].value_counts().to_dict()
    target_size = max(class_counts.values())

    logging.info("Class counts (pre-oversample): %s", class_counts)

    chunks = []
    for cls, group in df_train.groupby("Class"):
        if len(group) < target_size:
            group_over = resample(
                group,
                replace=True,
                n_samples=target_size,
                random_state=42,  # keep the 42 here as requested
            )
            chunks.append(group_over)
        else:
            chunks.append(group)

    df_bal = pd.concat(chunks).sample(frac=1, random_state=42).reset_index(drop=True)
    y_bal = df_bal["Class"]
    X_bal = df_bal.drop(columns=["Class"])

    logging.info("Balanced to %d per class.", target_size)
    return X_bal, y_bal


# MODELING / EVALUATION ---
def train_and_evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    out_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Fit multiple classifiers and evaluate on validation set.
    Saves confusion matrices and validation_results.csv.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "KNeighbors": KNeighborsClassifier(n_neighbors=7),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, multi_class="auto", class_weight="balanced", solver="lbfgs"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    records = []
    fitted: Dict[str, Pipeline] = {}

    for name, clf in models.items():
        logging.info("Training %s ...", name)
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)

        acc = accuracy_score(y_val, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, preds, average="macro", zero_division=0
        )

        # Save confusion matrix plot with seaborn
        cm = confusion_matrix(y_val, preds)
        plt.figure(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=sorted(y_val.unique()),
            yticklabels=sorted(y_val.unique()),
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"confusion_matrix_{name}.png")
        plt.close()

        records.append(
            {
                "Model": name,
                "Accuracy": acc,
                "MacroPrecision": precision,
                "MacroRecall": recall,
                "MacroF1": f1,
            }
        )
        fitted[name] = pipe

    metrics_df = pd.DataFrame(records).sort_values(by="MacroF1", ascending=False)
    metrics_df.to_csv(out_dir / "validation_results.csv", index=False)
    return metrics_df, fitted


def evaluate_on_holdout(
    models: Dict[str, Pipeline],
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    out_dir: Path,
) -> pd.DataFrame:
    """Evaluate all models on the hold-out and save CSV."""
    rows = []
    for name, model in models.items():
        preds = model.predict(X_holdout)
        acc = accuracy_score(y_holdout, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_holdout, preds, average="macro", zero_division=0
        )
        rows.append(
            {
                "Model": name,
                "Accuracy": acc,
                "MacroPrecision": precision,
                "MacroRecall": recall,
                "MacroF1": f1,
            }
        )
    hold_df = pd.DataFrame(rows).sort_values(by="MacroF1", ascending=False)
    hold_df.to_csv(out_dir / "holdout_results.csv", index=False)
    return hold_df


def train_sp_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series, # df1
    sp_columns: List[str],
    out_dir: Path,
) -> Tuple[DecisionTreeClassifier, str]:
    """
    Train a shallow tree (max_depth=4) using only SP features and export rules.
    """
    X_sp = X_train[sp_columns]
    dt = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=50,
        random_state=42,
        class_weight="balanced",
    )
    dt.fit(X_sp, y_train)
    tree_text = export_text(dt, feature_names=list(X_sp.columns))
    with open(out_dir / "sp_tree_rules.txt", "w") as f:
        f.write(tree_text)
    return dt, tree_text


# MAIN
def save_versions(out_dir: Path) -> None:
    """Write package versions for reproducibility."""
    import sklearn, xgboost
    with open(out_dir / "versions.txt", "w") as f:
        f.write(f"numpy=={np.__version__}\n")
        f.write(f"pandas=={pd.__version__}\n")
        f.write(f"scikit-learn=={sklearn.__version__}\n")
        f.write(f"xgboost=={xgboost.__version__}\n")
        f.write(f"seaborn=={sns.__version__}\n")
        f.write(f"matplotlib=={plt.matplotlib.__version__}\n")


def main() -> None:
    # SEtup
    init_logging()
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    logging.info("Loading data from %s", args.data)
    df = load_and_shuffle(args.data)

    if set(df["Class"].unique()) != {0, 1, 2}:
        raise ValueError("Expected Class column to contain labels {0,1,2}.")
    if len(df) < 15000:
        logging.warning("Dataset smaller than expected (%d rows). Proceeding anyway.", len(df))

    # Hold-out split
    df_holdout, df_remaining = take_holdout(df)
    counts = df_holdout["Class"].value_counts().to_dict()
    logging.info("Hold-out size: %d | class counts: %s", len(df_holdout), counts)

    # remaining data
    X_full, y_full, numeric_cols, cat_cols = prepare_datasets(df_remaining, out_dir=args.out)
    # hold-out data
    X_hold, y_hold, _, _ = prepare_datasets(df_holdout, out_dir=None)

    # Train/val split on remaining
    X_train_raw, X_val_raw, y_train_raw, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        stratify=y_full,
        random_state=42,  # keep 42
    )

    # Balance training via oversampling
    X_train_bal, y_train_bal = oversample_training_set(X_train_raw, y_train_raw)
    logging.info("Training rows (balanced): %d | Validation rows: %d", len(X_train_bal), len(X_val_raw))

    # Train & evaluate on validation
    metrics_df, fitted_models = train_and_evaluate_models(
        X_train_bal,
        y_train_bal,
        X_val_raw,
        y_val,
        numeric_cols,
        cat_cols,
        out_dir=args.out,
    )

    # Save validation bar chart (seaborn retained)
    plt.figure(figsize=(6, 2 + 0.3 * len(metrics_df)))
    sns.barplot(data=metrics_df, y="Model", x="MacroF1")
    plt.xlabel("Macro F1 Score")
    plt.ylabel("Model")
    plt.title("Validation Macro F1 Scores by Model")
    plt.tight_layout()
    plt.savefig(args.out / "model_comparison.png")
    plt.close()

    best_model_name = metrics_df.iloc[0]["Model"]
    logging.info("Best validation model: %s (MacroF1=%.3f)", best_model_name, metrics_df.iloc[0]["MacroF1"])
    joblib.dump(fitted_models[best_model_name], args.out / "best_model.pkl")

    # Evaluate all models on hold-out
    holdout_df = evaluate_on_holdout(fitted_models, X_hold, y_hold, out_dir=args.out)
    logging.info("Top hold-out model: %s (MacroF1=%.3f)", holdout_df.iloc[0]["Model"], holdout_df.iloc[0]["MacroF1"])

    # Set-point only tree
    sp_columns = [c for c in X_full.columns if c.endswith("SP")]
    if not sp_columns:
        logging.warning("No SP columns found for rule extraction.")
    else: 
        _, sp_tree_text = train_sp_decision_tree(X_train_raw, y_train_raw, sp_columns, out_dir=args.out)
        pd.Series(sp_columns, name="sp_columns").to_csv(args.out / "sp_columns.csv", index=False)
        delta_features = [c for c in X_full.columns if c.startswith("Delta ")]
        pd.Series(delta_features, name="delta_features").to_csv(args.out / "delta_features.csv", index=False)
    # if sp_columns:
    #
    
    # Save hold-out predictions from the best model
    holdout_preds = fitted_models[best_model_name].predict(X_hold)
    pd.DataFrame({"Actual": y_hold.values, "Predicted": holdout_preds}).to_csv(
        args.out / "holdout_predictions.csv", index=False
    )

    # Done. Save versions
    save_versions(args.out)

    logging.info("Done. Outputs written to: %s", args.out.resolve())


if __name__ == "__main__":
    main()
