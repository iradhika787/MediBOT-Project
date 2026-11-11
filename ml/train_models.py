"""Train and evaluate supervised disease classifiers for MediBOT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
MIN_CLASS_COUNT = 50
DEFAULT_DATASET_PATH = Path(
    r"C:\Users\radhi\Downloads\Final_Augmented_dataset_Diseases_and_Symptoms.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MediBOT multiclass disease classifiers."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the augmented disease/symptom CSV.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing models/ and reports/.",
    )
    return parser.parse_args()


def load_and_prepare_dataset(
    dataset_path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    header = pd.read_csv(dataset_path, nrows=0)
    if len(header.columns) < 2 or header.columns[0] != "diseases":
        raise ValueError("The CSV must have a 'diseases' target column first.")

    feature_names = list(header.columns[1:])
    dtype_map: dict[str, str] = {"diseases": "string"}
    dtype_map.update({name: "uint8" for name in feature_names})
    frame = pd.read_csv(dataset_path, dtype=dtype_map)

    invalid_features = [
        name
        for name in feature_names
        if not frame[name].isin([0, 1]).all()
    ]
    if invalid_features:
        raise ValueError(
            f"Non-binary values found in symptom columns: {invalid_features[:5]}"
        )
    if frame["diseases"].isna().any() or frame["diseases"].str.strip().eq("").any():
        raise ValueError("The dataset contains missing or blank disease labels.")

    original_rows = len(frame)
    frame = frame.drop_duplicates(ignore_index=True)
    duplicate_rows_removed = original_rows - len(frame)

    conflicting_counts = (
        frame.groupby(feature_names, sort=False, dropna=False)["diseases"]
        .transform("nunique")
    )
    conflicting_mask = conflicting_counts > 1
    conflicting_rows_removed = int(conflicting_mask.sum())
    conflicting_combinations = int(
        frame.loc[conflicting_mask, feature_names].drop_duplicates().shape[0]
    )
    frame = frame.loc[~conflicting_mask].reset_index(drop=True)

    constant_features = [
        name for name in feature_names if frame[name].nunique(dropna=False) <= 1
    ]
    feature_names = [
        name for name in feature_names if name not in constant_features
    ]
    frame = frame.loc[:, ["diseases", *feature_names]]

    class_counts = frame["diseases"].value_counts()
    retained_labels = class_counts[class_counts >= MIN_CLASS_COUNT].index
    frame = frame[frame["diseases"].isin(retained_labels)].reset_index(drop=True)
    final_class_counts = frame["diseases"].value_counts()

    preprocessing_summary = {
        "original_records": original_rows,
        "exact_duplicate_rows_removed": duplicate_rows_removed,
        "conflicting_symptom_combinations": conflicting_combinations,
        "conflicting_records_removed": conflicting_rows_removed,
        "constant_features_removed": len(constant_features),
        "usable_features": len(feature_names),
        "classes_before_support_filter": int(class_counts.size),
        "minimum_class_count": MIN_CLASS_COUNT,
        "final_records": len(frame),
        "final_classes": int(final_class_counts.size),
    }
    return frame, feature_names, preprocessing_summary


def make_models() -> dict[str, Pipeline]:
    return {
        "Multinomial Naive Bayes": Pipeline(
            [("classifier", MultinomialNB())]
        ),
        "Logistic Regression": Pipeline(
            [
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=50,
                        solver="saga",
                        n_jobs=-1,
                        tol=1e-3,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                )
            ]
        ),
    }


def calculate_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_confusion_matrix(
    matrix: np.ndarray, labels: np.ndarray, output_path: Path
) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_path.with_suffix(".csv"), index=True
    )
    figure_size = max(12, min(30, len(labels) / 12))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Selected model test confusion matrix",
        xlabel="Predicted disease",
        ylabel="True disease",
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def train_and_evaluate(
    frame: pd.DataFrame,
    feature_names: list[str],
    models_dir: Path,
    reports_dir: Path,
    preprocessing_summary: dict[str, int],
) -> None:
    X = csr_matrix(frame[feature_names].to_numpy(dtype=np.uint8, copy=False))
    y = frame["diseases"].to_numpy()
    indices = np.arange(len(frame))

    train_validation, test = train_test_split(
        indices,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=(0.15 / 0.85),
        random_state=RANDOM_STATE,
        stratify=y[train_validation],
    )
    print(
        f"Split sizes: train={len(train)}, validation={len(validation)}, "
        f"test={len(test)}",
        flush=True,
    )

    validation_results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    for model_name, model in make_models().items():
        print(f"Training {model_name}...", flush=True)
        model.fit(X[train], y[train])
        predictions = model.predict(X[validation])
        metrics = calculate_metrics(y[validation], predictions)
        validation_results.append({"model": model_name, **metrics})
        fitted_models[model_name] = model
        print(
            f"{model_name} validation macro F1: {metrics['macro_f1']:.4f}",
            flush=True,
        )

    comparison = pd.DataFrame(validation_results).sort_values(
        "macro_f1", ascending=False
    )
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    print(f"Selected model: {best_name}", flush=True)

    selected_model = fitted_models[best_name]
    refit_indices = np.concatenate([train, validation])
    print(f"Refitting {best_name} on train + validation...", flush=True)
    selected_model.fit(X[refit_indices], y[refit_indices])
    test_predictions = selected_model.predict(X[test])
    test_metrics = calculate_metrics(y[test], test_predictions)
    save_json(
        reports_dir / "final_test_metrics.json",
        {
            "model": best_name,
            "random_state": RANDOM_STATE,
            "metrics": test_metrics,
            "test_records": len(test),
            "preprocessing": preprocessing_summary,
        },
    )

    report = classification_report(
        y[test],
        test_predictions,
        labels=np.unique(y),
        target_names=np.unique(y),
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        reports_dir / "final_classification_report.csv"
    )
    labels = np.unique(y)
    matrix = confusion_matrix(y[test], test_predictions, labels=labels)
    save_confusion_matrix(
        matrix, labels, reports_dir / "final_confusion_matrix.png"
    )

    metadata = {
        "model": best_name,
        "random_state": RANDOM_STATE,
        "feature_names": feature_names,
        "classes": labels.tolist(),
        "preprocessing": preprocessing_summary,
    }
    joblib.dump(
        {"model": selected_model, "metadata": metadata},
        models_dir / "best_model.joblib",
    )
    save_json(models_dir / "feature_metadata.json", metadata)
    print("Saved model and evaluation reports.", flush=True)


def main() -> int:
    args = parse_args()
    models_dir = args.output_root / "models"
    reports_dir = args.output_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}", flush=True)
    frame, feature_names, summary = load_and_prepare_dataset(args.dataset)
    print(
        f"Prepared {summary['final_records']} records, "
        f"{summary['final_classes']} classes, "
        f"{len(feature_names)} features.",
        flush=True,
    )
    train_and_evaluate(
        frame, feature_names, models_dir, reports_dir, summary
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
