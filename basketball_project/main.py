"""
Main script for basketball activity recognition.

This script loads accelerometer and gyroscope data from the basketball dataset,
extracts summary statistics as features, trains several classifiers, and evaluates
their performance using cross-validation.

Usage:
    python main.py

Requirements:
    pandas, numpy, scikit-learn, matplotlib
"""

import os
import re
import argparse
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib

# Use Agg backend for headless environments (no GUI)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Basketball activity classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "basketball_dataset_unzipped", "proyecto"),
        help="Path to the directory containing the raw .txt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "outputs"),
        help="Directory to save results such as confusion matrix plots.",
    )
    return parser.parse_args()


def list_data_files(data_dir: str) -> List[str]:
    """Return a list of data file paths in the dataset directory, excluding metadata files."""
    files = []
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(".txt"):
            continue
        # skip readme and long continuous recordings
        if fname.lower().startswith("readme") or fname.lower().startswith("accelerometer"):
            continue
        files.append(os.path.join(data_dir, fname))
    return sorted(files)


def extract_label(file_name: str) -> str:
    """Extract activity label from file name. Assumes format Subject_ActivityTrial.txt."""
    base = os.path.basename(file_name)
    parts = base.split("_")
    if len(parts) < 2:
        return "unknown"
    activity_part = parts[1]  # e.g. pass3.txt
    match = re.match(r"([A-Za-z]+)", activity_part)
    return match.group(1).lower() if match else "unknown"


def extract_subject(file_name: str) -> str:
    """Extract subject identifier from file name."""
    base = os.path.basename(file_name)
    parts = base.split("_")
    return parts[0]


def load_time_series(file_path: str) -> pd.DataFrame:
    """Load a single time-series file into a DataFrame.

    Skips the first two lines (header and timestamp) and uses comma as delimiter.
    Columns: Time (s), X (m/s2), Y (m/s2), Z (m/s2), R (m/s2), Theta (deg), Phi (deg)
    """
    # Read with pandas, specifying column names
    columns = [
        "Time (s)",
        "X (m/s2)",
        "Y (m/s2)",
        "Z (m/s2)",
        "R (m/s2)",
        "Theta (deg)",
        "Phi (deg)",
    ]
    try:
        df = pd.read_csv(
            file_path,
            skiprows=2,
            names=columns,
            header=None,
            sep=",",
            engine="python",
            skipinitialspace=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}")
    return df


def compute_features(df: pd.DataFrame) -> Dict[str, float]:
    """Compute statistical features from a DataFrame of sensor data.

    Returns a dictionary of feature_name -> value.
    Features: mean, std, min, max, energy for each sensor axis (X, Y, Z, R, Theta, Phi).
    Energy is mean of squared values.
    """
    features = {}
    axes = [
        "X (m/s2)",
        "Y (m/s2)",
        "Z (m/s2)",
        "R (m/s2)",
        "Theta (deg)",
        "Phi (deg)",
    ]
    for axis in axes:
        # Convert column to numeric; coerce errors to NaN and drop them
        col_numeric = pd.to_numeric(df[axis], errors="coerce").dropna()
        # If no valid numeric values, set feature values to zero
        if col_numeric.empty:
            features[f"{axis}_mean"] = 0.0
            features[f"{axis}_std"] = 0.0
            features[f"{axis}_min"] = 0.0
            features[f"{axis}_max"] = 0.0
            features[f"{axis}_energy"] = 0.0
        else:
            features[f"{axis}_mean"] = col_numeric.mean()
            features[f"{axis}_std"] = col_numeric.std(ddof=0)
            features[f"{axis}_min"] = col_numeric.min()
            features[f"{axis}_max"] = col_numeric.max()
            features[f"{axis}_energy"] = (col_numeric ** 2).mean()
    return features


def build_dataset(files: List[str]) -> pd.DataFrame:
    """Load all files and construct a DataFrame with features and labels."""
    records = []
    for fpath in files:
        df = load_time_series(fpath)
        feats = compute_features(df)
        feats["label"] = extract_label(fpath)
        feats["subject"] = extract_subject(fpath)
        records.append(feats)
    return pd.DataFrame(records)


def evaluate_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray, output_dir: str):
    """Train and evaluate different classifiers.

    Uses GroupKFold cross-validation so that samples from the same subject are not in both training and test sets.
    Prints classification report and saves confusion matrix plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    models = {
        "kNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF kernel)": SVC(kernel="rbf", C=1.0, gamma="scale")
    }

    gkf = GroupKFold(n_splits=min(4, len(np.unique(groups))))

    results = {}
    for name, clf in models.items():
        # Create pipeline with scaler
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])

        # Cross-validated predictions
        y_pred = cross_val_predict(pipe, X, y, cv=gkf, groups=groups)

        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, y_pred)
        results[name] = {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "predictions": y_pred,
        }

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"Confusion Matrix: {name}")
        fig.colorbar(im, ax=ax)
        classes = sorted(list(set(y)))
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        # Annotate cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        img_path = os.path.join(output_dir, f"confusion_matrix_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png")
        fig.savefig(img_path)
        plt.close(fig)

    return results


def main():
    args = parse_arguments()
    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    files = list_data_files(data_dir)
    if not files:
        raise RuntimeError(f"No data files found in {data_dir}")
    dataset = build_dataset(files)
    # Prepare data
    feature_cols = [c for c in dataset.columns if c not in ("label", "subject")]
    X = dataset[feature_cols].values
    labels = dataset["label"].values
    subjects = dataset["subject"].values
    # Encode labels to ensure consistent ordering
    le = LabelEncoder()
    y = le.fit_transform(labels)
    # Evaluate models
    results = evaluate_models(X, y, groups=subjects, output_dir=output_dir)
    # Print summary
    for name, res in results.items():
        print("=" * 60)
        print(f"Results for {name}")
        print(f"Accuracy: {res['accuracy']:.4f}")
        # Convert classification report to a nicely formatted table
        report = res["report"]
        for class_label, metrics in report.items():
            if class_label in ("accuracy", "macro avg", "weighted avg"):
                continue
            print(f"Class {class_label}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1-score={metrics['f1-score']:.3f}, support={int(metrics['support'])}")
        print(f"Macro avg F1-score: {report['macro avg']['f1-score']:.3f}")
        print(f"Weighted avg F1-score: {report['weighted avg']['f1-score']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()