from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from model import FEATURES, normalize_record


TARGET = "mentally_exhausted"
C_VALUES = (0.000001, 0.001, 1.0)
DEFAULT_DATA = Path(__file__).resolve().with_name("indian_developer_burnout_2026.csv")
DEFAULT_MODEL = Path(__file__).resolve().with_name("burn_out_model.bin")


def read_training_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    data = pd.read_csv(path)
    required_columns = {*FEATURES, TARGET}
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing_columns)}")
    if data.empty:
        raise ValueError(f"{path} contains no training rows")

    labels = data[TARGET]
    if labels.isna().any() or set(labels) != {"No", "Yes"}:
        raise ValueError(f"{TARGET} must contain both 'No' and 'Yes' labels and no missing values")

    records = [
        normalize_record(row)
        for row in data.loc[:, FEATURES].to_dict(orient="records")
    ]
    return records, labels.tolist()


def fit_model(
    records: list[dict[str, Any]],
    targets: list[str],
    c_value: float,
) -> tuple[LogisticRegression, DictVectorizer]:
    vectorizer = DictVectorizer(sparse=False)
    features = vectorizer.fit_transform(records)
    model = LogisticRegression(solver="liblinear", C=c_value, max_iter=1000)
    model.fit(features, targets)
    return model, vectorizer


def select_c(records: list[dict[str, Any]], targets: list[str]) -> float:
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=1)
    mean_scores: dict[float, float] = {}

    for c_value in C_VALUES:
        scores = []
        for train_indices, validation_indices in cross_validation.split(records):
            train_records = [records[index] for index in train_indices]
            validation_records = [records[index] for index in validation_indices]
            train_targets = [targets[index] for index in train_indices]
            validation_targets = [targets[index] for index in validation_indices]

            model, vectorizer = fit_model(train_records, train_targets, c_value)
            yes_index = list(model.classes_).index("Yes")
            probabilities = model.predict_proba(
                vectorizer.transform(validation_records)
            )[:, yes_index]
            scores.append(roc_auc_score(validation_targets, probabilities))

        mean_score = float(np.mean(scores))
        mean_scores[c_value] = mean_score
        std_score = float(np.std(scores))
        print(
            f"C={c_value}: mean ROC AUC={mean_score:.3f}, "
            f"std ROC AUC={std_score:.3f}"
        )

    best_c = max(mean_scores, key=mean_scores.get)
    print(f"Best C: {best_c} (mean ROC AUC={mean_scores[best_c]:.3f})")
    return best_c


def train(data_path: Path, model_path: Path) -> None:
    records, targets = read_training_rows(data_path)
    best_c = select_c(records, targets)
    model, vectorizer = fit_model(records, targets, best_c)

    with model_path.open("wb") as model_file:
        pickle.dump((model, vectorizer), model_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the burnout-risk model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    train(args.data, args.model)
