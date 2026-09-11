import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model_config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    FROM_TEST_PATH,
    FROM_TRAIN_PATH,
    MAX_MONTHLY_SALARY,
    METADATA_PATH,
    METRICS_PATH,
    MIN_MONTHLY_SALARY,
    MODEL_FROM_PATH,
    MODEL_SOURCE_PATH,
    MODEL_TO_PATH,
    RANDOM_STATE,
    SAMPLE_PREDICTIONS_PATH,
    TO_TEST_PATH,
    TO_TRAIN_PATH,
    ensure_output_dirs,
)


def load_dataset(path):
    df = pd.read_csv(path)
    for column in BINARY_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Не указано").astype(str)
    return df


def build_model():
    # Категориальные признаки кодируются, бинарные признаки передаются как 0/1.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=4,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def filter_target_range(df, target):
    values = pd.to_numeric(df[target], errors="coerce")
    mask = values.between(MIN_MONTHLY_SALARY, MAX_MONTHLY_SALARY)
    return df[mask].copy()


def evaluate(model, df, target, dataset_name):
    x = df[FEATURES]
    y_true = pd.to_numeric(df[target], errors="coerce")
    y_pred = model.predict(x)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "target": target,
        "dataset": dataset_name,
        "rows": len(df),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
    }


def print_metrics(metrics):
    print("\nМетрики качества модели")
    print("-" * 72)
    for row in metrics:
        print(
            f"{row['target']} | {row['dataset']} | rows={row['rows']} | "
            f"MAE={row['MAE']:,.0f} руб. | RMSE={row['RMSE']:,.0f} руб. | "
            f"R2={row['R2']:.4f}"
        )
    print("-" * 72)


def save_sample_predictions(model_from, model_to, from_test, to_test):
    # Берем несколько вакансий, где известны обе границы, чтобы показать примеры прогноза.
    both = pd.concat([from_test, to_test], ignore_index=True)
    both = both.drop_duplicates(subset=["vacancy_id"])
    both = both[both["salary_from"].notna() & both["salary_to"].notna()].head(10).copy()
    if both.empty:
        return

    pred_from = model_from.predict(both[FEATURES])
    pred_to = model_to.predict(both[FEATURES])
    both["pred_from"] = np.minimum(pred_from, pred_to).round(0)
    both["pred_to"] = np.maximum(pred_from, pred_to).round(0)

    columns = [
        "vacancy_id",
        "name",
        "primary_role",
        "city_group",
        "experience_name",
        "schedule_name",
        "salary_from",
        "salary_to",
        "pred_from",
        "pred_to",
        "alternate_url",
    ]
    both[columns].to_csv(SAMPLE_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")


def main():
    ensure_output_dirs()

    from_train = load_dataset(FROM_TRAIN_PATH)
    from_test = load_dataset(FROM_TEST_PATH)
    to_train = load_dataset(TO_TRAIN_PATH)
    to_test = load_dataset(TO_TEST_PATH)

    # Убираем очевидно немесячные или ошибочные зарплаты, чтобы они не сбивали модель.
    from_train = filter_target_range(from_train, "salary_from")
    from_test = filter_target_range(from_test, "salary_from")
    to_train = filter_target_range(to_train, "salary_to")
    to_test = filter_target_range(to_test, "salary_to")

    model_from = build_model()
    model_to = build_model()

    # Один алгоритм обучается отдельно для нижней и верхней границы зарплаты.
    model_from.fit(from_train[FEATURES], pd.to_numeric(from_train["salary_from"]))
    model_to.fit(to_train[FEATURES], pd.to_numeric(to_train["salary_to"]))

    metrics = [
        evaluate(model_from, from_train, "salary_from", "train"),
        evaluate(model_from, from_test, "salary_from", "test"),
        evaluate(model_to, to_train, "salary_to", "train"),
        evaluate(model_to, to_test, "salary_to", "test"),
    ]

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")

    joblib.dump(model_from, MODEL_FROM_PATH)
    joblib.dump(model_to, MODEL_TO_PATH)
    save_sample_predictions(model_from, model_to, from_test, to_test)

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "RandomForestRegressor",
        "source_csv": "data/processed/vacancies_for_model_final.csv",
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "binary_features": BINARY_FEATURES,
        "min_monthly_salary": MIN_MONTHLY_SALARY,
        "max_monthly_salary": MAX_MONTHLY_SALARY,
        "random_state": RANDOM_STATE,
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 4,
        "model_from_path": "data/model/models/model_from.joblib",
        "model_to_path": "data/model/models/model_to.joblib",
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print_metrics(metrics)
    print(f"Метрики сохранены: {METRICS_PATH}")
    print(f"Модель salary_from сохранена: {MODEL_FROM_PATH}")
    print(f"Модель salary_to сохранена: {MODEL_TO_PATH}")
    print(f"Параметры модели сохранены: {METADATA_PATH}")
    if SAMPLE_PREDICTIONS_PATH.exists():
        print(f"Примеры прогнозов сохранены: {SAMPLE_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
