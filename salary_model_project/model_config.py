from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DATA_DIR = ROOT / "data" / "model"
TRAIN_TEST_DIR = MODEL_DATA_DIR / "train_test"
METRICS_DIR = MODEL_DATA_DIR / "metrics"
PREDICTIONS_DIR = MODEL_DATA_DIR / "predictions"
MODELS_DIR = MODEL_DATA_DIR / "models"
METADATA_DIR = MODEL_DATA_DIR / "metadata"

MODEL_SOURCE_PATH = PROCESSED_DIR / "vacancies_for_model_final.csv"

FROM_TRAIN_PATH = TRAIN_TEST_DIR / "model_from_train.csv"
FROM_TEST_PATH = TRAIN_TEST_DIR / "model_from_test.csv"
TO_TRAIN_PATH = TRAIN_TEST_DIR / "model_to_train.csv"
TO_TEST_PATH = TRAIN_TEST_DIR / "model_to_test.csv"

METRICS_PATH = METRICS_DIR / "model_metrics.csv"
SAMPLE_PREDICTIONS_PATH = PREDICTIONS_DIR / "sample_predictions.csv"
MODEL_FROM_PATH = MODELS_DIR / "model_from.joblib"
MODEL_TO_PATH = MODELS_DIR / "model_to.joblib"
METADATA_PATH = METADATA_DIR / "model_metadata.json"

ROUND_TO = 5000
RANDOM_STATE = 42
MIN_MONTHLY_SALARY = 20000
MAX_MONTHLY_SALARY = 800000

FEATURES = [
    "primary_role",
    "city_group",
    "experience_name",
    "schedule_name",
    "employment_name",
    "is_business_analyst",
    "is_system_analyst",
    "is_bi_analyst",
    "is_data_analyst",
    "is_product_analyst",
    "is_ds_ml",
    "has_sql",
    "has_python",
    "has_excel",
    "has_power_bi",
    "has_tableau",
    "has_bi_tools",
    "has_dwh",
    "has_etl",
    "has_ml",
    "has_statistics",
    "has_ab_testing",
    "has_bpmn",
    "has_uml",
    "has_api",
    "has_requirements",
    "has_1c",
    "has_jira",
]

CATEGORICAL_FEATURES = [
    "primary_role",
    "city_group",
    "experience_name",
    "schedule_name",
    "employment_name",
]

BINARY_FEATURES = [
    "is_business_analyst",
    "is_system_analyst",
    "is_bi_analyst",
    "is_data_analyst",
    "is_product_analyst",
    "is_ds_ml",
    "has_sql",
    "has_python",
    "has_excel",
    "has_power_bi",
    "has_tableau",
    "has_bi_tools",
    "has_dwh",
    "has_etl",
    "has_ml",
    "has_statistics",
    "has_ab_testing",
    "has_bpmn",
    "has_uml",
    "has_api",
    "has_requirements",
    "has_1c",
    "has_jira",
]

ROLE_OPTIONS = {
    "1": {
        "label": "Business Analytics",
        "primary_role": "Business Analytics",
        "flags": {"is_business_analyst": 1},
    },
    "2": {
        "label": "System Analytics",
        "primary_role": "System Analytics",
        "flags": {"is_system_analyst": 1},
    },
    "3": {
        "label": "BI/DWH Analytics",
        "primary_role": "BI/DWH Analytics",
        "flags": {"is_bi_analyst": 1},
    },
    "4": {
        "label": "Data Analytics",
        "primary_role": "Data Analytics",
        "flags": {"is_data_analyst": 1},
    },
    "5": {
        "label": "Product Analytics",
        "primary_role": "Product Analytics",
        "flags": {"is_product_analyst": 1},
    },
    "6": {
        "label": "DS/ML Analytics",
        "primary_role": "DS/ML Analytics",
        "flags": {"is_ds_ml": 1},
    },
}

EXPERIENCE_OPTIONS = {
    "1": "Нет опыта",
    "2": "От 1 года до 3 лет",
    "3": "От 3 до 6 лет",
    "4": "Более 6 лет",
}

CITY_OPTIONS = {
    "1": "Москва",
    "2": "Санкт-Петербург",
    "3": "Регионы",
}

SCHEDULE_OPTIONS = {
    "1": "На месте работодателя",
    "2": "Удалённо",
    "3": "Гибрид",
}

DEFAULT_EMPLOYMENT = "Полная"

SKILL_OPTIONS = {
    "1": ("SQL", "has_sql"),
    "2": ("Python", "has_python"),
    "3": ("Excel", "has_excel"),
    "4": ("BPMN / бизнес-процессы", "has_bpmn"),
    "5": ("UML / диаграммы", "has_uml"),
    "6": ("API / интеграции", "has_api"),
    "7": ("Требования / ТЗ / user stories", "has_requirements"),
    "8": ("1C", "has_1c"),
    "9": ("Jira / Confluence", "has_jira"),
    "10": ("Power BI", "has_power_bi"),
    "11": ("Tableau", "has_tableau"),
    "12": ("DWH", "has_dwh"),
    "13": ("ETL", "has_etl"),
    "14": ("ML", "has_ml"),
    "15": ("Статистика", "has_statistics"),
    "16": ("A/B-тестирование", "has_ab_testing"),
}


def ensure_output_dirs():
    for path in [TRAIN_TEST_DIR, METRICS_DIR, PREDICTIONS_DIR, MODELS_DIR, METADATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)
