import joblib
import pandas as pd

from model_config import (
    BINARY_FEATURES,
    CITY_OPTIONS,
    DEFAULT_EMPLOYMENT,
    EXPERIENCE_OPTIONS,
    FEATURES,
    MODEL_FROM_PATH,
    MODEL_TO_PATH,
    ROLE_OPTIONS,
    ROUND_TO,
    SCHEDULE_OPTIONS,
    SKILL_OPTIONS,
)


def print_menu(title, options):
    print(f"\n{title}")
    for key, value in options.items():
        label = value["label"] if isinstance(value, dict) else value
        print(f"{key}. {label}")


def ask_choice(title, options):
    while True:
        print_menu(title, options)
        choice = input("Введите номер: ").strip()
        if choice in options:
            return choice
        print("Такого варианта нет. Попробуйте еще раз.")


def ask_skills():
    print("\nВыберите навыки через пробел:")
    for key, (label, _) in SKILL_OPTIONS.items():
        print(f"{key}. {label}")

    raw = input("Введите номера навыков: ").strip()
    selected = set(raw.split()) if raw else set()

    skill_flags = {feature: 0 for _, feature in SKILL_OPTIONS.values()}
    for item in selected:
        if item in SKILL_OPTIONS:
            _, feature = SKILL_OPTIONS[item]
            skill_flags[feature] = 1
        else:
            print(f"Навык с номером {item} пропущен: такого номера нет.")
    if skill_flags.get("has_power_bi") == 1 or skill_flags.get("has_tableau") == 1:
        skill_flags["has_bi_tools"] = 1
    return skill_flags


def build_user_row(role_choice, experience_choice, city_choice, schedule_choice, skill_flags):
    role = ROLE_OPTIONS[role_choice]

    row = {
        "primary_role": role["primary_role"],
        "city_group": CITY_OPTIONS[city_choice],
        "experience_name": EXPERIENCE_OPTIONS[experience_choice],
        "schedule_name": SCHEDULE_OPTIONS[schedule_choice],
        "employment_name": DEFAULT_EMPLOYMENT,
    }

    # Сначала все бинарные признаки равны 0, потом выбранные роль и навыки включаются.
    for feature in BINARY_FEATURES:
        row[feature] = 0
    for feature, value in role["flags"].items():
        row[feature] = value
    row.update(skill_flags)

    return pd.DataFrame([row], columns=FEATURES)


def round_salary(value):
    return int(round(value / ROUND_TO) * ROUND_TO)


def main():
    if not MODEL_FROM_PATH.exists() or not MODEL_TO_PATH.exists():
        print("Сначала обучите модель командой: python3 train_salary_model.py")
        return

    model_from = joblib.load(MODEL_FROM_PATH)
    model_to = joblib.load(MODEL_TO_PATH)

    print("Прототип оценки зарплатной вилки аналитика")
    print(
        "Внимание: модель не проверяет реалистичность сочетания роли и навыков. "
        "Если выбрать редкую или противоречивую комбинацию, прогноз может быть менее надежным."
    )

    role_choice = ask_choice("Выберите роль:", ROLE_OPTIONS)
    experience_choice = ask_choice("Выберите опыт работы:", EXPERIENCE_OPTIONS)
    city_choice = ask_choice("Выберите городскую группу:", CITY_OPTIONS)
    schedule_choice = ask_choice("Выберите формат работы:", SCHEDULE_OPTIONS)
    skill_flags = ask_skills()
    print(
        "\nВнимание: если выбранные навыки плохо соответствуют выбранной роли, "
        "модель все равно рассчитает вилку, но такую оценку нужно интерпретировать осторожно."
    )

    user_row = build_user_row(
        role_choice,
        experience_choice,
        city_choice,
        schedule_choice,
        skill_flags,
    )

    pred_from = float(model_from.predict(user_row)[0])
    pred_to = float(model_to.predict(user_row)[0])

    low = round_salary(min(pred_from, pred_to))
    high = round_salary(max(pred_from, pred_to))

    print("\nВведенные параметры:")
    print(f"Роль: {ROLE_OPTIONS[role_choice]['label']}")
    print(f"Опыт: {EXPERIENCE_OPTIONS[experience_choice]}")
    print(f"Городская группа: {CITY_OPTIONS[city_choice]}")
    print(f"Формат работы: {SCHEDULE_OPTIONS[schedule_choice]}")
    print(f"Тип занятости: {DEFAULT_EMPLOYMENT}")

    selected_skill_names = [
        label for key, (label, feature) in SKILL_OPTIONS.items() if skill_flags[feature] == 1
    ]
    print("Навыки: " + (", ".join(selected_skill_names) if selected_skill_names else "не выбраны"))

    print(f"\nОриентировочная зарплатная вилка: {low:,} - {high:,} руб.".replace(",", " "))
    print("Это прототипная оценка по вакансиям hh.ru, а не гарантия фактической зарплаты.")


if __name__ == "__main__":
    main()
