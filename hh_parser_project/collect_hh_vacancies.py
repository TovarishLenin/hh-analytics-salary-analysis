import argparse
import csv
import html
import json
import re
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "vacancies_raw.json"
DETAILS_PATH = ROOT / "data" / "interim" / "vacancies_details.json"
CLEAN_CSV_PATH = ROOT / "data" / "processed" / "vacancies_clean.csv"
MODEL_CSV_PATH = ROOT / "data" / "processed" / "vacancies_for_model.csv"

API_URL = "https://api.hh.ru/vacancies"
AREAS_URL = "https://api.hh.ru/areas/113"
USER_AGENT = "HHCourseworkParser/1.0 (student-coursework)"

SEARCH_QUERIES = [
    "бизнес-аналитик",
    "бизнес аналитик",
    "business analyst",
    "системный аналитик",
    "системный-аналитик",
    "system analyst",
    "BI-аналитик",
    "BI аналитик",
    "би аналитик",
    "BI analyst",
    "DWH аналитик",
    "аналитик данных",
    "дата аналитик",
    "data analyst",
    "продуктовый аналитик",
    "product analyst",
    "data scientist",
    "ML аналитик",
    "machine learning",
]

CSV_COLUMNS = [
    "vacancy_id",
    "name",
    "alternate_url",
    "employer_name",
    "area_name",
    "city_group",
    "experience_name",
    "schedule_name",
    "employment_name",
    "salary_from",
    "salary_to",
    "salary_currency",
    "salary_gross",
    "has_salary",
    "salary_mid",
    "key_skills",
    "description_text",
    "primary_role",
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


def api_get(url, params=None):
    headers = {
        "User-Agent": USER_AGENT,
        "HH-User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def clean_html(text):
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"</p>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def get_name(data, key):
    value = data.get(key) or {}
    if isinstance(value, dict):
        return value.get("name", "").replace("\xa0", " ")
    return ""


def get_names(data, key):
    values = data.get(key) or []
    if isinstance(values, list):
        names = [item.get("name", "").replace("\xa0", " ") for item in values]
        return ", ".join(name for name in names if name)
    return ""


def calculate_salary_mid(salary_from, salary_to):
    if salary_from and salary_to:
        return round((float(salary_from) + float(salary_to)) / 2)
    if salary_from:
        return salary_from
    if salary_to:
        return salary_to
    return ""


def contains(text, *patterns):
    for pattern in patterns:
        if re.search(pattern, text, flags=re.I):
            return 1
    return 0


def get_city_group(area_name, schedule_name):
    text = f"{area_name} {schedule_name}".lower()
    if "удален" in text or "remote" in text:
        return "Удаленно/прочее"
    if area_name == "Москва":
        return "Москва"
    if area_name == "Санкт-Петербург":
        return "Санкт-Петербург"
    if area_name:
        return "Регионы"
    return "Удаленно/прочее"


def get_role_flags(text):
    return {
        "is_business_analyst": contains(
            text, r"бизнес[-\s]?аналит", r"business\s+analyst", r"\bba\b"
        ),
        "is_system_analyst": contains(
            text, r"системн\w*\s+аналит", r"system\s+analyst", r"\bsa\b"
        ),
        "is_bi_analyst": contains(
            text, r"\bbi\b", r"bi[-\s]?аналит", r"\bdwh\b", r"data\s+warehouse"
        ),
        "is_data_analyst": contains(
            text, r"аналитик\s+данных", r"data\s+analyst", r"data\s+analytics"
        ),
        "is_product_analyst": contains(
            text, r"продуктов\w*\s+аналит", r"product\s+analyst", r"product\s+analytics"
        ),
        "is_ds_ml": contains(
            text,
            r"data\s+scientist",
            r"machine\s+learning",
            r"\bml\b",
            r"машинн\w*\s+обуч",
        ),
    }


def get_primary_role(flags):
    priority = [
        ("is_ds_ml", "DS/ML Analytics"),
        ("is_product_analyst", "Product Analytics"),
        ("is_bi_analyst", "BI/DWH Analytics"),
        ("is_data_analyst", "Data Analytics"),
        ("is_system_analyst", "System Analytics"),
        ("is_business_analyst", "Business Analytics"),
    ]
    for field, role_name in priority:
        if flags[field] == 1:
            return role_name
    return "Other/Mixed Analytics"


def get_skill_flags(text):
    flags = {
        "has_sql": contains(text, r"\bsql\b", r"postgres", r"mysql", r"oracle"),
        "has_python": contains(text, r"\bpython\b", r"pandas", r"numpy"),
        "has_excel": contains(text, r"\bexcel\b", r"эксель", r"google\s+sheets"),
        "has_power_bi": contains(text, r"power\s*bi", r"\bpbi\b"),
        "has_tableau": contains(text, r"tableau"),
        "has_dwh": contains(text, r"\bdwh\b", r"data\s+warehouse", r"хранилищ\w*\s+данных"),
        "has_etl": contains(text, r"\betl\b", r"airflow"),
        "has_ml": contains(text, r"\bml\b", r"machine\s+learning", r"машинн\w*\s+обуч"),
        "has_statistics": contains(text, r"статист", r"statistics"),
        "has_ab_testing": contains(text, r"a/b", r"ab[-\s]?тест", r"а/б"),
        "has_bpmn": contains(
            text, r"\bbpmn\b", r"бизнес[-\s]?процесс", r"описан\w*\s+процесс"
        ),
        "has_uml": contains(text, r"\buml\b", r"use\s*case", r"диаграмм\w*"),
        "has_api": contains(text, r"\bapi\b", r"\brest\b", r"\bsoap\b", r"интеграц"),
        "has_requirements": contains(
            text,
            r"требован",
            r"техническ\w*\s+задан",
            r"\bтз\b",
            r"user\s+stor",
            r"постановк\w*\s+задач",
        ),
        "has_1c": contains(text, r"\b1с\b", r"\b1c\b"),
        "has_jira": contains(text, r"\bjira\b", r"confluence"),
    }
    flags["has_bi_tools"] = int(
        flags["has_power_bi"] == 1
        or flags["has_tableau"] == 1
        or contains(text, r"\bbi\b") == 1
    )
    return flags


def build_row(vacancy):
    salary = vacancy.get("salary") or {}
    salary_from = salary.get("from") or ""
    salary_to = salary.get("to") or ""
    salary_currency = salary.get("currency") or ""
    salary_gross = salary.get("gross")
    has_salary = int(bool(salary_from or salary_to))

    key_skills = ", ".join(skill.get("name", "") for skill in vacancy.get("key_skills", []))
    description_text = clean_html(vacancy.get("description", ""))
    name = vacancy.get("name", "")
    area_name = get_name(vacancy, "area")
    schedule_name = get_names(vacancy, "work_format") or get_name(vacancy, "schedule")
    employment_name = get_name(vacancy, "employment_form") or get_name(vacancy, "employment")

    role_flags = get_role_flags(name.lower())
    skill_text = " ".join([name, key_skills, description_text]).lower()
    skill_flags = get_skill_flags(skill_text)

    row = {
        "vacancy_id": vacancy.get("id", ""),
        "name": name,
        "alternate_url": vacancy.get("alternate_url", ""),
        "employer_name": get_name(vacancy, "employer"),
        "area_name": area_name,
        "city_group": get_city_group(area_name, schedule_name),
        "experience_name": get_name(vacancy, "experience"),
        "schedule_name": schedule_name,
        "employment_name": employment_name,
        "salary_from": salary_from,
        "salary_to": salary_to,
        "salary_currency": salary_currency,
        "salary_gross": "" if salary_gross is None else salary_gross,
        "has_salary": has_salary,
        "salary_mid": calculate_salary_mid(salary_from, salary_to),
        "key_skills": key_skills,
        "description_text": description_text,
        "primary_role": get_primary_role(role_flags),
    }
    row.update(role_flags)
    row.update(skill_flags)
    return row


def is_target_analytics_vacancy(row):
    role_fields = [
        "is_business_analyst",
        "is_system_analyst",
        "is_bi_analyst",
        "is_data_analyst",
        "is_product_analyst",
        "is_ds_ml",
    ]
    return any(row[field] == 1 for field in role_fields)


def get_russian_subjects():
    data = api_get(AREAS_URL)
    return [{"id": area["id"], "name": area["name"]} for area in data.get("areas", [])]


def search_vacancies(areas, max_pages, delay):
    raw_responses = []
    vacancies = {}

    for area in areas:
        print(f"Search area: {area['name']} ({area['id']})")
        for query in SEARCH_QUERIES:
            for page in range(max_pages):
                params = {
                    "text": query,
                    "area": area["id"],
                    "per_page": 100,
                    "page": page,
                    "search_field": "name",
                }
                data = api_get(API_URL, params)
                raw_responses.append(
                    {
                        "query": query,
                        "area_id": area["id"],
                        "area_name": area["name"],
                        "page": page,
                        "response": data,
                    }
                )

                for item in data.get("items", []):
                    vacancy_id = item.get("id")
                    if vacancy_id:
                        vacancies[vacancy_id] = item

                if page + 1 >= data.get("pages", 0):
                    break
                time.sleep(delay)

    return raw_responses, vacancies


def load_details(vacancies, max_details, delay):
    details = []
    vacancy_ids = list(vacancies.keys())
    if max_details:
        vacancy_ids = vacancy_ids[:max_details]

    for index, vacancy_id in enumerate(vacancy_ids, start=1):
        try:
            detail = api_get(f"{API_URL}/{vacancy_id}")
        except requests.exceptions.RequestException as error:
            print(f"Skipped vacancy {vacancy_id}: {error}")
            continue
        details.append(detail)
        print(f"Loaded details: {index}/{len(vacancy_ids)}")
        time.sleep(delay)

    return details


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Collect analytics vacancies from hh.ru API.")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--max-areas", type=int, default=None)
    parser.add_argument("--max-details", type=int, default=None)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--details-delay", type=float, default=0.2)
    args = parser.parse_args()

    areas = get_russian_subjects()
    if args.max_areas:
        areas = areas[: args.max_areas]

    raw_responses, vacancies = search_vacancies(areas, args.max_pages, args.delay)
    write_json(RAW_PATH, raw_responses)
    print(f"Unique vacancies found: {len(vacancies)}")

    details = load_details(vacancies, args.max_details, args.details_delay)
    write_json(DETAILS_PATH, details)

    rows = []
    for vacancy in details:
        row = build_row(vacancy)
        if is_target_analytics_vacancy(row):
            rows.append(row)
    write_csv(CLEAN_CSV_PATH, rows)

    model_rows = [
        row for row in rows if row["salary_currency"] == "RUR" and row["has_salary"] == 1
    ]
    write_csv(MODEL_CSV_PATH, model_rows)

    print(f"Saved clean CSV: {CLEAN_CSV_PATH} ({len(rows)} rows)")
    print(f"Saved model CSV: {MODEL_CSV_PATH} ({len(model_rows)} rows)")


if __name__ == "__main__":
    main()
