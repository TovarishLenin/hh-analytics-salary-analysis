import argparse
import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_CSV_PATH = ROOT / "data" / "processed" / "vacancies_for_model_final.csv"
OUTPUT_DIR = ROOT / "data" / "model" / "train_test"


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames


def write_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_rows(rows, test_size, seed):
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    test_count = round(len(rows) * test_size)
    test_rows = rows[:test_count]
    train_rows = rows[test_count:]
    return train_rows, test_rows


def main():
    parser = argparse.ArgumentParser(description="Split model CSV into train and test files.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test share, for example 0.2 or 0.3.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split.")
    args = parser.parse_args()

    rows, fieldnames = read_rows(MODEL_CSV_PATH)

    from_rows = [row for row in rows if row["salary_from"]]
    to_rows = [row for row in rows if row["salary_to"]]

    from_train, from_test = split_rows(from_rows, args.test_size, args.seed)
    to_train, to_test = split_rows(to_rows, args.test_size, args.seed)

    write_rows(OUTPUT_DIR / "model_from_train.csv", from_train, fieldnames)
    write_rows(OUTPUT_DIR / "model_from_test.csv", from_test, fieldnames)
    write_rows(OUTPUT_DIR / "model_to_train.csv", to_train, fieldnames)
    write_rows(OUTPUT_DIR / "model_to_test.csv", to_test, fieldnames)

    print(f"salary_from train: {len(from_train)}, test: {len(from_test)}")
    print(f"salary_to train: {len(to_train)}, test: {len(to_test)}")


if __name__ == "__main__":
    main()
