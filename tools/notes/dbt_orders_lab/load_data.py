"""Apply a fixture SQL file to the lab's local DuckDB database."""
import argparse
from pathlib import Path

import duckdb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", choices=["initial", "late", "bad", "fix", "order-status"])
    args = parser.parse_args()
    project = Path(__file__).resolve().parent

    with duckdb.connect(str(project / "orders_lab.duckdb")) as connection:
        connection.execute((project / "data" / f"{args.batch}.sql").read_text())
    print(f"Applied {args.batch} to orders_lab.duckdb")


if __name__ == "__main__":
    main()
