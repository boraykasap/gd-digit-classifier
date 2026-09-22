"""MATH-329 HW1 — entry point.

Run as: python main.py  (from code/, per the submission instructions)
Reads data from ../data/, writes results to ../results/.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / ".." / "data"
RESULTS_DIR = Path(__file__).resolve().parent / ".." / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    # TODO: load data from DATA_DIR, run the optimization, write
    # requested outputs to RESULTS_DIR.


if __name__ == "__main__":
    main()
