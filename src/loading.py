"""
Loading the raw Kickstarter scrape.

The raw file has two gotchas that will stop you before you see a single row:

1. It is NOT UTF-8. It contains byte 0x92 (a Windows smart quote) and byte
   0x8f, which is undefined in cp1252. `latin-1` decodes every byte in the
   0x80-0xFF range without raising, so it is the only safe choice here.
   Nothing is silently corrupted that matters: the affected bytes are
   curly quotes and dashes inside project names, which we do not model on.

2. Column names contain spaces ("project id", "funded percentage",
   "reward levels"). We normalise to snake_case once, at load time, so no
   downstream code has to remember which columns need bracket access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_ENCODING = "latin-1"

# Canonical snake_case names, in the raw file's column order.
COLUMN_RENAMES = {
    "project id": "project_id",
    "name": "name",
    "url": "url",
    "category": "category",
    "subcategory": "subcategory",
    "location": "location",
    "status": "status",
    "goal": "goal",
    "pledged": "pledged",
    "funded percentage": "funded_percentage",
    "backers": "backers",
    "funded date": "funded_date",
    "levels": "levels",
    "reward levels": "reward_levels",
    "updates": "updates",
    "comments": "comments",
    "duration": "duration",
}

# Default location of the dataset relative to the repository root.
DEFAULT_PATH = Path("data/DSI_kickstarterscrape_dataset_csv.zip")


def load_raw(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Read the raw scrape exactly as distributed, with no filtering.

    Accepts either the .csv or the .zip that contains it -- pandas reads
    single-member zip archives natively, which is why we commit the 4.4 MB
    zip to the repo instead of the 12.9 MB csv.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download the dataset from "
            "https://www.kaggle.com/parienza/kickstarter and place the csv or "
            "zip in the data/ directory."
        )

    df = pd.read_csv(path, encoding=RAW_ENCODING)
    df = df.rename(columns=COLUMN_RENAMES)

    # Fail loudly if the file is not the one we documented against, rather
    # than letting a differently-shaped file flow silently downstream.
    expected = set(COLUMN_RENAMES.values())
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema -- missing columns: {sorted(missing)}")

    return df


def raw_overview(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, null count, distinct count, an example value.

    This is the table to read first (Task 1) and to paste into the README.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "nulls": int(s.isna().sum()),
                "null_pct": round(100 * s.isna().mean(), 2),
                "n_unique": int(non_null.nunique()),
                "example": (str(non_null.iloc[0])[:48] if len(non_null) else ""),
            }
        )
    return pd.DataFrame(rows)
