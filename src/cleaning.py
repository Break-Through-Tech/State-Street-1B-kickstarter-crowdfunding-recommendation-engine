"""
Cleaning the Kickstarter scrape into a modelling table.

Every transformation appends a row to a step log, so the cleaning notebook
can print an auditable trail of "rows in -> rows out, and why". The advisor
asked for reproducible, documented preprocessing; the log *is* that
documentation.

Deliberate deviations from the original project's recipe are marked with
DEVIATION in the comments below. There are three, and all three are
defensible; the point is to name them, not hide them.
"""

from __future__ import annotations

import html
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# 50 states + DC + the five inhabited territories. An explicit whitelist
# beats "last token is 2 characters long" because it also rejects genuine
# non-state tokens, but it MUST be applied case-insensitively: Montana
# appears in the raw data as both "MT" (89 rows) and "Mt" (36 rows).
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP",
}

# Statuses that represent a resolved binary outcome. live/canceled/suspended
# are dropped: "live" has no outcome yet, and canceled/suspended campaigns
# ended for reasons unrelated to whether they would have funded.
BINARY_STATUSES = ("successful", "failed")

# 2**31 - 1 in cents. Appears once as a goal value and is an integer
# overflow artefact from the scraper, not a real fundraising target.
INT32_SENTINEL = 21474836.47

# Kickstarter capped campaign length at 60 days in June 2011. Longer
# durations are legitimate but indicate a pre-2011 campaign, which is a
# different regime -- we flag rather than drop.
KICKSTARTER_MAX_DURATION_DAYS = 60


class StepLog:
    """Records row counts before and after each cleaning step."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def record(self, name: str, before: int, after: int, note: str = "") -> None:
        self.steps.append(
            {
                "step": name,
                "rows_before": before,
                "rows_after": after,
                "removed": before - after,
                "note": note,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


# ---------------------------------------------------------------------------
# Individual cleaning operations
# ---------------------------------------------------------------------------

def unescape_html_columns(df: pd.DataFrame, columns=("category", "subcategory", "name")) -> pd.DataFrame:
    """Decode HTML entities that the scraper never unescaped.

    This is the highest-impact single fix in the whole pipeline. The raw file
    contains BOTH "Film &amp; Video" (10,925 rows) and "Film & Video" (411
    rows) as separate category strings -- the same category, split in two,
    with the smaller group looking like a rare category to any model.

    Also affects subcategories "Board &amp; Card Games" and
    "Country &amp; Folk". Fixing it takes categories 14 -> 13 and
    subcategories 51 -> 49.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("string").map(
                lambda v: html.unescape(v) if pd.notna(v) else v
            )
    return df


def split_location(df: pd.DataFrame) -> pd.DataFrame:
    """Split `location` into `city` and `state_raw`.

    The field is mostly "City, ST" but not always. Of 44,635 non-null
    values, 8 do not have exactly one comma:

      - "St Croix, Virgin Islands, U.S."      (US territory, two commas)
      - "Nakagyo Ward, Kyoto, Japan"          (international, two commas)
      - "10, Middleburg, MD"                  (malformed, leading number)
      - "Maar"                                (no comma at all)

    Splitting on the LAST comma handles the malformed-leading-number case
    correctly and leaves the territory/international rows with a long
    `state_raw` that the US filter will reject.
    """
    df = df.copy()
    parts = df["location"].astype("string").str.rsplit(",", n=1)
    df["city"] = parts.str[0].str.strip()
    df["state_raw"] = parts.str[-1].str.strip()
    # A comma-less value like "Maar" yields city == state_raw; blank the city.
    no_comma = df["location"].notna() & ~df["location"].astype("string").str.contains(",", na=False)
    df.loc[no_comma, "city"] = pd.NA
    return df


def parse_funded_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `funded_date` and recover the launch date.

    `funded_date` is the campaign's END date (RFC-2822 format, e.g.
    "Fri, 19 Aug 2011 19:28:17 -0000"), NOT the launch date. Since
    `duration` is the campaign length in days, launch = end - duration.

    Recovering launch matters twice over: it gives launch-timing features
    (year, month, weekday) that are genuinely known at prediction time, and
    it lets you do a chronological train/test split instead of a random one.
    """
    df = df.copy()
    end = pd.to_datetime(df["funded_date"], format="%a, %d %b %Y %H:%M:%S %z", errors="coerce", utc=True)
    df["end_date"] = end
    df["launch_date"] = end - pd.to_timedelta(df["duration"], unit="D")
    return df


def flag_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean quality flags without dropping anything.

    Flagging rather than dropping keeps the decision visible and reversible
    in the EDA notebook. Drop later, explicitly, if the evidence says so.
    """
    df = df.copy()
    df["flag_goal_sentinel"] = df["goal"].eq(INT32_SENTINEL)
    df["flag_goal_tiny"] = df["goal"].lt(100)
    df["flag_goal_huge"] = df["goal"].gt(1_000_000)
    df["flag_duration_over_cap"] = df["duration"].gt(KICKSTARTER_MAX_DURATION_DAYS)
    df["flag_bad_date"] = df["end_date"].isna()
    quality_cols = [c for c in df.columns if c.startswith("flag_")]
    df["n_quality_flags"] = df[quality_cols].sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def clean(
    df: pd.DataFrame,
    *,
    us_only: bool = True,
    dedupe: bool = True,
    drop_goal_sentinel: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline.

    Returns (clean_df, step_log_df).

    Passing dedupe=False and drop_goal_sentinel=False reproduces the
    original project's 38,492-row table almost exactly, which is useful for
    validating against their published benchmark.
    """
    log = StepLog()
    n0 = len(df)
    log.record("raw", n0, n0, "as distributed")

    # --- Step 1: resolved binary outcomes only -----------------------------
    before = len(df)
    df = df[df["status"].isin(BINARY_STATUSES)].copy()
    log.record(
        "filter_status",
        before,
        len(df),
        "dropped live (no outcome yet), canceled, suspended",
    )

    # --- Step 2: unescape HTML before any grouping on category -------------
    df = unescape_html_columns(df)
    log.record("unescape_html", len(df), len(df), "Film &amp; Video -> Film & Video")

    # --- Step 3: split location, then filter to the US ---------------------
    df = split_location(df)
    if us_only:
        before = len(df)
        # Case-insensitive on purpose. Montana is spelled "MT" in most rows
        # and "Mt" in 36 others (Kalispell, Whitefish, Gallatin Gateway --
        # unambiguously Montana). A case-sensitive check silently deletes
        # those campaigns; a "len == 2" check keeps them but as a SEPARATE
        # state level, which is the same bug as `Film &amp; Video`: one real
        # entity split into two encoded categories.
        state_code = df["state_raw"].str.upper()
        df = df[state_code.isin(US_STATES)].copy()
        # Normalise so "Mt" and "MT" group as one state downstream.
        df["state_raw"] = df["state_raw"].str.upper()
        log.record(
            "filter_us",
            before,
            len(df),
            "state_raw in US whitelist, case-insensitive; codes upper-cased",
        )

    # --- Step 4: dates ------------------------------------------------------
    df = parse_funded_date(df)
    log.record("parse_dates", len(df), len(df), "recovered launch_date = end - duration")

    # --- Step 5: missing values --------------------------------------------
    # Only two columns still carry nulls at this point: `pledged` (a leaky
    # outcome column we will not model on) and `reward_levels`. Dropping on
    # ALL columns, as the original did, silently deletes rows for the sake of
    # a column that never enters the model.
    before = len(df)
    required = ["goal", "duration", "levels", "category", "subcategory", "city", "state_raw", "status"]
    df = df.dropna(subset=required).copy()
    log.record(
        "drop_missing_required",
        before,
        len(df),
        f"dropna on modelling columns only: {', '.join(required)}",
    )

    before = len(df)
    df = df[df["reward_levels"].notna()].copy()
    log.record("drop_missing_reward_levels", before, len(df), "needed for tier features")

    # --- Step 6: duplicates -------------------------------------------------
    if dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=["project_id"], keep="first").copy()
        # DEVIATION 2: the original project did not dedupe. 113 duplicated
        # project_ids survive its filters, so our table is intentionally
        # ~113 rows smaller than their published 38,491.
        log.record(
            "dedupe_project_id",
            before,
            len(df),
            "DEVIATION: original project kept duplicates",
        )

    # --- Step 7: quality flags, then the one drop we can justify -----------
    df = flag_data_quality(df)
    if drop_goal_sentinel:
        before = len(df)
        df = df[~df["flag_goal_sentinel"]].copy()
        # DEVIATION 3: $21,474,836.47 is 2**31-1 cents -- a scraper overflow,
        # not a fundraising target. Keeping it distorts every goal statistic.
        log.record("drop_goal_sentinel", before, len(df), "int32 overflow artefact")

    # --- Step 8: the target -------------------------------------------------
    df["success"] = (df["status"] == "successful").astype("int8")
    log.record("add_target", len(df), len(df), "success = 1 if status == 'successful'")

    df = df.reset_index(drop=True)
    step_log = log.to_frame()

    if verbose:
        print(step_log.to_string(index=False))
        print(f"\nFinal: {len(df):,} rows x {df.shape[1]} columns")
        rate = df["success"].mean()
        print(f"Class balance: {rate:.1%} successful / {1 - rate:.1%} failed")
        print(f"Majority-class accuracy baseline: {max(rate, 1 - rate):.1%}")

    return df, step_log
