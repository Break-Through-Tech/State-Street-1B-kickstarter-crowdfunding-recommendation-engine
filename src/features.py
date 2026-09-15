"""
Feature preparation.

The organising idea in this module is the *prediction point*. The challenge
brief asks for a model that uses "information available around campaign
launch", so every feature built here is tagged as either:

  LAUNCH  -- knowable the moment the campaign goes live
  CAMPAIGN -- only knowable while the campaign is running
  LEAKY   -- an outcome, or a direct function of one

Two feature sets are exported (LAUNCH_ONLY and PLUS_CAMPAIGN) so the
modelling notebook can quantify what the CAMPAIGN features are worth
instead of assuming.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Leakage register -- columns that must never enter a model
# ---------------------------------------------------------------------------

LEAKY_COLUMNS = {
    "pledged": "Dollars raised. This IS the outcome.",
    "funded_percentage": "pledged / goal. Mechanically determines success.",
    "backers": "Count of people who pledged. Pure outcome.",
    "comments": "Audience-generated volume; scales with backers.",
    "status": "The raw target.",
    "success": "The encoded target.",
    "funded_date": "Campaign END date; unknown at launch.",
    "end_date": "Parsed campaign end; unknown at launch.",
    "url": "Identifier, not a feature.",
    "project_id": "Identifier, not a feature.",
    "location": "Superseded by city / state_raw.",
    "reward_levels": "Raw string; superseded by parsed tier features.",
    "name": "Raw string; superseded by parsed title features.",
}

# ---------------------------------------------------------------------------
# Reward tier parsing
# ---------------------------------------------------------------------------

# The reward_levels field looks like:
#   "$25,$50,$100,$250,$500,$1,000,$2,500"
# It is comma-separated AND the amounts contain thousands separators, so a
# naive .split(",") turns "$1,000" into "$1" and "000". This pattern only
# consumes a comma when it is followed by exactly three digits, which
# resolves the ambiguity correctly in both directions:
#   "$1,$5"    -> 1, 5        (comma is a separator)
#   "$1,000"   -> 1000        (comma is a thousands mark)
REWARD_PATTERN = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")


def parse_reward_levels(raw: str) -> list[float]:
    """Extract the numeric reward tiers from one reward_levels string."""
    if not isinstance(raw, str):
        return []
    return [float(m.replace(",", "")) for m in REWARD_PATTERN.findall(raw)]


def add_reward_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the reward_levels string into tier statistics. All LAUNCH."""
    df = df.copy()
    parsed = df["reward_levels"].map(parse_reward_levels)

    df["n_tiers_parsed"] = parsed.map(len)
    df["min_tier"] = parsed.map(lambda t: min(t) if t else np.nan)
    df["max_tier"] = parsed.map(lambda t: max(t) if t else np.nan)
    df["median_tier"] = parsed.map(lambda t: float(np.median(t)) if t else np.nan)
    df["tier_spread"] = df["max_tier"] - df["min_tier"]

    # Entry price relative to the ask: a $1 tier on a $50,000 goal is a very
    # different strategy from a $1 tier on a $500 goal.
    df["min_tier_pct_of_goal"] = df["min_tier"] / df["goal"]
    df["log_min_tier"] = np.log1p(df["min_tier"])
    df["log_max_tier"] = np.log1p(df["max_tier"])

    # Cross-check against the scraper's own `levels` count. Disagreement
    # flags rows where the reward string is truncated or malformed -- worth
    # inspecting in the EDA notebook rather than trusting blindly.
    df["tier_count_mismatch"] = df["n_tiers_parsed"] != df["levels"]
    return df


# ---------------------------------------------------------------------------
# Goal, duration, timing, title
# ---------------------------------------------------------------------------

def add_goal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Goal transforms. All LAUNCH.

    Goal is heavily right-skewed (median $4,000, max in the millions), so
    log1p is the version any linear model should see.
    """
    df = df.copy()
    df["log_goal"] = np.log1p(df["goal"])
    df["goal_per_tier"] = df["goal"] / df["levels"].replace(0, np.nan)
    df["log_goal_per_tier"] = np.log1p(df["goal_per_tier"])
    # Round-number asks ($5,000 vs $4,750) may signal deliberation.
    df["goal_is_round_1k"] = (df["goal"] % 1000 == 0).astype("int8")
    return df


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Duration transforms. All LAUNCH."""
    df = df.copy()
    df["duration_days"] = df["duration"]
    # 30 days is Kickstarter's default and by far the modal choice; being on
    # the default is itself informative.
    df["is_30_day"] = df["duration"].between(29.5, 30.5).astype("int8")
    df["duration_bucket"] = pd.cut(
        df["duration"],
        bins=[0, 20, 30, 45, 60, 100],
        labels=["<=20d", "21-30d", "31-45d", "46-60d", ">60d"],
    )
    df["goal_per_day"] = df["goal"] / df["duration"].replace(0, np.nan)
    df["log_goal_per_day"] = np.log1p(df["goal_per_day"])
    return df


def add_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Launch-timing features. All LAUNCH (they use launch_date, not end_date)."""
    df = df.copy()
    ld = df["launch_date"]
    df["launch_year"] = ld.dt.year
    df["launch_month"] = ld.dt.month
    df["launch_dow"] = ld.dt.dayofweek
    df["launch_quarter"] = ld.dt.quarter
    df["launch_is_weekend"] = ld.dt.dayofweek.isin([5, 6]).astype("int8")
    return df


def add_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap text features from the project title. All LAUNCH."""
    df = df.copy()
    name = df["name"].astype("string").fillna("")
    df["name_length"] = name.str.len()
    df["name_word_count"] = name.str.split().map(len)
    df["name_has_exclaim"] = name.str.contains("!", regex=False).astype("int8")
    df["name_has_digit"] = name.str.contains(r"\d", regex=True).astype("int8")
    # Shouty titles: share of alphabetic characters that are uppercase.
    alpha = name.str.count(r"[A-Za-z]").replace(0, np.nan)
    df["name_upper_ratio"] = (name.str.count(r"[A-Z]") / alpha).fillna(0)
    return df


# ---------------------------------------------------------------------------
# Categorical handling
# ---------------------------------------------------------------------------

def add_location_frequency(df: pd.DataFrame, reference: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Frequency-encode city, which has far too many levels to one-hot.

    Returns (df, reference) so the notebook can FIT the frequency table on
    the training split and APPLY it to the test split. Fitting it on the
    full dataset is a mild form of leakage -- it lets test rows influence a
    feature value -- and it is trivial to avoid, so we avoid it.
    """
    df = df.copy()
    city_key = df["city"].astype("string") + ", " + df["state_raw"].astype("string")
    if reference is None:
        reference = city_key.value_counts()
    df["city_project_count"] = city_key.map(reference).fillna(0)
    df["log_city_project_count"] = np.log1p(df["city_project_count"])
    return df, reference


def add_relative_goal(df: pd.DataFrame, reference: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Goal relative to the median goal in the same subcategory.

    A $20,000 ask is ambitious for a Short Film and modest for a Video Game.
    This feature encodes that context. Same fit/apply discipline as above --
    the median must come from training data only.
    """
    df = df.copy()
    if reference is None:
        reference = df.groupby("subcategory", observed=True)["goal"].median()
    subcat_median = df["subcategory"].map(reference)
    df["goal_vs_subcat_median"] = df["goal"] / subcat_median
    df["log_goal_vs_subcat_median"] = np.log1p(df["goal_vs_subcat_median"])
    return df, reference


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every stateless feature transform in order.

    The two stateful transforms (add_location_frequency, add_relative_goal)
    are deliberately NOT called here -- they need a train/test split to be
    done honestly. Call them from the modelling notebook.
    """
    df = add_reward_features(df)
    df = add_goal_features(df)
    df = add_duration_features(df)
    df = add_timing_features(df)
    df = add_title_features(df)
    return df


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

LAUNCH_NUMERIC = [
    "log_goal",
    "goal_is_round_1k",
    "duration_days",
    "is_30_day",
    "log_goal_per_day",
    "levels",
    "n_tiers_parsed",
    "log_min_tier",
    "log_max_tier",
    "median_tier",
    "tier_spread",
    "min_tier_pct_of_goal",
    "log_goal_per_tier",
    "launch_year",
    "launch_month",
    "launch_dow",
    "launch_is_weekend",
    "name_length",
    "name_word_count",
    "name_has_exclaim",
    "name_has_digit",
    "name_upper_ratio",
]

LAUNCH_CATEGORICAL = ["category", "subcategory", "state_raw"]

# Creator behaviour during the campaign. `updates` is the single strongest
# predictor in the dataset (success rate climbs from ~20% at 0 updates to
# ~92% at 13+), and it is ALSO unavailable at launch. That tension is the
# central methodological question of this project -- see notebooks/04.
CAMPAIGN_NUMERIC = ["updates"]


def feature_columns(feature_set: str = "launch_only") -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for a named feature set."""
    if feature_set == "launch_only":
        return list(LAUNCH_NUMERIC), list(LAUNCH_CATEGORICAL)
    if feature_set == "plus_campaign":
        return list(LAUNCH_NUMERIC) + list(CAMPAIGN_NUMERIC), list(LAUNCH_CATEGORICAL)
    raise ValueError(f"Unknown feature_set {feature_set!r}; use 'launch_only' or 'plus_campaign'.")


def build_model_matrix(
    df: pd.DataFrame,
    feature_set: str = "launch_only",
    *,
    extra_numeric: list[str] | None = None,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode the categoricals and return (X, y).

    Cardinality is modest after cleaning -- 13 categories, 49 subcategories,
    51 states -- so one-hot is fine and keeps coefficients interpretable for
    the Logistic Regression baseline. City is handled by frequency encoding
    instead (see add_location_frequency); pass it via extra_numeric.
    """
    numeric, categorical = feature_columns(feature_set)
    if extra_numeric:
        numeric = numeric + [c for c in extra_numeric if c in df.columns]

    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}. Did you run build_features()?")

    leaked = set(numeric + categorical) & set(LEAKY_COLUMNS)
    if leaked:
        raise ValueError(f"Leaky columns requested: {sorted(leaked)}")

    X = pd.get_dummies(
        df[numeric + categorical],
        columns=categorical,
        drop_first=drop_first,
        dtype="float64",
    )
    X = X.fillna(X.median(numeric_only=True))
    y = df["success"]
    return X, y
