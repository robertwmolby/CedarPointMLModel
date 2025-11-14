import pandas as pd
from pandas import DataFrame

from cpml.common.logging import get_logger
from cpml.common import cp_date_handler as dh


logger = get_logger("predictive.preprocess")

def celsius_to_farenheit(celsius_temp):
    """Celsius → Fahrenheit. Works with scalars/Series; NaNs pass through."""
    return celsius_temp * 9.0 / 5.0 + 32.0

def fill_forecast_and_actual(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Fill forecast temp/rain with actuals when forecast is NaN.
    - For wind, fill forecast from actual, then fill actual from (now-updated) forecast.
    - Convert all temperature fields from celsius to fahrenheit.
    """
    df = df.copy()

    # 1) Fill forecast temp/rain with actuals
    if {"forecast_temp","actual_temp"}.issubset(df.columns):
        df["forecast_temp"] = df["forecast_temp"].fillna(df["actual_temp"])
    if {"forecast_rain","actual_rain"}.issubset(df.columns):
        df["forecast_rain"] = df["forecast_rain"].fillna(df["actual_rain"])

    # 2) Wind: forecast <- actual, then actual <- forecast (post-update)
    if {"forecast_wind","actual_wind"}.issubset(df.columns):
        df["forecast_wind"] = df["forecast_wind"].fillna(df["actual_wind"])
        df["actual_wind"]   = df["actual_wind"].fillna(df["forecast_wind"])

    # 3) Convert temperatures to Fahrenheit (after filling)
    if "actual_temp" in df.columns:
        df["actual_temp"] = celsius_to_farenheit(df["actual_temp"])
    if "forecast_temp" in df.columns:
        df["forecast_temp"] = celsius_to_farenheit(df["forecast_temp"])

    return df

def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures 'date' column is datetime and adds year, month, and day columns.
    """
    df = df.copy()
    if "date" not in df.columns:
        raise KeyError("'date' column missing from dataframe")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    return df

def drop_low_value_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Return a copy of `df` with specified columns removed.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cols : list[str]
        Columns to drop if present.

    Returns
    -------
    pd.DataFrame
        New dataframe without the listed columns.
    """
    to_drop = [c for c in cols if c in df.columns]
    if not to_drop:
        logger.warning("No low-value columns found to drop, just returning copy of "
                       "original dataframe. Requested: %s", cols)
        return df.copy()
    logger.info("Dropping low-value columns: %s", to_drop)
    return df.drop(columns=to_drop)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich dataframe with calendar-based event and season features.
    """
    df_enriched: DataFrame = df.copy()
    dates = df_enriched["date"].dt.date
    df_enriched["labor_day"] = dates.map(dh.is_labor_day)
    df_enriched["memorial_day"] = dates.map(dh.is_memorial_day)
    df_enriched["fourth_of_july"] = dates.map(dh.is_fourth_of_july)
    df_enriched["fathers_day"] = dates.map(dh.is_fathers_day)
    df_enriched["mothers_day"] = dates.map(dh.is_mothers_day)
    df_enriched["hallowweekend"] = dates.map(dh.is_halloweekend)
    df_enriched["school_season"] = dates.map(dh.is_school_day)
    df_enriched["opening_day"] = dates.map(dh.is_opening_day)
    df_enriched["closing_day"] = dates.map(dh.is_closing_day)
    df_enriched["coaster_mania"] = dates.map(dh.is_coaster_mania)
    df_enriched["day_of_week"] = df_enriched["date"].dt.weekday
    df_enriched["season_week"] = dates.map(dh.get_season_week)
    return df_enriched

def remove_year(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows from the COVID-disrupted year and drop the year column since
    year trending is not considered."""
    df_non_covid = df.loc[df["year"] != 2020].copy()
    return df_non_covid.drop(columns="year", errors="ignore")

def prepare_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform final preprocessing before model training:
    - Drop non-feature columns like 'date'
    - Convert booleans to ints (0/1)
    Returns a clean copy of the DataFrame ready for modeling.
    """
    df_updated = df.copy()

    # Drop 'date' if present
    if "date" in df_updated.columns:
        df_updated = df_updated.drop(columns=["date"])
        logger.info("Dropped non-feature column: 'date'")

    # convert day of week to a category...
    df = pd.get_dummies(df, columns=["day_of_week"], prefix="dow", dtype="uint8")

    # Normalize boolean columns
    bool_cols = df_updated.select_dtypes(include=["bool"]).columns
    if not bool_cols.empty:
        df_updated[bool_cols] = df_updated[bool_cols].astype(int)
        logger.info("Converted boolean columns to int: %s", list(bool_cols))

    return df_updated