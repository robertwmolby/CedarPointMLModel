from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Iterable
import pandas as pd

from cpml.common.cp_date_handler import (
    is_labor_day,
    is_memorial_day,
    is_fathers_day,
    is_mothers_day,
    is_halloweekend,
    is_school_day,
    is_opening_day,
    is_closing_day,
    is_coaster_mania,
    is_fourth_of_july,
    get_season_week,
)

@dataclass(slots=True)
class PredictionRequest:
    """
    Class to hold all necessary information to use for predicting cedar point crowd levels.  By default only
    the date, temperature and rain are required.  Others are dervied from that but can be specified if so desired
    """
    # inputs
    prediction_date: date
    actual_temp: float
    actual_rain: float
    forecast_temp: float | None = None
    forecast_rain: float | None = None

    # derived/calendar
    month: int = field(init=False)
    day: int = field(init=False)
    year: int = field(init=False)
    labor_day: bool = field(init=False)
    memorial_day: bool = field(init=False)
    fourth_of_july: bool = field(init=False)
    fathers_day: bool = field(init=False)
    mothers_day: bool = field(init=False)
    hallowweekend: bool = field(init=False)
    school_season: bool = field(init=False)
    opening_day: bool = field(init=False)
    closing_day: bool = field(init=False)
    coaster_mania: bool = field(init=False)
    season_week: int = field(init=False)

    # weekday one-hots (Sun=0 … Sat=6)
    day_of_week_0: bool = field(init=False, default=False)
    day_of_week_1: bool = field(init=False, default=False)
    day_of_week_2: bool = field(init=False, default=False)
    day_of_week_3: bool = field(init=False, default=False)
    day_of_week_4: bool = field(init=False, default=False)
    day_of_week_5: bool = field(init=False, default=False)
    day_of_week_6: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        # Forecast defaults to actual if not provided
        if self.forecast_temp is None:
            self.forecast_temp = float(self.actual_temp)
        if self.forecast_rain is None:
            self.forecast_rain = float(self.actual_rain)

        d = self.prediction_date
        self.year = d.year
        self.month = d.month
        self.day = d.day

        # Holidays / flags
        self.labor_day = is_labor_day(d)
        self.memorial_day = is_memorial_day(d)
        self.fathers_day = is_fathers_day(d)
        self.mothers_day = is_mothers_day(d)
        self.hallowweekend = is_halloweekend(d)
        self.school_season = is_school_day(d)
        self.opening_day = is_opening_day(d)
        self.closing_day = is_closing_day(d)
        self.coaster_mania = is_coaster_mania(d)
        self.fourth_of_july = is_fourth_of_july(d)
        self.season_week = get_season_week(d)

        # Weekday one-hot: make Sunday=0 … Saturday=6
        # Python weekday(): Monday=0 … Sunday=6  ⇒ rotate by +1 mod 7
        sun0 = (d.weekday() + 1) % 7
        for i in range(7):
            setattr(self, f"day_of_week_{i}", i == sun0)

    def create_df(self, *, drop: Iterable[str] = ("year", "prediction_date")) -> pd.DataFrame:
        """Return a single-row DataFrame of features."""
        df = pd.DataFrame([asdict(self)])
        df.drop(columns=list(drop), errors="ignore", inplace=True)
        return df
