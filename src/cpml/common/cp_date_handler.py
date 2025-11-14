"""
cp_dates.py

Utility functions for determining Cedar Point (CP) special dates:
- Opening/closing days (explicit map with rule-based fallback)
- Major U.S. holidays used in modeling
- Ohio-style school in-session checks
- Halloweekends ranges

Conventions
-----------
- All public functions accept/return `datetime.date`.
- Known dates are stored as constants and can be updated yearly.
- Fallback rules are documented in each function.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Tuple
from dateutil.relativedelta import relativedelta, SU, MO, FR, TH

# ------------------------------ Constants ------------------------------

# Known CoasterMania event dates
# TODO: Update annually as events are announced.
COASTER_MANIA_DAYS = {
    2017: date(2017, 6, 2),
    2018: date(2018, 6, 1),
    2019: date(2019, 6, 28),
    2021: date(2021, 6, 6),
    2022: date(2022, 6, 3),
    2023: date(2023, 6, 2),
    2024: date(2024, 6, 7),
    2025: date(2025, 6, 6),
}


# Explicit opening days by year.
# Fallback rule: first Saturday strictly AFTER May 3 (see is_opening_day).
# TODO: Append new seasons here when known.
OPENING_DAYS: Dict[int, date] = {
    2016: date.fromisoformat("2016-05-07"),
    2017: date.fromisoformat("2017-05-06"),
    2018: date.fromisoformat("2018-05-05"),
    2019: date.fromisoformat("2019-05-11"),
    2020: date.fromisoformat("2020-07-09"),  # COVID-delayed opening
    2021: date.fromisoformat("2021-05-14"),
    2022: date.fromisoformat("2022-05-07"),
    2023: date.fromisoformat("2023-05-06"),
    2024: date.fromisoformat("2024-05-04"),
    2025: date.fromisoformat("2025-05-10"),
}

# Explicit closing days by year.
# Fallback rule: first Sunday strictly PRIOR to Nov 3 (see is_closing_day).
# TODO: Append new seasons here when known.
CLOSING_DAYS: Dict[int, date] = {
    2016: date.fromisoformat("2016-10-30"),
    2017: date.fromisoformat("2017-10-29"),
    2018: date.fromisoformat("2018-10-28"),
    2019: date.fromisoformat("2019-10-27"),
    2020: date.fromisoformat("2020-11-01"),
    2021: date.fromisoformat("2021-10-31"),
    2022: date.fromisoformat("2022-10-30"),
    2023: date.fromisoformat("2023-10-29"),
    2024: date.fromisoformat("2024-11-02"),
}

# Known Halloweekend ranges by year.
# Fallback rule: 2nd Thursday in September through Oct 31 (see is_halloweekend).
# TODO: Add future ranges when released.
HALLOWEEKEND_RANGES: Dict[int, Tuple[date, date]] = {
    2017: (date(2017, 9, 17), date(2017, 10, 29)),
    2018: (date(2018, 9, 11), date(2018, 10, 31)),
    2019: (date(2019, 9, 13), date(2019, 10, 27)),
    2021: (date(2021, 9, 17), date(2021, 10, 31)),
    2022: (date(2022, 9, 15), date(2022, 10, 30)),
    2023: (date(2023, 9, 14), date(2023, 10, 29)),
    2024: (date(2024, 9, 12), date(2024, 10, 31)),
}

# ------------------------------ Opening / Closing ------------------------------

def is_opening_day(given_date: date) -> bool:
    return given_date == opening_day(given_date.year)


def get_season_week(given_date: date) -> int:
    """
    Return the number of weeks since opening day (week 1 = opening week).
    """
    year = given_date.year

    # Get the opening day for the year
    if year in OPENING_DAYS:
        opening_day = OPENING_DAYS[year]
    else:
        # Replicate your fallback logic for missing years
        may_3 = date(year, 5, 3)
        days_until_sat = (5 - may_3.weekday()) % 7
        opening_day = may_3 + timedelta(days=days_until_sat or 7)

    # Compute how many full weeks have passed
    delta_days = (given_date - opening_day).days
    if delta_days < 0:
        return 0  # before opening day
    return (delta_days // 7) + 1  # week 1 = opening week

def closing_day(year: int) -> date:
    """
    Return the park's closing day for the given year.

    Logic
    -----
    - If the year is in CLOSING_DAYS: return that date.
    - Else: first Sunday strictly PRIOR to November 3 of that year.
    """
    if year in CLOSING_DAYS:
        return CLOSING_DAYS[year]

    nov_3 = date(year, 11, 3)
    # Sunday = 6; `% 7` gives steps back; `or 7` enforces "strictly prior"
    days_back_to_sun = (nov_3.weekday() - 6) % 7
    return nov_3 - timedelta(days=days_back_to_sun or 7)

def is_closing_day(given_date: date) -> bool:
    return given_date == closing_day(given_date.year)

# ------------------------------ Holidays ------------------------------

def mothers_day(year: int) -> date:
    """Second Sunday in May."""
    return date(year, 5, 1) + relativedelta(weekday=SU(+2))

def fathers_day(year: int) -> date:
    """Third Sunday in June."""
    return date(year, 6, 1) + relativedelta(weekday=SU(+3))

def fourth_of_july(year: int) -> date:
    """July 4 (fixed)."""
    return date(year, 7, 4)

def memorial_day(year: int) -> date:
    """Last Monday in May."""
    return date(year, 5, 31) + relativedelta(weekday=MO(-1))

def labor_day(year: int) -> date:
    """First Monday in September."""
    return date(year, 9, 1) + relativedelta(weekday=MO(+1))

def opening_day(year: int) -> date:
    """
    Return the park's opening day for the given year.

    Logic
    -----
    - If the year is in OPENING_DAYS: return that date.
    - Else: first Saturday strictly AFTER May 3 of that year.
    """
    if year in OPENING_DAYS:
        return OPENING_DAYS[year]

    may_3 = date(year, 5, 3)
    # Saturday = 5; `% 7` gives offset to next Saturday; `or 7` enforces "strictly after"
    days_until_sat = (5 - may_3.weekday()) % 7
    return may_3 + timedelta(days=days_until_sat or 7)


def is_mothers_day(given_date: date) -> bool:
    """True iff `given_date` equals `mothers_day(year)`."""
    return given_date == mothers_day(given_date.year)

def is_fathers_day(given_date: date) -> bool:
    """True iff `given_date` equals `fathers_day(year)`."""
    return given_date == fathers_day(given_date.year)

def is_fourth_of_july(given_date: date) -> bool:
    """True iff `given_date` is July 4 of its year."""
    return given_date == fourth_of_july(given_date.year)

def is_memorial_day(given_date: date) -> bool:
    """True iff `given_date` equals `memorial_day(year)`."""
    return given_date == memorial_day(given_date.year)

def is_labor_day(given_date: date) -> bool:
    """True iff `given_date` equals `labor_day(year)`."""
    return given_date == labor_day(given_date.year)

# ------------------------------ School Year ------------------------------

def school_closing_day(year: int) -> date:
    """
    Last Friday on or before May 25.

    Rationale: generalized Ohio pattern for end of school year.
    """
    may_25 = date(year, 5, 25)
    # Friday = 4
    days_back = (may_25.weekday() - 4) % 7
    return may_25 - timedelta(days=days_back)

def school_opening_day(year: int) -> date:
    """
    First Monday strictly AFTER August 16.

    Rationale: generalized Ohio start pattern.
    """
    aug_16 = date(year, 8, 16)
    # Monday = 0
    days_fwd = (0 - aug_16.weekday()) % 7
    if days_fwd == 0:
        days_fwd = 7  # strictly AFTER
    return aug_16 + timedelta(days=days_fwd)

def is_school_day(given_date: date) -> bool:
    """
    True if the date is during the typical school year segments:
    - Jan .. school_closing_day(year)
    - school_opening_day(year) .. Dec
    """
    year = given_date.year
    opening = school_opening_day(year)
    closing = school_closing_day(year)
    return given_date >= opening or given_date <= closing

# ------------------------------ Halloweekends ------------------------------

def is_halloweekend(given_date: date) -> bool:
    """
    True if `given_date` falls in a defined Halloweekends range.

    Logic
    -----
    - If HALLOWEEKEND_RANGES has the year: use explicit (start, end).
    - Else: fallback to 2nd Thursday in September through October 31 (inclusive).
    """
    year = given_date.year
    if year in HALLOWEEKEND_RANGES:
        start, end = HALLOWEEKEND_RANGES[year]
        return start <= given_date <= end

    second_thu = date(year, 9, 1) + relativedelta(weekday=TH(+2))
    halloween = date(year, 10, 31)
    return second_thu <= given_date <= halloween

def is_coaster_mania(given_date: date) -> bool:
    """
    Return True if `given_date` is a CoasterMania event day.

    Logic
    -----
    - If the year is in COASTER_MANIA_DAYS, returns True only if the date matches.
    - If the year is not in the dict, returns True if the date is the first Friday in June.
    """
    year = given_date.year

    # Case 1: Explicit known year
    if year in COASTER_MANIA_DAYS:
        return given_date == COASTER_MANIA_DAYS[year]

    # Case 2: Fallback → first Friday in June
    june_1 = date(year, 6, 1)
    days_until_friday = (4 - june_1.weekday()) % 7  # Friday = 4
    first_friday = june_1 + timedelta(days=days_until_friday)
    return given_date == first_friday

def is_open(d: date) -> bool:
    """Open if within season window, and (after Labor Day → Thu–Sun only)."""
    od = opening_day(d.year)
    cd = closing_day(d.year)
    if not (od <= d <= cd):
        return False

    ld = labor_day(d.year)
    if d <= ld:
        return True  # pre-Labor Day: open every day (inside season window)

    # post-Labor Day: only Thu(3)–Sun(6)
    return d.weekday() >= 3

# ------------------------------ Quick Self-Check ------------------------------

if __name__ == "__main__":
    # Minimal smoke tests you can expand or convert to pytest.
    assert is_opening_day(date(2025, 5, 10))
    assert is_closing_day(date(2024, 11, 2))
    assert is_memorial_day(date(2025, 5, 26))
    assert is_labor_day(date(2025, 9, 1))
    # School boundaries sanity
    sc = school_closing_day(2025)
    so = school_opening_day(2025)
    assert sc.weekday() == 4  # Friday
    assert so.weekday() == 0  # Monday
    print("cp_dates: basic checks passed.")


__all__ = [
    "is_opening_day", "is_closing_day", "is_halloweekend", "is_school_day",
    "mothers_day", "fathers_day", "fourth_of_july", "memorial_day", "labor_day",
    "is_mothers_day", "is_fathers_day", "is_fourth_of_july",
    "is_memorial_day", "is_labor_day","is_coaster_mania",
    "school_opening_day", "school_closing_day", "get_season_week",
    "COASTER_MANIA_DAYS", "OPENING_DAYS", "CLOSING_DAYS", "HALLOWEEKEND_RANGES",
    "is_open", "opening_day", "closing_day"
]
