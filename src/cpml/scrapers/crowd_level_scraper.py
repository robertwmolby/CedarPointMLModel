# crowd_level_scraper.py

import requests
from bs4 import BeautifulSoup
from datetime import datetime, date
from datetime import timedelta
import sqlite3
import re
import psycopg2
import params

DB_STRING_PLACEHOLDER = '%s'  # for PostgreSQL
DB_INTEGER_PLACEHOLDER = '%s::integer'  # for PostgreSQL
DB_NUMERIC_PLACEHOLDER = '%s::numeric'  # for PostgreSQL
DB_DATE_PLACEHOLDER = '%s::date'  # for PostgreSQL
# DB_PLACEHOLDER = '?'  # for SqlLite



def main():
    # local sqllite database
    #conn = sqlite3.connect('../cp.db')
    # aws postgres database  jdbc:postgresql://cp-ai.cbsscwgeqp5j.us-east-2.rds.amazonaws.com:5432/postgres
    conn = psycopg2.connect(
        host='cp-ai.cbsscwgeqp5j.us-east-2.rds.amazonaws.com',
        port=5432,
        database='postgres',
        user='postgres',
        password='CedarP0int'
    )

    cursor = conn.cursor();
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS crowd_levels
                   (
                       date
                       TEXT
                       PRIMARY
                       KEY,
                       crowd_level
                       INTEGER,
                       year
                       INTEGER,
                       day_of_week
                       TEXT,
                       month
                       INTEGER,
                       is_open
                       TEXT CHECK(is_open IN ('Y','N')),
                       forecast_temp REAL,
                       actual_temp REAL,
                       forecast_wind REAL,
                       actual_wind REAL,
                       forecast_rain REAL,
                       actual_rain REAL,
                       hallowweekend TEXT CHECK(hallowweekend IN ('Y','N')),
                       military_days TEXT CHECK(military_days IN ('Y','N')),
                       light_up_the_point TEXT CHECK(light_up_the_point IN ('Y','N')),
                       coaster_mania TEXT CHECK(coaster_mania IN ('Y','N')),
                       opening_day TEXT CHECK(opening_day IN ('Y','N')),
                       boardwalk_nights TEXT CHECK(boardwalk_nights IN ('Y','N')),
                       fathers_day TEXT CHECK(fathers_day IN ('Y','N')),
                       mothers_day TEXT CHECK(mothers_day IN ('Y','N')),
                       fourth_of_july TEXT CHECK(fourth_of_july IN ('Y','N')),
                       memorial_day TEXT CHECK(memorial_day IN ('Y','N')),
                       labor_day TEXT CHECK(labor_day IN ('Y','N')),
                       closing_day TEXT CHECK(closing_day IN ('Y','N')),
                       covid_19_day TEXT CHECK(covid_19_day IN ('Y','N')),
                       reload_row TEXT CHECK(reload_row IN ('Y','N'))
                   )
                   ''')

    start_date = date(2016, 5, 1)
    end_date = date.today() - timedelta(days=1)

    current_date = start_date
    while (current_date < end_date):
        current_date = current_date + timedelta(days=1)
        date_str = current_date.strftime('%Y-%m-%d')
        if (should_date_be_processed(cursor, date_str) == False):
            continue
        url_date = current_date.strftime("%Y/%m/%d")
        is_open = 'N'
        crowd_level = 0
        day_of_week = current_date.strftime("%A")
        month = current_date.month
        year = current_date.year
        forecast_temp = None
        actual_temp = None
        forecast_wind = None
        actual_wind = None
        forecast_rain = None
        actual_rain = None
        # assuming all of 2020 is a covid-19 day since it skews results
        covid_19_day = 'N'
        if year == 2020:
            covid_19_day = 'Y'

        print(f"url_date is {url_date}")
        park = "50"
        url = "https://queue-times.com/en-US/parks/" + park + "/calendar/" + url_date
        response = requests.get(url)

        if response.status_code == 200:
            is_open = 'Y'
            soup = BeautifulSoup(response.text, 'lxml')
            label_span = soup.find("span", string="Crowd level")
            if label_span:
                # Go to the parent div, then find all spans inside it
                parent_div = label_span.find_parent("div")
                spans = parent_div.find_all("span")
                if len(spans) >= 2:
                    crowd_level = string_to_number(spans[1].text.strip())
                    print(f"Date {url_date}, Crowd level:{crowd_level}")
                    actual_temp = get_value(soup,"Temperature","Actual average")
                    print(f"Actual temp: {actual_temp}")
                    forecast_temp = get_value(soup,"Temperature","Forecast average")
                    actual_rain = get_value(soup,"Precipitation","Actual average")
                    print(f"Actual rain: {actual_rain}")
                    forecast_rain = get_value(soup,"Precipitation","Forecast average")
                    actual_wind = get_value(soup,"Wind speed","Actual average")
                    print(f"Actual wind: {actual_wind}")
                    forecast_wind = get_value(soup,"Wind speed","Forecast average")
                else:
                    print("Couldn't find second span in div.")
                    continue;
            else:
                print("Crowd level label not found.")
                continue;
        elif response.status_code == 404:
            is_open = 'N'
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            continue

            continue;
        cursor.execute(f'''
            INSERT INTO crowd_levels (
                date,
                crowd_level,
                day_of_week,
                month,
                is_open,
                year,
                forecast_temp,
                actual_temp,
                forecast_wind,
                actual_wind,
                forecast_rain,
                actual_rain,
                covid_19_day,
                reload_row
            ) VALUES ({DB_STRING_PLACEHOLDER}, {DB_STRING_PLACEHOLDER}, {DB_STRING_PLACEHOLDER}, 
                      {DB_INTEGER_PLACEHOLDER}, {DB_STRING_PLACEHOLDER}, {DB_INTEGER_PLACEHOLDER}, 
                      {DB_NUMERIC_PLACEHOLDER}, {DB_NUMERIC_PLACEHOLDER}, {DB_NUMERIC_PLACEHOLDER}, 
                      {DB_NUMERIC_PLACEHOLDER}, {DB_NUMERIC_PLACEHOLDER}, {DB_NUMERIC_PLACEHOLDER}, 
                      {DB_STRING_PLACEHOLDER}, 'N')
        ''', (date_str, crowd_level, day_of_week, month, is_open, year,
              forecast_temp, actual_temp, forecast_wind, actual_wind,
              forecast_rain, actual_rain, covid_19_day))

        conn.commit()  # ✅ Required to save the change to the .db file

    conn.close()

def get_value(soup, weather_type, label):
    weather_type_div = soup.select_one("div.panel:has(h2:contains('" + weather_type + "'))")
    # Find the label span
    label_span = weather_type_div.find("span", string=lambda text: label in text if text else False)
    # Get the next sibling span (the value)
    if (label_span is not None):
        return string_to_number(label_span.find_next_sibling("span").text.strip())
    return None

def celsius_to_fahrenheit(celsius):
    if celsius is None:
        return None
    return round(float(celsius * 9/5) + 32,2)

def string_to_number(string):
    if string is None:
        return None
    try:
        # Remove all non-digit characters (keeps decimal points)
        number_value = re.sub(r'[^\d.-]', '', str(string))
        # Convert to float first to handle decimal points
        return round(float(number_value),2)
    except (ValueError, TypeError):
        print(f"Error converting string to number: {string}")
        return None


# check to see if a record exists for the given date or if one exists that
# indicates it should be reloaded.  If it indicates it should be reloaded,
# delete it prior to processing.  return a boolean indicating whether or not
# the date should be processed.
def should_date_be_processed(cursor, processing_date):
    # Check if record exists and get its reload flag
    sql = f"SELECT reload_row FROM crowd_levels WHERE date = {DB_STRING_PLACEHOLDER}"
    cursor.execute(sql, (processing_date,))
    result = cursor.fetchone()

    if result is not None:
        reload_row = result[0]  # Get the reload value from the database
        if reload_row == 'N':
            return False
        else:  # reload = 'Y'
            delete_sql = f"DELETE FROM crowd_levels WHERE date = {DB_STRING_PLACEHOLDER}"
            cursor.execute(delete_sql, (processing_date,))
            cursor.connection.commit()
    return True

if __name__ == "__main__":
    main()



