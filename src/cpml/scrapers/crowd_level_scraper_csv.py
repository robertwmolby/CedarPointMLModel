# crowd_level_scraper.py
from typing import Union
import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import re
import os
import csv
import time
import wakepy


def main():
    path = '/storage';
    start_date = date(2016, 5, 1)
    end_date = date.today() - timedelta(days=1)

    file_name: str = path + os.path.sep + 'crowd_level.csv'
    if os.path.exists(file_name):
        os.remove(file_name)

    field_names: list[str] = ["date","crowd_level","actual_temp","actual_rain","actual_wind","forecast_temp","forecast_rain","forecast_wind"]

    with wakepy.keep.running():
        with open(file_name,"w") as file:
            writer = csv.DictWriter(file, fieldnames=field_names)
            writer.writeheader()
            current_date = start_date
            while current_date < end_date:
                current_date = current_date + timedelta(days=1)
                date_str = current_date.strftime('%Y-%m-%d')
                url_date = current_date.strftime("%Y/%m/%d")
                crowd_level = 0
                forecast_temp = None
                actual_temp = None
                forecast_wind = None
                actual_wind = None
                forecast_rain = None
                actual_rain = None
                print(f"url_date is {url_date}")
                park = "50"
                url = "https://queue-times.com/en-US/parks/" + park + "/calendar/" + url_date
                response = requests.get(url)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'lxml')
                    label_span = soup.find("span", string="Crowd level")
                    if label_span:
                        # Go to the parent div, then find all spans inside it
                        parent_div = label_span.find_parent("div")
                        spans = parent_div.find_all("span")
                        if len(spans) >= 2:
                            crowd_level = string_to_number(spans[1].text.strip())
                            actual_temp = get_value(soup,"Temperature","Actual average")
                            forecast_temp = get_value(soup,"Temperature","Forecast average")
                            actual_rain = get_value(soup,"Precipitation","Actual average")
                            forecast_rain = get_value(soup,"Precipitation","Forecast average")
                            actual_wind = get_value(soup,"Wind speed","Actual average")
                            forecast_wind = get_value(soup,"Wind speed","Forecast average")
                        else:
                            print("Couldn't find second span in div.")
                            continue
                    else:
                        print("Crowd level label not found.")
                        continue
                    crowd_level_details: dict[str, Union[int, float, date]] = {
                        "date": current_date,
                        "crowd_level": crowd_level,
                        "actual_temp": actual_temp,
                        "actual_rain":  actual_rain,
                        "forecast_temp": forecast_temp,
                        "forecast_rain": forecast_rain,
                        "forecast_wind": forecast_wind
                    }
                    print(f"Writing for date {url_date}.  Crowd level:{crowd_level}")
                    writer.writerow(crowd_level_details)
                elif response.status_code != 404: #404 just indicates the park isn't open
                    print(f"Failed to fetch storage. Status code: {response.status_code}")
                #time.sleep(1)

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


if __name__ == "__main__":
    main()



