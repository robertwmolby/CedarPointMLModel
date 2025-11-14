# crowd_level_scraper.py
from datetime import datetime

import psycopg2
import requests
from bs4 import BeautifulSoup
from cpml.scrapers.coaster_page import CoasterPage

def create_section_map(soup: BeautifulSoup):
    section_map = {}
    sections = soup.find_all('section')
    for section in sections:
        section_headers = section.find_all("h3")
        for section_header in section_headers:
            section_name = section_header.text
            section_rows = section.find_all("tr")
            for section_row in section_rows:
                table_headers = section_row.find_all("th")
                table_cells = section_row.find_all("td")
                if len(table_headers) > 0 and len(table_cells) > 0:
                    label = section_row.find_all("th")[0].text
                    value = section_row.find_all("td")[0].text
                    map_entry_name = section_name + ":" + label
                    section_map[map_entry_name] = value
    return section_map

def time_to_seconds(time_str):
    try:
        # Add "0:" in front to handle MM:SS format as time
        time_obj = datetime.strptime(f"0:{time_str}", "%H:%M:%S")
        return time_obj.minute * 60 + time_obj.second
    except ValueError:
        return None


def populate_details_from_page(coaster_page: CoasterPage):
    page_url = "https://rcdb.com" + coaster_page.url
    response = requests.get(page_url)
    if (response.status_code == 200):
        soup = BeautifulSoup(response.text, 'lxml')
        # labels are in anchor tags here.  First can be ignored (is "Roller Coaster")    #feature > ul:nth-child(3)


        # manufacturer, model, submodel are 3 anchor tags in the following
        # #feature > div.scroll > p > a:nth-child(1).

        # Length, Height, Drop, Spped, Inversions, Vertical Angle, Duration are in
        # body > section:nth-child(3) > table  look throw rows.  Each row has a table header (th) with field
        # type and then a td with the field value.

        section_map = create_section_map(soup)
        length = section_map.get("Tracks:Length")
        if length is not None:
            try:
                length = float(str.replace(length, "ft", "").strip(" "))
            except ValueError as e:
                length = None
        coaster_page.length = length
        height = section_map.get("Tracks:Height")
        if height is not None:
            try:
                height = float(str.replace(height, "ft", "").strip(" "))
            except ValueError as e:
                height = None
        coaster_page.height = height
        drop = section_map.get("Tracks:Drop")
        if drop is not None:
            try:
                drop = float(str.replace(drop,"ft","").strip(" "))
            except ValueError as e:
                drop = None
        coaster_page.drop = drop
        speed = section_map.get("Tracks:Speed")
        if speed is not None:
            try:
                speed = float(str.replace(speed,"mph","").strip(""))
            except ValueError as e:
                speed = None
        coaster_page.speed = speed
        inversion_count = section_map.get("Tracks:Inversions")
        if inversion_count is not None:
            try:
                inversion_count = int(inversion_count)
            except ValueError as e:
                inversion_count = None
        coaster_page.inversion_count = inversion_count
        vertical_angle = section_map.get("Tracks:Vertical Angle")
        if vertical_angle is not None:
            try:
                vertical_angle = float(str.replace(vertical_angle,"°","").strip(""))
            except ValueError as e:
                vertical_angle = None
        coaster_page.vertical_angle = vertical_angle
        duration = section_map.get("Tracks:Duration")
        if duration is not None:
            duration = time_to_seconds(duration)
            if duration is not None:
                try:
                    duration = int(duration)
                except ValueError as e:
                    duration = None
        coaster_page.duration = duration
        coaster_page.g_force = section_map.get("Tracks:G-Force")
        coaster_page.restraints = section_map.get("Trains:Restraints")

        main_data_contents = soup.select("#feature")
        if (len(main_data_contents) > 0):
            ul_contents = main_data_contents[0].find_all("ul")
            if (len(ul_contents) > 0):
                li_details = ul_contents[0].find_all("li")
                if len(li_details) == 4:
                    intensity = li_details[3].text
                    coaster_page.intensity = intensity
        make_and_model_sections = soup.select("#feature > div.scroll > p")
        if (len(make_and_model_sections) > 0):
            make_and_model_contents =  make_and_model_sections[0].text
            # First, get the manufacturer (part between "Make:" and "Model:")
            make_and_model_parts = make_and_model_contents.split("Model:")
            manufacturer = make_and_model_parts[0].replace("Make:", "").strip()
            if (len(make_and_model_parts) == 1):
                model = "unknown"
            else:
                model_details = make_and_model_parts[1].strip()
                if "/" in model_details:
                    model = model_details.split("/")[1].strip()
                else:
                    model = model_details.strip()
            coaster_page.manufacturer = manufacturer
            coaster_page.model = model










def main():
    print("start")

    coaster_pages = []
    more_pages_exists = True
    page_number = 1
    while more_pages_exists:
        url = f"https://rcdb.com/r.htm?page={page_number}&ot=2"
        print(f"Loading page {page_number} with url {url}.  {len(coaster_pages)} records created.")
        page_number = page_number+1
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')
            table = soup.select('body > section > div.stdtbl.rer > table')
            for table_row in table[0].select('tr'):
                # ignore table header
                if len(table_row.find_all("td")) > 0:
                    coaster_page = CoasterPage.from_table_row(table_row)
                    if coaster_page.name != 'unknown' and coaster_page.status == 'Operating':
                        populate_details_from_page(coaster_page)
                        coaster_pages.append(coaster_page)
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}.  url: {url}")
            more_pages_exists = False

    conn = psycopg2.connect(
        host='cp-ai.cbsscwgeqp5j.us-east-2.rds.amazonaws.com',
        port=5432,
        database='postgres',
        user='postgres',
        password='CedarP0int'
    )

    print("records loaded.  beginning inserts.")
    cursor = conn.cursor();
    sql = """
          INSERT INTO roller_coasters (id, name, url, amusement_park, type, design, status, 
                                       manufacturer, model, length, height, drop, 
                                       inversion_count, speed, vertical_angle, duration, 
                                       restraints, g_force, intensity) 
          VALUES (%s, %s, %s, %s, %s, %s, %s, 
                  %s, %s, %s, %s, %s, 
                  %s, %s, %s, %s, 
                  %s, %s, %s) 
          """
    for coaster_page in coaster_pages:
        cursor.execute(sql, (
            coaster_page.id,
            coaster_page.name,
            coaster_page.url,
            coaster_page.amusement_park,
            coaster_page.coaster_type,
            coaster_page.design,
            coaster_page.status,
            coaster_page.manufacturer,
            coaster_page.model,
            coaster_page.length,
            coaster_page.height,
            coaster_page.drop,
            coaster_page.inversion_count,
            coaster_page.speed,
            coaster_page.vertical_angle,
            coaster_page.duration,
            coaster_page.restraints,
            coaster_page.g_force,
            coaster_page.intensity
        ))
        conn.commit()

if __name__ == "__main__":
    main()
