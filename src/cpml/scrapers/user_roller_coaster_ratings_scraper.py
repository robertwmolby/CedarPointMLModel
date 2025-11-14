# crowd_level_scraper.py

import psycopg2
import requests
from bs4 import BeautifulSoup

import os
import psycopg2
from psycopg2.extensions import connection as PGConnection

def get_db_connection() -> PGConnection | None:
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        return conn

    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None


def main():
    print("start")

    conn = get_db_connection()
    cursor = conn.cursor()

    sql = """
          INSERT INTO user_coaster_ratings_export (user_name, coaster_name, park_name, rating) 
          VALUES (%s, %s, %s, %s) 
          """


    more_users_exists = True
    user_number = 23552
    failure_count = 0
    while more_users_exists:
        user_number = user_number + 1
        database_records = []
        user_url = f"https://captaincoaster.com/en/users/{user_number}/ratings/1?sort=r.value&direction=desc"
        response = requests.get(user_url)
        if response.status_code == 200:
            page_number = 1
            soup = BeautifulSoup(response.text, 'lxml')
            user_name_headers = soup.select("body > div.page-container > div > div.content-wrapper > div.page-header.page-header-default > div > div.page-title.row > h1")
            if len(user_name_headers) > 0:
                user_name = user_name_headers[0].text.lower().replace(" ","_")
                user_name = user_name.replace("_-_ratings","")
                print(f"User name is {user_name}")
            else:
                continue
            print(f"User {user_number}: {user_name}")
            page_link_section = soup.select("body > div.page-container > div > div.content-wrapper > div.content > div.text-center.content-group.pt-20 > ul")
            if len(page_link_section) != 0:
                li_sections = page_link_section[0].find_all("li")
                if len(li_sections) > 0:
                    last_li_section  = li_sections[len(li_sections)-2]
                    last_page_anchor = last_li_section.find_all("a")
                    if len(last_page_anchor) > 0:
                        page_count = last_page_anchor[0].text
                    else:
                        continue
                else:
                    continue
            else:
                continue
            if int(page_count) >= 10:
                print("skipping to next user since user has so many ratings")
                continue
            while page_number <= int(page_count):
                page_url = f"https://captaincoaster.com/en/users/{user_number}/ratings/{page_number}?sort=r.value&direction=desc"
                response = requests.get(page_url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'lxml')
                    page_number = page_number + 1
                    coaster_rows = soup.select('[id^="tr-coaster"]')
                    if len(coaster_rows) > 0:
                        for coaster_row in coaster_rows:
                            rating_table_links = coaster_row.find_all("a")
                            if len(rating_table_links) > 0:
                                roller_coast_name = rating_table_links[0].text
                            else:
                                print("Roller coast name not found")
                                continue
                            park_name_span = coaster_row.find("div", class_="text-muted")
                            if park_name_span is None:
                                print("Park name not found")
                                continue
                            else:
                                park_name = park_name_span.text.strip()
                            rating_value_div = coaster_row.find("div", class_="rating-coaster")
                            if rating_value_div is None:
                                print("no rating found")
                                continue
                            else:
                                rating = rating_value_div.get("storage-rateit-value")
                                output_line = roller_coast_name + " (" + park_name + ")"
                                output_line = output_line.rjust(80," ") + ": " + rating
                                print(output_line)
                                database_records.append((user_name, roller_coast_name, park_name, float(rating)))

        else:
            print(f"User status code is {response.status_code}")
            if response.status_code == 404:
                print(f"Failed to find user number {user_number} so jumping ahead 25.")
                user_number = user_number + 25
                failure_count = failure_count + 1
                if failure_count > 100:
                    more_users_exists = False
            else:
                print(f"Unexpected response code occurred.  Quitting.  Code: {response.status_code}.")
                more_users_exists = False
        if len(database_records) > 0:
            cursor.executemany(sql, database_records)
        conn.commit()
        user_number = user_number + 1
        more_users_exists = False


if __name__ == "__main__":
    main()
