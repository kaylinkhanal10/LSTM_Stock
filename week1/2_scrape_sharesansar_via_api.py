import json
import os
import time
import re
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ---------------- CONFIG ----------------
OUTPUT_FILE = "sharesansar_corpus.json"
TARGET_COUNT = 5000

# You already have data AFTER this date; go older than this
START_DATE = datetime(2025, 11, 18)
END_DATE   = datetime(2024, 1, 1)   # safety lower bound

PAGE_WAIT = 1.0
ARTICLE_WAIT = 0.6

# ---------------- HELPERS ----------------
def load_existing_urls():
    if not os.path.exists(OUTPUT_FILE):
        return set()
    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(a["link"] for a in data if "link" in a)
    except:
        return set()

def save_article(article):
    data = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                pass
    data.append(article)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_iso_from_url(url):
    m = re.search(r'(\d{4}-\d{2}-\d{2})', url)
    return m.group(1) if m else None

# ---------------- SCRAPER ----------------
def scrape():
    driver = webdriver.Chrome()
    seen_urls = load_existing_urls()
    count = len(seen_urls)

    print(f"Resuming with {count} existing articles")

    try:
        cur_date = START_DATE

        while cur_date >= END_DATE and count < TARGET_COUNT:
            date_str = cur_date.strftime("%Y-%m-%d")
            list_url = f"https://www.sharesansar.com/category/latest?date={date_str}"

            print(f"\nFetching date: {date_str}")
            driver.get(list_url)

            try:
                WebDriverWait(driver, 8).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "featured-news-list"))
                )
            except:
                print("No articles on this date.")
                cur_date -= timedelta(days=1)
                continue

            blocks = driver.find_elements(By.CLASS_NAME, "featured-news-list")

            for b in blocks:
                if count >= TARGET_COUNT:
                    break

                # Robustly find the correct news link
                links = b.find_elements(By.TAG_NAME, "a")
                news_link = None
                for a in links:
                    href = a.get_attribute("href")
                    if href and "/newsdetail/" in href:
                        news_link = href
                        break

                if not news_link:
                    continue

                if news_link in seen_urls:
                    continue

                try:
                    date_text = b.find_element(By.CLASS_NAME, "text-org").text.strip()
                except:
                    date_text = None

                driver.get(news_link)
                try:
                    content_box = WebDriverWait(driver, 8).until(
                        EC.presence_of_element_located((By.ID, "newsdetail-content"))
                    )

                    title = driver.find_element(By.TAG_NAME, "h1").text.strip()
                    paragraphs = [
                        p.text.strip()
                        for p in content_box.find_elements(By.TAG_NAME, "p")
                        if p.text.strip()
                    ]

                    if not paragraphs:
                        driver.back()
                        continue

                    article = {
                        "title": title,
                        "link": news_link,
                        "source": "sharesansar",
                        "publishedText": date_text,
                        "publishedISO": extract_iso_from_url(news_link),
                        "content": "\n\n".join(paragraphs)
                    }

                    save_article(article)
                    seen_urls.add(news_link)
                    count += 1

                    print(f"[{count}/{TARGET_COUNT}] {title[:70]}")

                except Exception as e:
                    print(f"Failed article: {e}")

                driver.back()
                time.sleep(ARTICLE_WAIT)

            cur_date -= timedelta(days=1)
            time.sleep(PAGE_WAIT)

    finally:
        driver.quit()

    print(f"\nDONE. Total articles: {count}")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    scrape()
