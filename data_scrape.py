import json
import os
import time
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

OUTPUT_FILE = "sharesansar_corpus.json"
TARGET_COUNT = 2500

def save_to_json(new_data, filename=OUTPUT_FILE):
    """Appends a single article to the JSON list safely."""
    data = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []

    data.append(new_data)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def scrape_incrementally():
    driver = webdriver.Chrome()
    driver.get("https://www.sharesansar.com/category/latest")
    
    count = 0
    try:
        while count < TARGET_COUNT:
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "featured-news-list")))
            
            # Record current page links and their displayed dates
            blocks = driver.find_elements(By.CLASS_NAME, "featured-news-list")
            
            # We store link and date preview from the list page
            batch = []
            for b in blocks:
                link = b.find_element(By.TAG_NAME, "a").get_attribute("href")
                # ShareSansar usually has the date in a <span class="text-org">
                try:
                    date_text = b.find_element(By.CLASS_NAME, "text-org").text.strip()
                except:
                    date_text = None
                batch.append((link, date_text))

            for url, date_text in batch:
                if count >= TARGET_COUNT: break
                
                driver.get(url)
                try:
                    content_box = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.ID, "newsdetail-content"))
                    )
                    
                    # Regex to find YYYY-MM-DD in URL if the text-org is missing
                    iso_date = None
                    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', url)
                    if date_match:
                        iso_date = date_match.group(1)

                    article = {
                        "title": driver.find_element(By.TAG_NAME, "h1").text.strip(),
                        "link": url,
                        "source": "sharesansar",
                        "publishedText": date_text, # e.g., "Tuesday, Dec 30, 2025"
                        "publishedISO": iso_date,   # e.g., "2025-12-30"
                        "content": "\n\n".join([p.text for p in content_box.find_elements(By.TAG_NAME, "p") if p.text.strip()])
                    }
                    
                    save_to_json(article)
                    count += 1
                    print(f"Saved {count}/{TARGET_COUNT}: {article['publishedISO']} | {article['title'][:40]}...")
                    
                except Exception as e:
                    print(f"Error on {url}: {e}")
                
                driver.back()
                time.sleep(1)

            # Pagination
            try:
                next_btn = driver.find_element(By.XPATH, "//a[@rel='next']")
                next_btn.click()
                time.sleep(2)
            except:
                print("No more pages.")
                break
    finally:
        driver.quit()

if __name__ == "__main__":
    scrape_incrementally()