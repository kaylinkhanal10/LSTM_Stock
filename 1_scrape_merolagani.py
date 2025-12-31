import requests
import json
import os
import time
from datetime import datetime
from bs4 import BeautifulSoup

OUT_FILE = "merolagani_corpus.json"
BASE = "https://eng.merolagani.com"
API = BASE + "/handlers/webrequesthandler.ashx"

TARGET_COUNT = 2500
PAGE_SIZE = 20   # you can safely raise to 30–50

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def save(article):
    data = []
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                pass
    data.append(article)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def parse_iso(ts):
    try:
        dt = datetime.fromisoformat(ts.replace("Z", ""))
        return dt.date().isoformat(), dt.isoformat()
    except:
        return None, None

def fetch_detail(news_id):
    url = f"{BASE}/NewsDetail.aspx?newsID={news_id}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "lxml")

    title = soup.select_one("#ctl00_ContentPlaceHolder1_newsTitle")
    date_el = soup.select_one("#ctl00_ContentPlaceHolder1_newsDate")

    body_blocks = soup.select(
        "#ctl00_ContentPlaceHolder1_newsOverview p, "
        "#ctl00_ContentPlaceHolder1_newsDetail p"
    )

    if not title or not body_blocks:
        return None

    content = [p.get_text(strip=True) for p in body_blocks if p.get_text(strip=True)]
    if len(content) < 2:
        return None

    return {
        "title": title.get_text(strip=True),
        "content": "\n\n".join(content),
        "publishedText": date_el.get_text(strip=True) if date_el else None,
        "link": url,
    }

def scrape():
    collected = 0
    page = 1
    seen_ids = set()

    while collected < TARGET_COUNT:
        params = {
            "type": "get_news",
            "newsID": 0,
            "newsCategoryID": 0,
            "symbol": "",
            "page": page,
            "pageSize": PAGE_SIZE,
            "popular": "false",
            "includeFeatured": "true",
            "news": "#ctl00_ContentPlaceHolder1_txtNews",
            "languageType": "EN",
        }

        r = requests.get(API, params=params, headers=HEADERS, timeout=10)
        items = r.json()

        if not items:
            break

        for item in items:
            nid = item["newsID"]
            if nid in seen_ids:
                continue
            seen_ids.add(nid)

            iso_date, iso_ts = parse_iso(item["newsDateAD"])

            detail = fetch_detail(nid)
            if not detail:
                continue

            article = {
                "title": detail["title"],
                "link": detail["link"],
                "source": "merolagani",
                "publishedText": detail["publishedText"],
                "publishedISO": iso_date,
                "publishedTimestamp": iso_ts,
                "content": detail["content"],
            }

            save(article)
            collected += 1
            print(f"[{collected}] {detail['title'][:70]}")

            if collected >= TARGET_COUNT:
                break

            time.sleep(0.3)

        page += 1

    print(f"Done. Total articles: {collected}")

if __name__ == "__main__":
    scrape()
