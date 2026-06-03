"""
Loads Electron launch event data.
Primary source: local CSV (electron_launches_clean.csv from existing repo assets).
Auto-refreshes from Wikipedia if CSV is absent or older than LAUNCHES_REFRESH_DAYS.

Output DataFrame columns: date (DatetimeIndex), success (bool), mission_name (str).
"""

import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from config import LAUNCHES_CSV, LAUNCHES_REFRESH_DAYS


def _scrape_wikipedia_launches() -> pd.DataFrame:
    """
    Scrapes Electron launch list from Wikipedia using BeautifulSoup.
    Parses each wikitable row by column position to avoid rowspan/NaN issues
    that break pd.read_html on Wikipedia's merged-cell tables.
    """
    url = "https://en.wikipedia.org/wiki/List_of_Electron_launches"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = []

    for table in soup.find_all("table", class_="wikitable"):
        # Build header map from the first row containing <th> elements
        header_row = table.find("tr")
        if not header_row:
            continue
        headers = [th.get_text(" ", strip=True).lower() for th in header_row.find_all("th")]

        date_idx    = next((i for i, h in enumerate(headers) if "date" in h), None)
        outcome_idx = next((i for i, h in enumerate(headers) if "outcome" in h), None)
        name_idx    = next((i for i, h in enumerate(headers) if h == "name"), None)

        if date_idx is None or outcome_idx is None:
            continue

        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) <= max(date_idx, outcome_idx):
                continue

            # Date cell may contain "25 May 2017 05:00" — grab date part only
            raw_date = cells[date_idx].get_text(" ", strip=True).split()[0:3]
            date = pd.to_datetime(" ".join(raw_date), errors="coerce")
            if pd.isna(date):
                continue

            outcome = cells[outcome_idx].get_text(" ", strip=True).lower()
            if any(w in outcome for w in ("planned", "tbd", "tbc", "upcoming")):
                continue

            success = "success" in outcome and "partial" not in outcome and "failure" not in outcome
            mission = cells[name_idx].get_text(" ", strip=True).strip('"') if name_idx and name_idx < len(cells) else ""
            rows.append({"date": date.normalize(), "success": success, "mission_name": mission})

    df = pd.DataFrame(rows).drop_duplicates(subset="date").sort_values("date")
    return df.reset_index(drop=True)


def _is_stale() -> bool:
    """Returns True if the CSV is missing or older than LAUNCHES_REFRESH_DAYS."""
    if not LAUNCHES_CSV.exists():
        return True
    age_days = (time.time() - LAUNCHES_CSV.stat().st_mtime) / 86400
    return age_days > LAUNCHES_REFRESH_DAYS


def load_launches(force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns launch DataFrame with columns: date, success, mission_name.
    date is a tz-naive Timestamp, normalized to midnight.

    Auto-refreshes from Wikipedia when the CSV is older than LAUNCHES_REFRESH_DAYS.
    Pass force_refresh=True to re-scrape immediately.
    """
    if force_refresh or _is_stale():
        reason = "forced refresh" if force_refresh else f"CSV older than {LAUNCHES_REFRESH_DAYS} days"
        print(f"Refreshing launch data ({reason}) — scraping Wikipedia...")
        fresh = _scrape_wikipedia_launches()
        LAUNCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
        fresh.to_csv(LAUNCHES_CSV, index=False)
        return fresh

    if LAUNCHES_CSV.exists():
        df = pd.read_csv(LAUNCHES_CSV)
        # Normalise expected column names gracefully
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        date_col = next((c for c in df.columns if "date" in c), None)
        success_col = next((c for c in df.columns if "success" in c or "outcome" in c or "result" in c), None)
        mission_col = next((c for c in df.columns if "mission" in c or "name" in c), None)

        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
        out["success"] = df[success_col].astype(str).str.lower().str.contains("success") if success_col else True
        out["mission_name"] = df[mission_col] if mission_col else ""
        out = out.dropna(subset=["date"]).drop_duplicates(subset="date").sort_values("date")
        return out.reset_index(drop=True)

    print("Launch CSV not found — scraping Wikipedia...")
    df = _scrape_wikipedia_launches()
    df.to_csv(LAUNCHES_CSV, index=False)
    return df


if __name__ == "__main__":
    df = load_launches()
    print(df.tail(10).to_string())
    print(f"\nTotal launches: {len(df)}")
