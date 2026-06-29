"""
fetch_jst_data.py
-----------------
Downloads the Jordà-Schularick-Taylor (JST) Macrohistory Database R6
and saves it as a CSV to data/raw/jst_dataset.csv.

The JST dataset covers 18 developed countries from 1870–2020 with annual
nominal and real returns for equities, long-term government bonds, and
short-term bills, plus CPI and exchange rates vs USD.

This is the closest publicly-available proxy to the dataset used in:
  Anarkulova, Cederburg, O'Doherty (2023) "Beyond the Status Quo"
which used the proprietary Global Financial Data (GFDatabase).

Usage:
    python fetch_jst_data.py

If the automatic download fails, manually download from:
    https://www.macrohistory.net/database/
and place the file at: data/raw/jst_dataset.csv
"""

import os
import sys
import urllib.request
import pandas as pd

# Known download locations for JST R6 dataset (try in order)
JST_DOWNLOAD_URLS = [
    # Official JST dataset download (CSV version, when available)
    "https://www.macrohistory.net/app/download/9834512049/JSTdatasetR6.xlsx",
    # Alternative mirror / older version
    "https://data.macrohistory.net/JSTdatasetR6.xlsx",
]

RAW_DATA_DIR = "data/raw"
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "jst_dataset.csv")
RAW_XLSX = os.path.join(RAW_DATA_DIR, "jst_dataset.xlsx")


def download_jst(output_path: str = RAW_XLSX) -> bool:
    """Attempt to download JST dataset. Returns True on success."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for url in JST_DOWNLOAD_URLS:
        try:
            print(f"Trying: {url}")
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; research-replication-script/1.0)"
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
            with open(output_path, "wb") as f:
                f.write(data)
            print(f"✓ Downloaded to {output_path}")
            return True
        except Exception as e:
            print(f"  Failed ({e})")

    return False


def xlsx_to_csv(xlsx_path: str, csv_path: str) -> bool:
    """Convert the downloaded XLSX to CSV."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name="Data")
        df.to_csv(csv_path, index=False)
        print(f"✓ Converted to {csv_path} ({len(df)} rows)")
        return True
    except Exception as e:
        # Try without sheet_name
        try:
            df = pd.read_excel(xlsx_path)
            df.to_csv(csv_path, index=False)
            print(f"✓ Converted to {csv_path} ({len(df)} rows)")
            return True
        except Exception as e2:
            print(f"  Conversion failed: {e2}")
            return False


def check_existing() -> bool:
    """Returns True if a usable CSV already exists."""
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE, nrows=5)
            if "iso" in df.columns or "country" in df.columns:
                print(f"✓ Found existing {OUTPUT_FILE} — skipping download.")
                return True
        except Exception:
            pass
    return False


if __name__ == "__main__":
    if check_existing():
        sys.exit(0)

    print("=" * 60)
    print("JST Macrohistory Database Fetcher")
    print("=" * 60)

    # Step 1: Download
    if not os.path.exists(RAW_XLSX):
        ok = download_jst(RAW_XLSX)
    else:
        print(f"✓ Raw XLSX already present at {RAW_XLSX}")
        ok = True

    if not ok:
        print()
        print("Automatic download failed. Please:")
        print("  1. Visit: https://www.macrohistory.net/database/")
        print("  2. Download the latest JST dataset (Excel or Stata format).")
        print(f"  3. Place the file at: {RAW_XLSX}")
        print("  4. Re-run this script.")
        sys.exit(1)

    # Step 2: Convert to CSV
    if not xlsx_to_csv(RAW_XLSX, OUTPUT_FILE):
        print("Conversion failed. Try opening the XLSX manually and saving as CSV.")
        sys.exit(1)

    print()
    print("Done. Run process_historical_data.py next to build the bootstrap dataset.")
