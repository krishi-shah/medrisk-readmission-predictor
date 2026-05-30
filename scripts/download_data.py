"""Download the UCI Diabetes 130-US Hospitals dataset.

The dataset is hosted on the UCI ML Repository. This script fetches the
zip archive, extracts the CSV, and places it at the path expected by
config.yaml (data/raw/diabetic_data.csv).

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "raw" / "diabetic_data.csv"

# UCI ML Repository direct download URL for dataset #296
_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
_INNER_CSV = "diabetic_data.csv"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"Dataset already present at {dest} — skipping download.")
        return

    print(f"Downloading dataset from UCI ML Repository …")
    print(f"  URL: {url}")

    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            raw_bytes = response.read()
    except Exception as exc:
        print(f"\nERROR: Could not download dataset: {exc}", file=sys.stderr)
        print(
            "\nManual download instructions:",
            "\n  1. Visit https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008",
            "\n  2. Download the zip file",
            f"\n  3. Extract 'diabetic_data.csv' to {dest}",
            file=sys.stderr,
        )
        sys.exit(1)

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        names = zf.namelist()
        csv_name = next((n for n in names if n.endswith(_INNER_CSV)), None)
        if csv_name is None:
            print(
                f"ERROR: Could not find '{_INNER_CSV}' inside the downloaded zip.",
                f"Available files: {names}",
                file=sys.stderr,
            )
            sys.exit(1)
        dest.write_bytes(zf.read(csv_name))

    print(f"Dataset saved to {dest}  ({dest.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    download(_DATASET_URL, _OUTPUT_PATH)
