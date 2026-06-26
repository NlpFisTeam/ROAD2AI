#!/usr/bin/env python3
"""Build optional effectiveness sidecar for TVPL parquet documents."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from document_filters import parse_metadata_date
from ingest_parquet_to_qdrant import resolve_data_dir

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "effectiveness.parquet"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"


def fetch_vietlex_by_number(document_number: str) -> dict | None:
    query = urllib.parse.quote(document_number)
    search_url = f"https://vietlex.vn/api/v1/search?q={query}&limit=20"
    req = urllib.request.Request(search_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    target = document_number.strip().upper()
    doc_id = None
    for item in payload.get("results", []):
        if (item.get("soHieu") or "").strip().upper() == target:
            doc_id = item.get("id")
            break
    if not doc_id:
        return None

    detail_url = f"https://vietlex.vn/api/v1/document/{doc_id}"
    req = urllib.request.Request(detail_url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        detail = json.loads(resp.read().decode("utf-8")).get("document") or {}

    return {
        "eff_code": detail.get("hieuLucCode") or "",
        "eff_status": detail.get("hieuLuc") or "",
        "effective_date": detail.get("ngayBanHanh") or "",
        "expiry_date": "",
        "source": "vietlex",
        "source_id": doc_id,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data/effectiveness.parquet sidecar")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--input-metadata", type=Path, default=None, help="Pre-filtered metadata parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between API calls")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)

    if args.input_metadata and args.input_metadata.is_file():
        meta = pd.read_parquet(args.input_metadata)
    else:
        meta = pd.read_parquet(data_dir / "metadata.parquet")

    if args.limit:
        meta = meta.head(args.limit)

    rows: list[dict] = []
    total = len(meta)
    print(f"Fetching effectiveness for {total:,} documents...")

    for idx, row in enumerate(meta.itertuples(index=False), start=1):
        row_dict = row._asdict()
        doc_id = int(row_dict["id"])
        document_number = str(row_dict.get("document_number") or "").strip()
        record = {
            "id": doc_id,
            "document_number": document_number,
            "eff_code": "",
            "eff_status": "",
            "effective_date": "",
            "expiry_date": "",
            "source": "unknown",
            "source_id": "",
        }
        if document_number:
            try:
                fetched = fetch_vietlex_by_number(document_number)
                if fetched:
                    record.update(fetched)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"  [{idx}/{total}] id={doc_id} lỗi API: {exc}")
        rows.append(record)
        if idx % 50 == 0 or idx == total:
            print(f"  [{idx}/{total}] fetched")
        time.sleep(args.sleep)

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    matched = (out["source"] == "vietlex").sum()
    print(f"Đã lưu {len(out):,} rows → {args.output} (matched VietLex: {matched:,})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
