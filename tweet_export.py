"""Load tweet text from local JSON, JSONL, NDJSON, or CSV exports."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

TEXT_FIELDS = ("text", "full_text", "tweetText", "tweet_text", "content", "body")
LIST_FIELDS = ("tweets", "items", "data", "results")
MAX_EXPORT_BYTES = 5 * 1024 * 1024
MAX_EXPORT_RECORDS = 10_000


def load_tweet_texts(file_bytes: bytes, filename: str = "") -> list[str]:
    """Extract tweet text from a JSON, JSONL, NDJSON, or CSV export."""
    if len(file_bytes) > MAX_EXPORT_BYTES:
        raise ValueError("The uploaded export exceeds the 5 MB limit.")

    try:
        raw_text = file_bytes.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("The uploaded export must use UTF-8 encoding.") from exc

    if not raw_text:
        raise ValueError("The uploaded export is empty.")

    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        records = list(csv.DictReader(StringIO(raw_text)))
    else:
        records = _load_json_records(raw_text)

    if len(records) > MAX_EXPORT_RECORDS:
        raise ValueError("The uploaded export exceeds the 10,000-record limit.")

    texts = [_extract_text(record) for record in records]
    cleaned = [text for text in texts if text]
    if not cleaned:
        raise ValueError("No tweet text fields were found in the uploaded export.")
    return cleaned


def _load_json_records(raw_text: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        records: list[dict[str, Any]] = []
        try:
            for line in raw_text.splitlines():
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        records.append(item)
        except json.JSONDecodeError as exc:
            raise ValueError("The uploaded export is not valid JSON or JSON Lines.") from exc
        return records

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if isinstance(payload, dict):
        for field in LIST_FIELDS:
            value = payload.get(field)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]

    return []


def _extract_text(record: dict[str, Any]) -> str:
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
