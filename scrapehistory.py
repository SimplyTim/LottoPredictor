"""Build a historical Lotto Plus CSV from currently reachable result sources.

Source strategy:
1. LotteryExtreme provides month-by-month draw history with draw numbers from
   July 2001 through early 2025.
2. Lottery Guru provides the current 2025-2026 history but without draw
   numbers. We align its overlap with LotteryExtreme and continue numbering
   sequentially into the latest draw.

This keeps the repository usable even while the older NLCB history endpoint is
timing out.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass, asdict, replace
from datetime import date, datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DRAW_DATE_FORMAT = "%d-%b-%y"
LOTTERYEXTREME_URL = "https://www.lotteryextreme.com/trinidad-and-tobago/lottoplus-results"
LOTTERYGURU_URL = (
    "https://lotteryguru.com/trinidad-and-tobago-lottery-results/"
    "tt-lotto-plus/tt-lotto-plus-results-history"
)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

LOTTERYEXTREME_HEADER_RE = re.compile(
    r"^Lotto Plus\s+(\d{4}-\d{2}-\d{2})\s+[A-Za-z]+\s+\((\d+)\)$"
)
LOTTERYGURU_DAY_RE = re.compile(r"^(Wednesday|Saturday)$")
LOTTERYGURU_DATE_RE = re.compile(r"^\d{2}\s+[A-Za-z]{3}$")
LOTTERYGURU_PAGEINFO_RE = re.compile(r'id="pageInfo"[^>]*lastPage="(\d+)"')


@dataclass(frozen=True)
class DrawRecord:
    draw_number: int
    date: str
    jackpot: float | None
    main_1: int
    main_2: int
    main_3: int
    main_4: int
    main_5: int
    powerball: int
    multiplier: int | None
    winners: int | None

    @property
    def signature(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.main_1,
            self.main_2,
            self.main_3,
            self.main_4,
            self.main_5,
            self.powerball,
        )

    def to_csv_row(self) -> dict[str, object]:
        row = asdict(self)
        return {
            "Draw Number": row["draw_number"],
            "Date": row["date"],
            "Jackpot": row["jackpot"],
            "1": row["main_1"],
            "2": row["main_2"],
            "3": row["main_3"],
            "4": row["main_4"],
            "5": row["main_5"],
            "PB": row["powerball"],
            "Multiplier": row["multiplier"],
            "Winners": row["winners"],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="draws.csv", help="CSV output path.")
    parser.add_argument("--sleep-seconds", type=float, default=0.05, help="Delay between requests.")
    return parser.parse_args()


def clean_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    return [line.strip() for line in soup.get_text("\n", strip=True).split("\n") if line.strip()]


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def iter_months() -> list[tuple[int, int]]:
    start = date(2001, 7, 1)
    today = date.today()
    months: list[tuple[int, int]] = []

    year, month = start.year, start.month
    while (year, month) <= (today.year, today.month):
        months.append((year, month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return months


def parse_lotteryextreme_month(html: str) -> list[DrawRecord]:
    lines = clean_lines(html)
    if "No draws found for this month." in lines:
        return []
    records_by_key: dict[tuple[str, tuple[int, int, int, int, int, int]], DrawRecord] = {}

    line_index = 0
    while line_index < len(lines):
        match = LOTTERYEXTREME_HEADER_RE.match(lines[line_index])
        if not match:
            line_index += 1
            continue

        iso_date, draw_number_text = match.groups()
        numeric_lines = lines[line_index + 1 : line_index + 7]
        if len(numeric_lines) < 6 or not all(token.isdigit() for token in numeric_lines):
            line_index += 1
            continue

        numbers = [int(token) for token in numeric_lines]
        draw_date = datetime.strptime(iso_date, "%Y-%m-%d").strftime(DRAW_DATE_FORMAT)
        record = DrawRecord(
            draw_number=int(draw_number_text),
            date=draw_date,
            jackpot=None,
            main_1=numbers[0],
            main_2=numbers[1],
            main_3=numbers[2],
            main_4=numbers[3],
            main_5=numbers[4],
            powerball=numbers[5],
            multiplier=None,
            winners=None,
        )
        key = (record.date, record.signature)
        existing = records_by_key.get(key)
        if existing is None or record.draw_number < existing.draw_number:
            records_by_key[key] = record
        line_index += 7

    return sorted(records_by_key.values(), key=lambda item: item.draw_number)


def scrape_lotteryextreme_history(sleep_seconds: float) -> list[DrawRecord]:
    session = build_session()
    records_by_draw: dict[int, DrawRecord] = {}

    for year, month in iter_months():
        year_month = f"{year}-{month:02d}"
        print(f"LotteryExtreme {year_month}...")
        response = session.post(
            LOTTERYEXTREME_URL,
            data={"mode": "month", "year_month": year_month},
            timeout=(20, 60),
        )
        response.raise_for_status()
        for record in parse_lotteryextreme_month(response.text):
            records_by_draw[record.draw_number] = record
        time.sleep(sleep_seconds)

    session.close()
    return repair_lotteryextreme_number_anomalies(list(records_by_draw.values()))


def repair_lotteryextreme_number_anomalies(records: list[DrawRecord]) -> list[DrawRecord]:
    by_date = sorted(records, key=lambda item: datetime.strptime(item.date, DRAW_DATE_FORMAT))
    repaired: list[DrawRecord] = []

    for index, record in enumerate(by_date):
        updated_record = record
        if 0 < index < len(by_date) - 1:
            previous_number = repaired[-1].draw_number
            next_number = by_date[index + 1].draw_number
            if record.draw_number > next_number and next_number == previous_number + 2:
                updated_record = replace(record, draw_number=previous_number + 1)
        repaired.append(updated_record)

    deduped_by_draw = {record.draw_number: record for record in repaired}
    return sorted(deduped_by_draw.values(), key=lambda item: item.draw_number)


def parse_lotteryguru_page(html: str) -> list[DrawRecord]:
    lines = clean_lines(html)
    history_indices = [index for index, line in enumerate(lines) if line == "History Results"]
    if history_indices:
        lines = lines[history_indices[-1] + 1 :]

    for marker in ("Previous", "Winning Numbers", "Home"):
        if marker in lines:
            lines = lines[: lines.index(marker)]
            break

    parsed: list[DrawRecord] = []

    line_index = 0
    while line_index < len(lines) - 8:
        if not LOTTERYGURU_DAY_RE.match(lines[line_index]):
            line_index += 1
            continue
        if not LOTTERYGURU_DATE_RE.match(lines[line_index + 1]) or not lines[line_index + 2].isdigit():
            line_index += 1
            continue

        numeric_lines = lines[line_index + 3 : line_index + 9]
        if len(numeric_lines) < 6 or not all(token.isdigit() for token in numeric_lines):
            line_index += 1
            continue

        draw_date = datetime.strptime(
            f"{lines[line_index + 1]} {lines[line_index + 2]}",
            "%d %b %Y",
        ).strftime(DRAW_DATE_FORMAT)
        numbers = [int(token) for token in numeric_lines]
        parsed.append(
            DrawRecord(
                draw_number=-1,
                date=draw_date,
                jackpot=None,
                main_1=numbers[0],
                main_2=numbers[1],
                main_3=numbers[2],
                main_4=numbers[3],
                main_5=numbers[4],
                powerball=numbers[5],
                multiplier=None,
                winners=None,
            )
        )
        line_index += 9

    return parsed


def scrape_lotteryguru_history(sleep_seconds: float) -> list[DrawRecord]:
    session = build_session()
    initial_response = session.get(LOTTERYGURU_URL, timeout=(20, 60))
    initial_response.raise_for_status()
    page_match = LOTTERYGURU_PAGEINFO_RE.search(initial_response.text)
    if page_match is None:
        raise RuntimeError("Could not determine Lottery Guru pagination.")

    last_page = int(page_match.group(1))
    all_records: list[DrawRecord] = []

    for page_number in range(1, last_page + 1):
        page_url = LOTTERYGURU_URL if page_number == 1 else f"{LOTTERYGURU_URL}?page={page_number}"
        print(f"LotteryGuru page {page_number}/{last_page}...")
        response = session.get(page_url, timeout=(20, 60))
        response.raise_for_status()
        all_records.extend(parse_lotteryguru_page(response.text))
        time.sleep(sleep_seconds)

    session.close()

    deduped: dict[tuple[str, tuple[int, int, int, int, int, int]], DrawRecord] = {}
    for record in all_records:
        deduped[(record.date, record.signature)] = record

    return sorted(
        deduped.values(),
        key=lambda item: datetime.strptime(item.date, DRAW_DATE_FORMAT),
    )


def assign_lotteryguru_draw_numbers(
    lotteryextreme_records: list[DrawRecord],
    lotteryguru_records: list[DrawRecord],
) -> list[DrawRecord]:
    extreme_sorted = sorted(lotteryextreme_records, key=lambda item: item.draw_number)
    guru_sorted = sorted(
        lotteryguru_records,
        key=lambda item: datetime.strptime(item.date, DRAW_DATE_FORMAT),
    )

    guru_signatures = [record.signature for record in guru_sorted]
    extreme_signatures = [record.signature for record in extreme_sorted]

    anchor_index: int | None = None
    overlap_length = min(8, len(guru_signatures))
    for start_index in range(len(extreme_signatures) - overlap_length + 1):
        if extreme_signatures[start_index : start_index + overlap_length] == guru_signatures[:overlap_length]:
            anchor_index = start_index
            break

    if anchor_index is None:
        raise RuntimeError("Could not align Lottery Guru history with LotteryExtreme draw numbers.")

    anchor_draw_number = extreme_sorted[anchor_index].draw_number
    assigned: list[DrawRecord] = []
    for offset, record in enumerate(guru_sorted):
        assigned.append(
            DrawRecord(
                draw_number=anchor_draw_number + offset,
                date=record.date,
                jackpot=record.jackpot,
                main_1=record.main_1,
                main_2=record.main_2,
                main_3=record.main_3,
                main_4=record.main_4,
                main_5=record.main_5,
                powerball=record.powerball,
                multiplier=record.multiplier,
                winners=record.winners,
            )
        )
    return assigned


def merge_histories(
    lotteryextreme_records: list[DrawRecord],
    lotteryguru_records: list[DrawRecord],
) -> list[DrawRecord]:
    merged: dict[int, DrawRecord] = {record.draw_number: record for record in lotteryextreme_records}
    for record in lotteryguru_records:
        merged[record.draw_number] = record
    return sorted(merged.values(), key=lambda item: item.draw_number)


def save_draws(records: list[DrawRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Draw Number",
        "Date",
        "Jackpot",
        "1",
        "2",
        "3",
        "4",
        "5",
        "PB",
        "Multiplier",
        "Winners",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_row())


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    lotteryextreme_records = scrape_lotteryextreme_history(sleep_seconds=args.sleep_seconds)
    lotteryguru_records = scrape_lotteryguru_history(sleep_seconds=args.sleep_seconds)
    numbered_lotteryguru_records = assign_lotteryguru_draw_numbers(
        lotteryextreme_records,
        lotteryguru_records,
    )
    merged_records = merge_histories(lotteryextreme_records, numbered_lotteryguru_records)
    save_draws(merged_records, output_path)

    print(f"Wrote {len(merged_records)} historical draws to {output_path}.")
    print(f"First draw: #{merged_records[0].draw_number} on {merged_records[0].date}")
    print(f"Latest draw: #{merged_records[-1].draw_number} on {merged_records[-1].date}")


if __name__ == "__main__":
    main()
