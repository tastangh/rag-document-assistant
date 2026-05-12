from __future__ import annotations

import re
from html import unescape
from typing import Dict, List, Tuple

import cv2
import numpy as np
from bs4 import BeautifulSoup


def normalize_markdown_tables(markdown_text: str) -> str:
    if not markdown_text:
        return ""
    table_pattern = re.compile(r"<table[\s\S]*?</table>", flags=re.IGNORECASE)

    def _replace(match: re.Match[str]) -> str:
        html_table = match.group(0)
        table_md = html_table_to_markdown(html_table)
        if not table_md:
            return ""
        return sanitize_markdown_table(table_md)

    normalized = table_pattern.sub(_replace, markdown_text)
    normalized = strip_html_wrappers(normalized)
    normalized = cleanup_markdown_whitespace(normalized)
    return normalized.strip()


def sanitize_markdown_table(table_md: str) -> str:
    lines = [ln.rstrip() for ln in table_md.splitlines() if ln.strip()]
    if len(lines) < 2:
        return table_md.strip()
    data_rows = lines[2:] if len(lines) > 2 else []
    if not data_rows:
        return table_md.strip()

    cells: List[str] = []
    for row in data_rows:
        if not row.startswith("|"):
            continue
        parts = [cell.strip() for cell in row.strip().strip("|").split("|")]
        cells.extend(parts)
    if not cells:
        return table_md.strip()

    total_cells = len(cells)
    empty_cells = sum(1 for c in cells if not c)
    short_cells = sum(1 for c in cells if 0 < len(c) <= 1)
    alnum_chars = sum(sum(ch.isalnum() for ch in c) for c in cells)
    total_chars = sum(len(c) for c in cells)

    empty_ratio = empty_cells / max(total_cells, 1)
    short_ratio = short_cells / max(total_cells, 1)
    alnum_ratio = alnum_chars / max(total_chars, 1)
    noisy = (
        (empty_ratio > 0.65 and len(data_rows) >= 3)
        or (short_ratio > 0.55 and len(data_rows) >= 4)
        or (alnum_ratio < 0.22 and len(data_rows) >= 4)
    )
    if not noisy:
        return table_md.strip()

    flattened_cells = [c for c in cells if c and len(c) > 1]
    if not flattened_cells:
        return ""
    compact_text = " ".join(flattened_cells)
    compact_text = re.sub(r"\s{2,}", " ", compact_text).strip()
    return compact_text


def strip_html_wrappers(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"</?(html|body)\b[^>]*>", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?div\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    return unescape(cleaned)


def cleanup_markdown_whitespace(text: str) -> str:
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def score_text_quality(text: str) -> float:
    if not text:
        return float("-inf")
    lowered = text.lower()
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    spaces = sum(ch.isspace() for ch in text)
    bad_chars = text.count("ï¿½")
    pipes = text.count("|")
    score = letters + (0.5 * digits) + (0.2 * spaces) - (20.0 * bad_chars) - (0.1 * pipes)
    if "<html" in lowered or "<body" in lowered:
        score -= 150.0
    if "<div" in lowered:
        score -= 50.0
    return score


def deskew_grayscale(binary_image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(binary_image < 250))
    if coords.size == 0 or coords.shape[0] < 800:
        return binary_image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.15 or abs(angle) > 20:
        return binary_image
    h, w = binary_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        binary_image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    max_side = max(h, w)
    if 0 < max_side < 1800:
        scale = 1800.0 / float(max_side)
        gray = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    binary = cv2.medianBlur(binary, 3)
    binary = deskew_grayscale(binary)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def html_table_to_markdown(html_table: str) -> str:
    if not html_table or "<table" not in html_table.lower():
        return ""
    soup = BeautifulSoup(html_table, "html.parser")
    table = soup.find("table")
    if table is None:
        return ""
    rows = table.find_all("tr")
    if not rows:
        return ""

    parsed_rows: List[List[str]] = []
    pending_rowspan: Dict[int, Tuple[int, str]] = {}
    for tr in rows:
        row_cells: List[str] = []
        col_idx = 0
        while col_idx in pending_rowspan:
            remain, value = pending_rowspan[col_idx]
            row_cells.append(value)
            if remain > 1:
                pending_rowspan[col_idx] = (remain - 1, value)
            else:
                pending_rowspan.pop(col_idx, None)
            col_idx += 1

        for cell in tr.find_all(["th", "td"]):
            while col_idx in pending_rowspan:
                remain, value = pending_rowspan[col_idx]
                row_cells.append(value)
                if remain > 1:
                    pending_rowspan[col_idx] = (remain - 1, value)
                else:
                    pending_rowspan.pop(col_idx, None)
                col_idx += 1

            text = " ".join(cell.get_text(" ", strip=True).split())
            colspan = max(int(cell.get("colspan", 1) or 1), 1)
            rowspan = max(int(cell.get("rowspan", 1) or 1), 1)
            row_cells.append(text)
            for _ in range(colspan - 1):
                row_cells.append("")
            if rowspan > 1:
                base_col = col_idx
                for offset in range(colspan):
                    pending_rowspan[base_col + offset] = (rowspan - 1, "")
            col_idx += colspan

        while col_idx in pending_rowspan:
            remain, value = pending_rowspan[col_idx]
            row_cells.append(value)
            if remain > 1:
                pending_rowspan[col_idx] = (remain - 1, value)
            else:
                pending_rowspan.pop(col_idx, None)
            col_idx += 1
        parsed_rows.append(row_cells)

    max_cols = max(len(r) for r in parsed_rows)
    normalized_rows = [r + [""] * (max_cols - len(r)) for r in parsed_rows]
    if max_cols == 0:
        return ""
    header = normalized_rows[0]
    separator = ["---"] * max_cols
    body = normalized_rows[1:] if len(normalized_rows) > 1 else []
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    md_lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(md_lines).strip()

