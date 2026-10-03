from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".zip", ".py", ".sql", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_CONTENT_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".csv": {"text/csv", "application/csv", "text/plain", "application/vnd.ms-excel", "application/octet-stream"},
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/octet-stream"},
    ".zip": {"application/zip", "application/x-zip-compressed", "application/octet-stream"},
    ".py": {"text/x-python", "text/python", "text/plain", "application/x-python-code", "application/octet-stream"},
    ".sql": {"application/sql", "application/x-sql", "text/sql", "text/plain", "application/octet-stream"},
    ".png": {"image/png", "application/octet-stream"},
    ".jpg": {"image/jpeg", "application/octet-stream"},
    ".jpeg": {"image/jpeg", "application/octet-stream"},
    ".webp": {"image/webp", "application/octet-stream"},
    ".bmp": {"image/bmp", "image/x-ms-bmp", "application/octet-stream"},
    ".tif": {"image/tiff", "application/octet-stream"},
    ".tiff": {"image/tiff", "application/octet-stream"},
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _sourced_text(parts: list[tuple[str, dict[str, Any]]], separator: str = "\n") -> tuple[str, list[dict[str, Any]]]:
    """Keep transient character spans for new chunks without storing duplicate text."""
    text_parts: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    for value, source in parts:
        value = value.replace("\x00", "").strip()
        if text_parts:
            cursor += len(separator)
        start = cursor
        text_parts.append(value)
        cursor += len(value)
        if value and source and start < 2_000_000:
            spans.append({"start": start, "end": min(cursor, 2_000_000), **source})
    return separator.join(text_parts), spans


def validate_attachment_type(filename: str, content_type: str) -> None:
    suffix = Path(filename).suffix.casefold()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported files are PDF, DOCX, TXT, CSV, XLSX, ZIP, PY, SQL, PNG, JPG, WEBP, BMP, and TIFF.")
    normalized_type = (content_type or "application/octet-stream").split(";", 1)[0].strip().casefold()
    if normalized_type not in SUPPORTED_CONTENT_TYPES[suffix]:
        raise ValueError("The file type does not match its extension.")


def extract_text(filename: str, content: bytes) -> tuple[str, dict[str, Any]]:
    suffix = Path(filename).suffix.casefold()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported files are PDF, DOCX, TXT, CSV, XLSX, ZIP, PY, SQL, PNG, JPG, WEBP, BMP, and TIFF.")
    if suffix in {".docx", ".xlsx", ".zip"}:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                entries = archive.infolist()
                if len(entries) > 5000 or sum(item.file_size for item in entries) > 50_000_000:
                    raise ValueError("The compressed document expands beyond the safe extraction limit.")
        except zipfile.BadZipFile as exc:
            raise ValueError("The document archive is invalid or corrupt.") from exc
    if suffix == ".zip":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            image_entries = [
                item for item in archive.infolist()
                if not item.is_dir() and Path(item.filename).suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            ]
            class_names = sorted({
                Path(item.filename).parts[-2]
                for item in image_entries if len(Path(item.filename).parts) >= 2
            })
        text = f"Image dataset archive metadata: {len(image_entries)} image files across {len(class_names)} class folders."
        metadata = {
            "format": "zip", "dataset_kind": "image_archive", "image_files": len(image_entries),
            "class_folders": class_names[:100], "inert_binary": True,
        }
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        from PIL import Image

        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            metadata = {
                "format": "image", "extension": suffix[1:], "width": width,
                "height": height, "mode": image.mode, "inert_binary": True,
            }
        text = f"Image attachment metadata: {width} by {height} pixels, format {suffix[1:].upper()}."
    elif suffix in {".txt", ".py", ".sql"}:
        text = content.decode("utf-8-sig", errors="replace")
        metadata = {"format": suffix[1:], "inert_source": suffix in {".py", ".sql"}}
    elif suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages = [(page.extract_text() or "") for page in reader.pages[:100]]
        text, spans = _sourced_text([(value, {"page": index}) for index, value in enumerate(pages, 1)], "\n\n")
        metadata = {"format": "pdf", "pages": len(reader.pages), "pages_extracted": len(pages)}
        metadata["_source_spans"] = spans
        title = str((reader.metadata or {}).get("/Title") or "").strip()
        if title:
            metadata["title"] = title[:240]
    elif suffix == ".docx":
        from docx import Document

        document = Document(io.BytesIO(content))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        table_rows = [
            " | ".join(_clean(cell.text) for cell in row.cells)
            for table in document.tables for row in table.rows
        ]
        section = ""
        parts: list[tuple[str, dict[str, Any]]] = []
        for index, paragraph in enumerate(document.paragraphs, 1):
            if not paragraph.text.strip():
                continue
            if str(paragraph.style.name or "").casefold().startswith("heading"):
                section = paragraph.text.strip()[:160]
            parts.append((paragraph.text, {"paragraph": index, **({"section": section} if section else {})}))
        parts.extend((row, {}) for row in table_rows)
        text, spans = _sourced_text(parts)
        metadata = {
            "format": "docx", "paragraphs": len(document.paragraphs),
            "tables": len(document.tables), "table_rows": len(table_rows),
        }
        metadata["_source_spans"] = spans
    elif suffix == ".csv":
        frame = pd.read_csv(io.BytesIO(content))
        frame = frame.iloc[:, :100]
        rows = [" | ".join(_clean(value) for value in frame.columns)]
        rows.extend(" | ".join(_clean(value) for value in row) for row in frame.fillna("").itertuples(index=False, name=None))
        parts = [(rows[0], {})]
        parts.extend(("\n".join(rows[start:start + 20]), {"row_start": start + 1, "row_end": min(start + 20, len(rows))})
                     for start in range(1, len(rows), 20))
        text, spans = _sourced_text(parts)
        metadata = {
            "format": "csv", "rows": len(frame), "columns": [str(value) for value in frame.columns],
            "sample_values": frame.head(5).fillna("").astype(str).to_dict(orient="records"),
        }
        metadata["_source_spans"] = spans
    else:
        from openpyxl import load_workbook

        row_counts: dict[str, int] = {}
        workbook_dimensions = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
        try:
            row_counts = {
                str(sheet.title): max(0, int(sheet.max_row or 0) - 1)
                for sheet in workbook_dimensions.worksheets[:50]
            }
        finally:
            workbook_dimensions.close()
        workbook = pd.ExcelFile(io.BytesIO(content))
        sheet_details: list[dict[str, Any]] = []
        source_parts: list[tuple[str, dict[str, Any]]] = []
        for sheet_name in workbook.sheet_names[:50]:
            frame = workbook.parse(sheet_name=sheet_name, nrows=5000).iloc[:, :100]
            sheet_details.append({
                "name": str(sheet_name), "rows": row_counts.get(str(sheet_name), len(frame)),
                "rows_extracted": len(frame),
                "columns": [str(value) for value in frame.columns],
                "sample_values": frame.head(5).fillna("").astype(str).to_dict(orient="records"),
            })
            rows = [f"Sheet: {sheet_name}", " | ".join(_clean(value) for value in frame.columns)]
            rows.extend(" | ".join(_clean(value) for value in row) for row in frame.fillna("").itertuples(index=False, name=None))
            if source_parts:
                source_parts.append(("", {}))
            source_parts.append(("\n".join(rows[:2]), {"sheet": str(sheet_name)}))
            source_parts.extend(("\n".join(rows[start:start + 20]),
                                 {"sheet": str(sheet_name), "row_start": start,
                                  "row_end": min(start + 19, len(rows) - 1)})
                                for start in range(2, len(rows), 20))
        workbook.close()
        text, spans = _sourced_text(source_parts)
        metadata = {"format": "xlsx", "sheet_names": workbook.sheet_names, "sheets": sheet_details}
        metadata["_source_spans"] = spans
    raw_text = text.replace("\x00", "")
    leading = len(raw_text) - len(raw_text.lstrip())
    normalized = raw_text.strip()
    if metadata.get("_source_spans"):
        metadata["_source_spans"] = [
            {**span, "start": max(0, span["start"] - leading), "end": min(len(normalized), span["end"] - leading)}
            for span in metadata["_source_spans"] if span["end"] > leading and span["start"] - leading < len(normalized)
        ]
    if not normalized:
        raise ValueError("No readable text was found in this file.")
    return normalized[:2_000_000], metadata


def chunk_text(text: str, size: int = 1400, overlap: int = 180) -> list[str]:
    return [item["content"] for item in chunk_text_with_offsets(text, size, overlap)]


def chunk_text_with_offsets(text: str, size: int = 1400, overlap: int = 180) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(text) and len(chunks) < 1500:
        end = min(len(text), cursor + size)
        if end < len(text):
            boundary = max(text.rfind("\n", cursor + size // 2, end), text.rfind(". ", cursor + size // 2, end))
            if boundary > cursor:
                end = boundary + 1
        chunk = text[cursor:end].strip()
        if chunk:
            chunks.append({"content": chunk, "start": cursor, "end": end})
        if end >= len(text):
            break
        cursor = max(cursor + 1, end - overlap)
    return chunks


def source_for_chunk(start: int, end: int, spans: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [(min(end, item["end"]) - max(start, item["start"]), item)
               for item in spans if item["start"] < end and item["end"] > start]
    if not matches:
        return {}
    dominant = max(matches, key=lambda pair: pair[0])[1]
    source = {key: value for key, value in dominant.items() if key not in {"start", "end"}}
    if len({item["sheet"] for _, item in matches if item.get("sheet")}) > 1:
        return {}
    if len({item["section"] for _, item in matches if item.get("section")}) > 1:
        source.pop("section", None)
    if "page" in source:
        pages = [item["page"] for _, item in matches if "page" in item]
        source = {"page_start": min(pages), "page_end": max(pages)}
    elif "row_start" in source:
        same_sheet = [item for _, item in matches if item.get("sheet") == source.get("sheet") and "row_start" in item]
        source["row_start"] = min(item["row_start"] for item in same_sheet)
        source["row_end"] = max(item["row_end"] for item in same_sheet)
    return source


def source_title(item: dict[str, Any]) -> str:
    filename = str(item.get("filename") or "document")
    source = item.get("source") or {}
    if "page_start" in source:
        start, end = source["page_start"], source["page_end"]
        return f"{filename} · Page {start}" if start == end else f"{filename} · Pages {start}-{end}"
    if source.get("section"):
        return f"{filename} · Section: {source['section']}"
    if source.get("paragraph"):
        return f"{filename} · Paragraph {source['paragraph']}"
    parts = [filename]
    if source.get("sheet"):
        parts.append(f"Sheet: {source['sheet']}")
    if "row_start" in source:
        parts.append(f"Rows {source['row_start']}-{source['row_end']}")
    if len(parts) > 1:
        return " · ".join(parts)
    index = item.get("chunk_index")
    return f"{filename} · Chunk {index + 1}" if isinstance(index, int) and index >= 0 else filename


def assemble_source_evidence(chunks: list[dict[str, Any]], max_chars: int = 18000) -> tuple[str, list[dict[str, str]]]:
    """Cite only passages that actually fit in the supplied model context."""
    if not chunks:
        return "", []
    per_chunk_chars = max(200, 17000 // len(chunks) - 250)
    passages: list[str] = []
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    used = 0
    for item in chunks:
        title = source_title(item)
        heading = f"{title[:180]}:\n"
        separator = 2 if passages else 0
        available = max_chars - used - separator - len(heading)
        if available <= 0:
            break
        content = str(item.get("content") or "")[:min(per_chunk_chars, available)]
        if not content.strip():
            continue
        passages.append(heading + content)
        used += separator + len(heading) + len(content)
        citation_key = (str(item.get("attachment_id")), title)
        if citation_key not in seen:
            seen.add(citation_key)
            anchor = "metadata" if item.get("kind") == "metadata" else f"chunk-{item['chunk_index']}"
            citations.append({"title": title, "url": f"attachment:{item['attachment_id']}#{anchor}"})
    return "\n\n".join(passages), citations
