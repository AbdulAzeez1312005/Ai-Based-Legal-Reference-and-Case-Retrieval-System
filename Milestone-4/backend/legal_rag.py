"""Generic ingestion and retrieval for a multi-document Indian legal RAG system."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

INGEST_VERSION = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
MIN_SEMANTIC_SCORE = 0.4
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx", ".html", ".htm"}
WORD_RE = re.compile(r"[a-z0-9]+", re.I)
STOP_WORDS = {
    "the", "and", "for", "what", "does", "say", "under", "according", "explain",
    "tell", "about", "with", "from", "this", "that", "when", "where", "which",
    "into", "have", "has", "can", "may", "shall", "are", "was", "were", "is",
    "provision", "section", "article", "rule", "regulation", "act", "code",
    "in", "of", "to", "by", "on", "as", "at", "did", "do", "it", "how",
}
LABELS = {
    "section": "section", "sections": "section", "sec": "section", "s": "section",
    "article": "article", "articles": "article", "art": "article",
    "rule": "rule", "rules": "rule", "regulation": "regulation",
    "regulations": "regulation", "reg": "regulation", "secton": "section", "secion": "section", "clause": "clause",
    "clauses": "clause", "order": "order", "orders": "order",
}
NUMBER = r"(?:\d+[A-Za-z]*(?:\s*-\s*[A-Za-z0-9]+)?|[IVXLCDM]{1,8})"
EXPLICIT_HEADING = re.compile(
    rf"(?im)^[ \t]*(?P<label>sections?|secs?\.?|s\.|articles?|arts?\.?|"
    rf"rules?|regulations?|regs?\.?|clauses?|orders?)\s+"
    rf"(?P<number>{NUMBER})(?![A-Za-z0-9])"
    rf"(?:\s*\(\s*(?P<subsection>\d+[A-Za-z]?)\s*\))?"
    rf"\s*[\.\):\-–—]?\s*(?P<title>[^\r\n]{{0,180}})"
)
BARE_HEADING = re.compile(
    rf"(?m)^[ \t]*(?:\d+\[)?(?P<number>\d+[A-Za-z]?(?:\s*-\s*[A-Za-z0-9]+)?)"
    rf"\s*[\.\)]\s+(?P<title>[A-Z][^\r\n]{{2,170}})"
)
QUERY_REFERENCE = re.compile(
    rf"\b(?P<label>sections?|secs?\.?|s\.|articles?|arts?\.?|rules?|"
    rf"regulations?|regs?\.?|clauses?|orders?)\s*(?:no\.?\s*)?"
    rf"(?:no\.?|number)?\s*(?P<number>{NUMBER})(?![A-Za-z0-9])"
    rf"(?:\s*\(\s*(?P<subsection>\d+[A-Za-z]?)\s*\))?",
    re.I,
)
BARE_QUERY_REFERENCE = re.compile(
    rf"\b(?:sec(?:tion)?|art(?:icle)?)\s*\.?\s*(?P<number>{NUMBER})"
    rf"(?![A-Za-z0-9])(?:\s*\(\s*(?P<subsection>\d+[A-Za-z]?)\s*\))?",
    re.I,
)


def normalize_number(value: str) -> str:
    return re.sub(r"[\s.]", "", str(value or "")).upper()


def extract_provision_references(question: str) -> list[dict[str, str]]:
    """Extract explicit provision labels and lists without knowing any Act's sections."""
    refs: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in QUERY_REFERENCE.finditer(question or ""):
        kind = LABELS.get(match.group("label").lower().rstrip("."))
        if not kind:
            continue
        item = (kind, normalize_number(match.group("number")),
                (match.group("subsection") or "").upper())
        if item not in seen:
            seen.add(item)
            refs.append({"type": item[0], "number": item[1], "subsection": item[2]})
    if not refs:
        for match in BARE_QUERY_REFERENCE.finditer(question or ""):
            kind = "article" if "art" in match.group(0).lower() else "section"
            item = (kind, normalize_number(match.group("number")),
                    (match.group("subsection") or "").upper())
            if item not in seen:
                seen.add(item)
                refs.append({"type": item[0], "number": item[1], "subsection": item[2]})
    if refs:
        list_match = re.search(r"\bsections?\b", question or "", re.I)
        if list_match:
            suffix = (question or "")[list_match.end():]
            for match in re.finditer(
                rf"(?:,|\band\b|\bor\b)\s*(?:sections?\s*)?"
                rf"(?P<number>{NUMBER})(?:\s*\(\s*(?P<sub>\d+[A-Za-z]?)\s*\))?",
                suffix, re.I,
            ):
                item = ("section", normalize_number(match.group("number")),
                        (match.group("sub") or "").upper())
                if item not in seen:
                    seen.add(item)
                    refs.append({"type": item[0], "number": item[1], "subsection": item[2]})
    return refs


def detect_document_type(source: str, text: str = "") -> str:
    """Classify each source broadly from its own name/content."""
    name = Path(source).stem.lower().replace("_", " ").replace("-", " ")
    sample = (text or "")[:12000].lower()
    if re.search(r"\b(vs?\.?|versus)\b|judg(?:e)?ments?|case study|landmark.*judg", name):
        return "case"
    if re.search(r"\b(definitions?|glossary|reference|study notes?)\b", name):
        return "reference"
    if "constitution" in name:
        return "constitution"
    if re.search(r"\b(act|code|statute|rules|regulations?)\b", name):
        return "statute"
    if re.search(r"\b(the .{2,80} act,? \d{4}|code of .{2,80},? \d{4})\b", sample[:8000]):
        return "statute"
    if re.search(r"case study|in the (?:supreme|high) court|judgment|judgement", sample[:1600]):
        return "case"
    return "reference"

def _label_type(label: str) -> str | None:
    return LABELS.get(label.lower().rstrip("."))


def find_provision_headings(text: str, document_type: str) -> list[dict[str, Any]]:
    """Detect explicit legal labels and numbered headings, including Constitution schedules."""
    if document_type not in {"statute", "constitution"}:
        return []
    candidates: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    schedule_names = (
        r"(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|"
        r"TENTH|ELEVENTH|TWELFTH|[0-9]+(?:ST|ND|RD|TH)?|[IVXLCDM]{1,8})"
    )
    schedule_re = re.compile(
        rf"(?im)^[ \t]*(?:THE[ \t]+)?(?P<schedule>"
        rf"(?:{schedule_names}[ \t]+SCHEDULE|SCHEDULE[ \t]+{schedule_names})"
        rf")\b[^\r\n]*"
    )
    schedule_matches = list(schedule_re.finditer(text))
    schedule_start = None
    for i, marker in enumerate(schedule_matches):
        previous_gap = (marker.start() - schedule_matches[i - 1].start()
                        if i else float("inf"))
        next_gap = (schedule_matches[i + 1].start() - marker.start()
                    if i + 1 < len(schedule_matches) else float("inf"))
        # Contents pages list schedule titles close together. The first real
        # schedule heading is separated from that list and starts substantial text.
        if previous_gap > 3500 and next_gap > 2500:
            schedule_start = marker.start()
            break

    for match in EXPLICIT_HEADING.finditer(text):
        kind = _label_type(match.group("label"))
        number = normalize_number(match.group("number"))
        if kind and number:
            candidates.append({
                "start": match.start(), "type": kind, "number": number,
                "title": (match.group("title") or "").strip(" \t:.-\u2013\u2014"),
            })
            occupied.append((match.start(), match.end()))
    for match in BARE_HEADING.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        number = normalize_number(match.group("number"))
        title = (match.group("title") or "").strip()
        if not title or re.search(r"\.{3,}|_{3,}", title):
            continue
        in_schedule = (
            document_type == "constitution" and schedule_start is not None
            and match.start() >= schedule_start
        )
        candidate: dict[str, Any] = {
            "start": match.start("number"),
            "type": ("paragraph" if in_schedule else
                     "article" if document_type == "constitution" else "section"),
            "number": number,
            "title": title,
        }
        if in_schedule:
            preceding = [item for item in schedule_matches
                         if schedule_start <= item.start() <= match.start()]
            if preceding:
                candidate["context"] = preceding[-1].group("schedule").upper()
        candidates.append(candidate)
    candidates.sort(key=lambda item: item["start"])
    lengths: list[int] = []
    for i, item in enumerate(candidates):
        end = candidates[i + 1]["start"] if i + 1 < len(candidates) else len(text)
        lengths.append(end - item["start"])
    best: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
    for item, length in zip(candidates, lengths):
        key = (item["type"], item["number"], str(item.get("context", "")))
        if key not in best or length > best[key][0]:
            best[key] = (length, item)
    chosen = [item for length, item in best.values() if length >= 35]
    return sorted(chosen, key=lambda item: item["start"])

def _chunk_text(text: str, size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> Iterable[tuple[int, str]]:
    """Split on paragraph/sentence boundaries when possible and retain character offsets."""
    value = re.sub(r"\r\n?", "\n", text or "").strip()
    start = 0
    while start < len(value):
        tentative = min(start + size, len(value))
        end = tentative
        if tentative < len(value):
            floor = start + int(size * 0.68)
            boundary = max(
                value.rfind("\n\n", floor, tentative),
                value.rfind("\n", floor, tentative),
                value.rfind(". ", floor, tentative),
                value.rfind("; ", floor, tentative),
            )
            if boundary > floor:
                end = boundary + (2 if value[boundary:boundary + 2] in ("\n\n", ". ", "; ") else 1)
        raw = value[start:end]
        chunk = raw.strip()
        if chunk:
            leading = len(raw) - len(raw.lstrip())
            yield start + leading, chunk
        if end >= len(value):
            break
        start = max(start + 1, end - overlap)


def _page_for_offset(offset: int, page_starts: list[tuple[int, int]]) -> int:
    page = 1
    for start, page_number in page_starts:
        if start > offset:
            break
        page = page_number
    return page


def _normalise_pdf_text(value: str) -> str:
    value = re.sub(r"[\u00ad\u200b\ufffd]", "", value or "")
    value = re.sub(r"(?<=\w)-\n(?=\w)", "", value)
    return re.sub(r"\r\n?", "\n", value)


def load_file_pages(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return [_normalise_pdf_text(page.extract_text() or "")
                for page in PdfReader(str(path)).pages]
    if suffix == ".txt":
        return [path.read_text(encoding="utf-8", errors="replace")]
    if suffix == ".docx":
        import docx2txt
        return [docx2txt.process(str(path)) or ""]
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        return [soup.get_text("\n", strip=True)]
    return []


def _record_id(source: str, provision_id: str, chunk_index: int, text: str) -> str:
    identity = "\0".join((source, provision_id, str(chunk_index), text))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def records_for_file(path: Path) -> list[dict[str, Any]]:
    """Parse a source into stable, page-cited chunks with dynamic provision metadata."""
    pages = load_file_pages(path)
    if not pages:
        return []
    page_starts: list[tuple[int, int]] = []
    page_parts: list[str] = []
    cursor = 0
    for page_number, page in enumerate(pages, start=1):
        page_starts.append((cursor, page_number))
        page_parts.append(page)
        cursor += len(page) + 2
    full_text = "\n\n".join(page_parts).strip()
    if not full_text:
        return []
    source = path.name
    doc_type = detect_document_type(source, full_text)
    document_id = hashlib.sha256(source.casefold().encode("utf-8")).hexdigest()[:20]
    headings = find_provision_headings(full_text, doc_type)
    segments: list[tuple[int, int, dict[str, Any] | None]] = []
    if headings:
        first = headings[0]["start"]
        if first > 0 and len(full_text[:first].strip()) >= 250:
            segments.append((0, first, None))
        for position, heading in enumerate(headings):
            end = headings[position + 1]["start"] if position + 1 < len(headings) else len(full_text)
            if end - heading["start"] >= 35:
                segments.append((heading["start"], end, heading))
    else:
        segments.append((0, len(full_text), None))

    records: list[dict[str, Any]] = []
    for segment_start, segment_end, heading in segments:
        segment = full_text[segment_start:segment_end].strip()
        if not segment:
            continue
        kind = heading["type"] if heading else ""
        number = heading["number"] if heading else ""
        provision_id = f"{document_id}:{kind}:{number}" if heading else f"{document_id}:document"
        if heading and heading.get("context"):
            context_id = re.sub(r"\W+", "-", str(heading["context"]).upper()).strip("-")
            provision_id += f":{context_id}"
        pieces = list(_chunk_text(segment))
        for chunk_index, (relative_start, chunk) in enumerate(pieces):
            page = _page_for_offset(segment_start + relative_start, page_starts)
            metadata: dict[str, Any] = {
                "text": chunk, "source": source, "document_id": document_id,
                "document_type": doc_type, "page": int(page),
                "chunk_index": int(chunk_index), "chunk_count": int(len(pieces)),
                "provision_id": provision_id, "ingest_version": INGEST_VERSION,
                "content_hash": hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            }
            if heading:
                metadata.update({
                    "provision_type": kind, "provision_number": number,
                    "provision_title": str(heading["title"])[:240],
                })
                if heading.get("context"):
                    metadata["provision_context"] = str(heading["context"])[:120]
            records.append({
                "id": _record_id(source, provision_id, chunk_index, chunk),
                "text": chunk, "metadata": metadata,
            })
    return records


def records_for_folder(folder: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Process supported top-level documents and return a per-file ingestion report."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Legal data directory does not exist: {folder}")
    all_records: list[dict[str, Any]] = []
    report: dict[str, Any] = {"files": [], "skipped": [], "records": 0}
    for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            file_records = records_for_file(path)
            if not file_records:
                report["skipped"].append({"source": path.name, "reason": "No extractable text"})
                continue
            provision_ids = {
                record["metadata"]["provision_id"] for record in file_records
                if record["metadata"].get("provision_type")
            }
            report["files"].append({
                "source": path.name,
                "document_type": file_records[0]["metadata"]["document_type"],
                "records": len(file_records), "provisions": len(provision_ids),
            })
            all_records.extend(file_records)
        except Exception as exc:
            report["skipped"].append({"source": path.name, "reason": str(exc)})
    report["records"] = len(all_records)
    return all_records, report


def _response_matches(response: Any) -> list[Any]:
    return (response.get("matches", []) or [] if isinstance(response, dict)
            else getattr(response, "matches", []) or [])


def _match_value(match: Any, key: str, default: Any = None) -> Any:
    return match.get(key, default) if isinstance(match, dict) else getattr(match, key, default)


def _metadata(match: Any) -> dict[str, Any]:
    return dict(_match_value(match, "metadata", {}) or {})


def _query(index: Any, embedding: list[float], namespace: str,
           top_k: int, metadata_filter: dict[str, Any] | None = None) -> list[Any]:
    args: dict[str, Any] = {
        "vector": embedding, "top_k": max(1, min(int(top_k), 1000)),
        "include_metadata": True, "namespace": namespace or "",
    }
    if metadata_filter:
        args["filter"] = metadata_filter
    return _response_matches(index.query(**args))


def _tokens(value: str) -> set[str]:
    result: set[str] = set()
    for token in WORD_RE.findall((value or "").lower()):
        if token in STOP_WORDS:
            continue
        if token.endswith("ies") and len(token) > 5:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        result.add(token)
    return result


def _source_title_matches(question: str, source: str) -> bool:
    """Recognize a named source from its title or a title-derived abbreviation."""
    source_words = WORD_RE.findall(Path(source).stem.lower())
    significant = [
        word for word in source_words
        if word not in {"the", "of", "and", "act", "india"}
        and not word.isdigit()
    ]
    query_terms = {term for term in _tokens(question) if not term.isdigit()}
    overlap = query_terms & set(significant)
    if overlap and len(overlap) >= min(2, len(set(significant))):
        return True

    for token in re.findall(r"\b[A-Z][A-Za-z]*\b", question):
        parts = re.findall(r"[A-Z][a-z]*", token)
        if len(parts) < 2 or sum(char.isupper() for char in token) < 2:
            continue
        available = list(dict.fromkeys(source_words))

        def assign_part(position: int, remaining: list[str]) -> bool:
            if position == len(parts):
                return True
            prefix = parts[position].lower()
            return any(
                word.startswith(prefix)
                and assign_part(position + 1, remaining[:index] + remaining[index + 1:])
                for index, word in enumerate(remaining)
            )

        if assign_part(0, available):
            return True
    return False


def _is_constitution_source(source: str) -> bool:
    return bool(re.search(r"\bconstitution\b", Path(source).stem, re.I))


def _relevance_score(question: str, labels: str, body: str, score: float) -> float:
    terms = {term for term in _tokens(question) if not term.isdigit()}
    if not terms:
        return score
    label_overlap = len(terms & _tokens(labels)) / len(terms)
    body_overlap = len(terms & _tokens(body)) / len(terms)
    return 0.35 * score + 0.85 * label_overlap + body_overlap


def _merge_chunks(matches: list[Any]) -> dict[str, Any]:
    ordered = sorted(matches, key=lambda item: int(_metadata(item).get("chunk_index", 0)))
    combined = ""
    for item in ordered:
        piece = str(_metadata(item).get("text", "")).strip()
        if not piece:
            continue
        limit = min(260, len(combined), len(piece))
        overlap = next((n for n in range(limit, 0, -1) if combined.endswith(piece[:n])), 0)
        combined += piece[overlap:]
    first = _metadata(ordered[0]) if ordered else {}
    pages = sorted({
        int(_metadata(item).get("page", first.get("page", 1)))
        for item in ordered
    })
    first["page"] = pages[0] if pages else 1
    first["pages"] = pages
    return {
        "page_content": combined, "metadata": first,
        "score": max((float(_match_value(item, "score", 0.0) or 0.0) for item in ordered), default=0.0),
        "chunks": len(ordered),
    }


def _exact_provision_groups(index: Any, embedding: list[float], namespace: str,
                            question: str, ref: dict[str, str]) -> list[dict[str, Any]]:
    provision_filter = {
        "ingest_version": INGEST_VERSION,
        "provision_type": ref["type"],
        "provision_number": ref["number"],
    }
    hits = _query(index, embedding, namespace, 1000, provision_filter)
    groups: dict[tuple[str, str], list[Any]] = {}
    for hit in hits:
        meta = _metadata(hit)
        groups.setdefault((str(meta.get("source", "")),
                           str(meta.get("provision_id", ""))), []).append(hit)
    named_sources = {
        source for source, _provision_id in groups
        if _source_title_matches(question, source)
    }
    if named_sources:
        groups = {
            key: items for key, items in groups.items()
            if key[0] in named_sources
        }
    elif ref["type"] == "article":
        constitution_groups = {
            key: items for key, items in groups.items()
            if _is_constitution_source(key[0])
        }
        if constitution_groups:
            groups = constitution_groups
    query_terms = {term for term in _tokens(question) if not term.isdigit()}
    ranked: list[tuple[float, tuple[str, str], list[Any]]] = []
    for key, items in groups.items():
        metas = [_metadata(item) for item in items]
        labels = " ".join(
            str(meta.get("provision_title", "")) + " " + str(meta.get("source", ""))
            for meta in metas
        )
        body = " ".join(str(meta.get("text", "")) for meta in metas)
        semantic_score = max(
            float(_match_value(item, "score", 0.0) or 0.0) for item in items
        )
        rank_score = _relevance_score(question, labels, body, semantic_score)
        ranked.append((rank_score, key, items))
    ranked.sort(key=lambda row: row[0], reverse=True)
    if not ranked:
        return []
    broad = bool(re.search(r"\b(all|across|compare|difference|different acts|each)\b", question, re.I))
    context_terms = {term for term in query_terms if not term.isdigit()}
    chosen = ranked[:3] if broad or (not context_terms and len(ranked) > 1) else ranked[:1]
    results = []
    for _score, key, candidates in chosen:
        meta = _metadata(candidates[0])
        complete_filter = {
            "ingest_version": INGEST_VERSION,
            "document_id": str(meta.get("document_id", "")),
            "provision_id": key[1],
        }
        complete = _query(index, embedding, namespace, 1000, complete_filter)
        results.append(_merge_chunks(complete or candidates))
    return results

def _semantic_results(question: str, index: Any, embedding: list[float],
                      namespace: str, top_k: int) -> list[dict[str, Any]]:
    hits = _query(index, embedding, namespace, max(top_k * 3, top_k),
                  {"ingest_version": INGEST_VERSION})
    if not hits:
        # Old vectors are kept searchable during transition until generic ingestion runs.
        hits = _query(index, embedding, namespace, max(top_k * 3, top_k))
    hits = [
        hit for hit in hits
        if not (
            _metadata(hit).get("document_type") == "constitution"
            and not _is_constitution_source(str(_metadata(hit).get("source", "")))
        )
    ]
    hits.sort(key=lambda item: _relevance_score(
        question,
        str(_metadata(item).get("source", "")) + " "
        + str(_metadata(item).get("provision_title", "")),
        str(_metadata(item).get("text", "")),
        float(_match_value(item, "score", 0.0) or 0.0),
    ), reverse=True)
    selected: dict[str, Any] = {}
    for hit in hits:
        meta = _metadata(hit)
        key = str(_match_value(hit, "id", "")) or (
            str(meta.get("source", "")) + ":" + str(meta.get("chunk_index", ""))
        )
        selected.setdefault(key, hit)
        if len(selected) >= top_k:
            break
    expanded = []
    for hit in selected.values():
        meta = _metadata(hit)
        chunk_index = meta.get("chunk_index")
        document_id = str(meta.get("document_id", ""))
        if document_id and isinstance(chunk_index, (int, float)):
            neighbor_filter = {"document_id": document_id}
            if meta.get("ingest_version") == INGEST_VERSION:
                neighbor_filter["ingest_version"] = INGEST_VERSION
            provision_id = str(meta.get("provision_id", ""))
            if provision_id:
                neighbor_filter["provision_id"] = provision_id
            candidates = _query(
                index, embedding, namespace, 1000, neighbor_filter
            )
            lower, upper = int(chunk_index) - 1, int(chunk_index) + 1
            neighbors = [
                candidate for candidate in candidates
                if isinstance(_metadata(candidate).get("chunk_index"), (int, float))
                and lower <= int(_metadata(candidate)["chunk_index"]) <= upper
            ]
            neighbors.sort(
                key=lambda item: float(_match_value(item, "score", 0.0) or 0.0),
                reverse=True,
            )
            neighbors = neighbors[:10]
            if neighbors:
                expanded.append(_merge_chunks(neighbors))
                continue
        expanded.append(_merge_chunks([hit]))
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for result in expanded:
        meta = result["metadata"]
        key = (str(meta.get("source", "")), str(meta.get("provision_id", "")))
        if key in unique:
            current = unique[key]
            if result["page_content"] not in current["page_content"]:
                current["page_content"] += "\n\n" + result["page_content"]
            current["score"] = max(current["score"], result["score"])
        else:
            unique[key] = result
    ranked = sorted(
        unique.values(),
        key=lambda row: _relevance_score(
            question,
            str(row["metadata"].get("source", "")) + " "
            + str(row["metadata"].get("provision_title", "")),
            row["page_content"],
            float(row["score"]),
        ),
        reverse=True,
    )[:top_k]
    return [
        row for row in ranked
        if float(row["score"]) >= MIN_SEMANTIC_SCORE
    ]


def retrieve_legal_context(question: str, index: Any, embedder: Any,
                           namespace: str = "", top_k: int = 6) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return complete explicitly requested provisions or semantically relevant windows."""
    encoded = embedder.encode([question], normalize_embeddings=True)
    vector = encoded[0].tolist() if hasattr(encoded[0], "tolist") else list(encoded[0])
    refs = extract_provision_references(question)
    results: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    ambiguous = False
    for ref in refs:
        groups = _exact_provision_groups(index, vector, namespace, question, ref)
        if not groups:
            trace.append({"mode": "provision-not-indexed",
                          "provision": f'{ref["type"].title()} {ref["number"]}'})
            continue
        context_terms = {
            term for term in _tokens(question)
            if not term.isdigit()
        }
        broad = bool(re.search(
            r"\b(all|across|compare|difference|different acts|each)\b",
            question, re.I,
        ))
        if ref["type"] != "article" and not context_terms and len(groups) > 1 and not broad:
            ambiguous = True
            trace.append({
                "mode": "ambiguous-provision",
                "provision": f'{ref["type"].title()} {ref["number"]}',
                "sources": [row["metadata"].get("source") for row in groups],
            })
            continue
        for result in groups:
            meta = result["metadata"]
            results.append(result)
            trace.append({
                "mode": "complete-provision",
                "provision": f'{ref["type"].title()} {ref["number"]}',
                "source": meta.get("source"), "page": meta.get("page"),
                "pages": meta.get("pages", [meta.get("page")]),
                "chunks": result["chunks"], "score": round(result["score"], 4),
            })
    if not results and not ambiguous:
        results = _semantic_results(
            question, index, vector, namespace, max(1, int(top_k))
        )
        trace.extend({
            "mode": "semantic-window", "source": row["metadata"].get("source"),
            "page": row["metadata"].get("page"),
            "pages": row["metadata"].get("pages", [row["metadata"].get("page")]),
            "chunks": row["chunks"],
            "score": round(row["score"], 4),
        } for row in results)
    return results, trace
