"""
Lightweight NLP enrichment for app_v25 (no spaCy).

- Normalize number words and domain synonyms
- Extract slide IDs by matching against the tile-server slide list (any format)
- Optional fuzzy HPC title → id lookup
- Validate plan entities against KB / slide list
"""

from __future__ import annotations

import re
from typing import Any

SLIDE_EXTENSIONS = (
    ".svs",
    ".ndpi",
    ".isyntax",
    ".tif",
    ".tiff",
    ".scn",
    ".mrxs",
    ".jpg",
    ".jpeg",
    ".png",
)

_UNITS: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def _build_number_words(max_value: int = 99) -> dict[str, int]:
    """All word forms from 0..max_value, including compounds (twenty-one, thirty two, …)."""
    out = dict(_UNITS)
    out.update(_TENS)
    for ten_word, ten_val in _TENS.items():
        if ten_val > max_value:
            continue
        for unit_word, unit_val in _UNITS.items():
            if unit_val == 0:
                continue
            n = ten_val + unit_val
            if n > max_value:
                continue
            out[f"{ten_word} {unit_word}"] = n
            out[f"{ten_word}-{unit_word}"] = n
    return out


# 0–99 in words (covers 21, 22, …, 31–39, … through at least 70 and beyond)
NUMBER_WORDS: dict[str, int] = _build_number_words(99)

_UNIT_ALT = "|".join(sorted((k for k in _UNITS if _UNITS[k] < 10), key=len, reverse=True))
_TEN_ALT = "|".join(sorted(_TENS.keys(), key=len, reverse=True))
_UNIT_1_9_ALT = "|".join(
    k for k, v in _UNITS.items() if 1 <= v <= 9
)

_COMPOUND_NUMBER_RE = re.compile(
    rf"\b({_TEN_ALT})[\s-]+({_UNIT_1_9_ALT})\b",
    re.I,
)
_SINGLE_NUMBER_RE = re.compile(
    rf"\b({_TEN_ALT}|{_UNIT_ALT})\b",
    re.I,
)

SYNONYM_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(whole\s+slide|wsi|section|case)\b", re.I), "slide"),
    (re.compile(r"\b(components?|clusters?|patterns?|subtypes?|classes?)\b", re.I), "hpc"),
    (re.compile(r"\b(patches?|regions?|crops?)\b", re.I), "tile"),
    (re.compile(r"\b(prognosis|outcomes?|hazard|kaplan)\b", re.I), "survival"),
    (re.compile(r"\b(tumou?rs?|cancerous|carcinoma)\b", re.I), "malignant"),
]


def _strip_punct(token: str) -> str:
    return token.strip("\"'`,.:;()[]{}").strip()


def _strip_slide_extension(token: str) -> str:
    low = token.lower()
    for ext in SLIDE_EXTENSIONS:
        if low.endswith(ext):
            return token[: -len(ext)]
    return token


def _phrase_to_int(phrase: str) -> int | None:
    key = phrase.lower().strip().replace("-", " ")
    if key in NUMBER_WORDS:
        return NUMBER_WORDS[key]
    parts = key.split()
    if len(parts) == 2:
        ten, unit = parts
        if ten in _TENS and unit in _UNITS:
            return _TENS[ten] + _UNITS[unit]
    if len(parts) == 1 and parts[0] in NUMBER_WORDS:
        return NUMBER_WORDS[parts[0]]
    return None


def normalise_number_words(text: str) -> str:
    if not text:
        return text

    def repl_compound(m: re.Match[str]) -> str:
        ten = m.group(1).lower()
        unit = m.group(2).lower()
        return str(_TENS[ten] + _UNITS[unit])

    out = _COMPOUND_NUMBER_RE.sub(repl_compound, text)

    def repl_single(m: re.Match[str]) -> str:
        val = _phrase_to_int(m.group(0))
        return str(val) if val is not None else m.group(0)

    out = _SINGLE_NUMBER_RE.sub(repl_single, out)

    def repl_after_hpc(m: re.Match[str]) -> str:
        phrase = m.group(1)
        val = _phrase_to_int(phrase.replace("-", " "))
        if val is None and m.lastindex and m.lastindex >= 2:
            ten, unit = m.group(1).lower(), m.group(2).lower()
            if ten in _TENS and unit in _UNITS:
                val = _TENS[ten] + _UNITS[unit]
        if val is not None:
            return f"hpc {val}"
        return m.group(0)

    out = re.sub(
        rf"\bhpc\s+({_TEN_ALT})[\s-]+({_UNIT_1_9_ALT})\b",
        repl_after_hpc,
        out,
        flags=re.I,
    )
    out = re.sub(
        rf"\bhpc\s+({_TEN_ALT}|{_UNIT_ALT})\b",
        repl_after_hpc,
        out,
        flags=re.I,
    )
    return out


def normalise_synonyms(text: str) -> str:
    if not text:
        return text
    out = text
    for pattern, replacement in SYNONYM_REPLACEMENTS:
        out = pattern.sub(replacement, out)
    return out


def enrich_prompt(text: str) -> str:
    return normalise_synonyms(normalise_number_words(text or ""))


def build_slide_index(slide_list: list[str]) -> dict[str, str]:
    return {s.lower(): s for s in slide_list if s}


def extract_slide_ids(text: str, slide_list: list[str]) -> list[str]:
    """Match tokens (and extension-stripped stems) against the live slide registry."""
    if not text or not slide_list:
        return []

    index = build_slide_index(slide_list)
    found: list[str] = []
    seen: set[str] = set()

    for raw in re.split(r"\s+", text):
        token = _strip_punct(raw)
        if not token:
            continue
        candidates = [token, _strip_slide_extension(token)]
        for cand in candidates:
            key = cand.strip().upper()
            hit = index.get(cand.lower()) or index.get(key.lower())
            if hit and hit not in seen:
                seen.add(hit)
                found.append(hit)

    if found:
        return found

    try:
        from rapidfuzz import fuzz, process

        choices = list(index.keys())
        for raw in re.split(r"\s+", text):
            token = _strip_punct(_strip_slide_extension(raw))
            if len(token) < 4:
                continue
            match = process.extractOne(token.lower(), choices, scorer=fuzz.ratio)
            if match and match[1] >= 90:
                sid = index[match[0]]
                if sid not in seen:
                    seen.add(sid)
                    found.append(sid)
    except ImportError:
        pass

    return found


def extract_hpc_ids_regex(text: str) -> list[str]:
    ids = re.findall(r"\bhpc[-_\s]*(\d+)\b", (text or "").lower(), re.I)
    if not ids:
        ids = re.findall(r"\bcomponent\s+(\d+)\b", (text or "").lower(), re.I)
    return list(dict.fromkeys(ids))


def hpc_id_from_title(text: str, hpc_title_map: dict[int, str], min_score: int = 80) -> int | None:
    if not text or not hpc_title_map:
        return None
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        return None

    titles = {hid: t for hid, t in hpc_title_map.items() if t}
    if not titles:
        return None

    q = text.lower()
    best_id = None
    best_score = 0
    for hid, title in titles.items():
        score = fuzz.partial_ratio(title.lower(), q)
        if score > best_score:
            best_score = score
            best_id = hid

    if best_score >= min_score and best_id is not None:
        return int(best_id)
    return None


def nlp_entity_hints(
    text: str,
    slide_list: list[str],
    hpc_title_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    enriched = enrich_prompt(text)
    slides = extract_slide_ids(enriched, slide_list)
    hpc_ids = extract_hpc_ids_regex(enriched)

    if not hpc_ids and hpc_title_map:
        hid = hpc_id_from_title(enriched, hpc_title_map)
        if hid is not None:
            hpc_ids = [str(hid)]

    tile_match = re.search(
        r"\b(tile[_\-]?\d+|[A-Za-z0-9_\-]+\.jpe?g|[A-Za-z0-9_\-]+\.png|[A-Za-z0-9_\-]+\.tif)\b",
        enriched,
        re.I,
    )
    tile = tile_match.group(0) if tile_match else None

    sample_match = re.search(r"\bsample\s*[:=]?\s*([^\s,;]+)", enriched, re.I)
    sample = sample_match.group(1).strip() if sample_match else None

    return {
        "enriched_text": enriched,
        "slide": slides,
        "hpc": hpc_ids,
        "tile": [tile] if tile else [],
        "sample": [sample] if sample else [],
    }


def validate_plan_entities(
    plan: dict[str, Any],
    slide_list: list[str],
    valid_hpc_ids: set[int] | None = None,
) -> dict[str, Any]:
    out = dict(plan)
    ents = dict(plan.get("entities") or {})
    slide_index = build_slide_index(slide_list)

    normalised_slides: list[str] = []
    for s in ents.get("slide") or []:
        hit = slide_index.get(str(s).strip().lower())
        if hit and hit not in normalised_slides:
            normalised_slides.append(hit)
    ents["slide"] = normalised_slides

    if valid_hpc_ids is not None:
        cleaned_hpc = []
        for h in ents.get("hpc") or []:
            m = re.search(r"(\d+)", str(h))
            if m and int(m.group(1)) in valid_hpc_ids:
                cleaned_hpc.append(str(int(m.group(1))))
        ents["hpc"] = list(dict.fromkeys(cleaned_hpc))

    out["entities"] = ents
    return out
