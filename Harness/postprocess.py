
import re

VALID_LABELS = {"PERSON", "ORG", "LOCATION", "EVENT"}

TITLE_PATTERN = re.compile(
    r"^(?:Mr|Mrs|Ms|Miss|Dr|Minister)\.?\s+",
    flags=re.IGNORECASE,
)

POSSESSIVE_PATTERN = re.compile(
    r"(?:'s|’s)$",
    flags=re.IGNORECASE,
)


def postprocess(raw, source_text):
    """
    Normalize NER predictions.

    Rules:
    1. Accept only PERSON, ORG, LOCATION, and EVENT.
    2. Remove common titles and possessive endings.
    3. Keep only entity names appearing in the source text.
    4. Deduplicate entities case-insensitively by text and label.
    5. Prefer longer entity names when spans overlap.
    """
    if isinstance(raw, dict):
        entities = raw.get("entities", [])
    elif isinstance(raw, list):
        entities = raw
    else:
        entities = []

    if not isinstance(entities, list):
        return []

    cleaned = []
    source_lower = source_text.casefold()

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        text = entity.get("text")
        label = entity.get("label")

        if not isinstance(text, str) or not isinstance(label, str):
            continue

        text = text.strip()
        label = label.strip().upper()

        if not text or label not in VALID_LABELS:
            continue

        text = TITLE_PATTERN.sub("", text).strip()
        text = POSSESSIVE_PATTERN.sub("", text).strip()

        if not text:
            continue

        if text.casefold() not in source_lower:
            continue

        cleaned.append({
            "text": text,
            "label": label,
        })

    # Remove exact duplicate text-label pairs.
    unique = []
    seen = set()

    for entity in cleaned:
        key = (
            entity["text"].casefold(),
            entity["label"],
        )

        if key not in seen:
            seen.add(key)
            unique.append(entity)

    # Prefer longer names when their source-text spans overlap.
    candidates = []

    for entity in unique:
        start = source_lower.find(entity["text"].casefold())

        if start == -1:
            continue

        end = start + len(entity["text"])

        candidates.append({
            **entity,
            "_start": start,
            "_end": end,
        })

    candidates.sort(
        key=lambda entity: (
            -(entity["_end"] - entity["_start"]),
            entity["_start"],
            entity["label"],
        )
    )

    kept = []

    for entity in candidates:
        overlaps = any(
            entity["_start"] < existing["_end"]
            and existing["_start"] < entity["_end"]
            for existing in kept
        )

        if not overlaps:
            kept.append(entity)

    # Return only the required output fields.
    kept.sort(key=lambda entity: entity["_start"])

    return [
        {
            "text": entity["text"],
            "label": entity["label"],
        }
        for entity in kept
    ]
