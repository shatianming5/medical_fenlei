from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Triplet:
    """
    Entity-attribute triplet extracted from a radiology report.

    location:
      - "L" / "R" / "B" / "LR" / "U"
    """

    location: str
    entity: str
    attribute: str
    negated: bool
    clause: str
    entity_span: tuple[int, int]
    attribute_span: tuple[int, int]

    def as_dict(self) -> dict:
        return {
            "location": self.location,
            "entity": self.entity,
            "attribute": self.attribute,
            "negated": bool(self.negated),
            "clause": self.clause,
            "entity_span": [int(self.entity_span[0]), int(self.entity_span[1])],
            "attribute_span": [int(self.attribute_span[0]), int(self.attribute_span[1])],
        }


_CLAUSE_SPLIT_RE = re.compile(r"[。\n\r;；]+")

# Very lightweight dictionaries (can be expanded over time).
_ENTITY_PATTERNS: dict[str, tuple[str, ...]] = {
    "外耳道": ("外耳道", "外耳"),
    "鼓膜": ("鼓膜",),
    "鼓室": ("鼓室", "中耳", "鼓室腔"),
    "上鼓室": ("上鼓室",),
    "鼓室窦": ("鼓室窦", "窦口", "鼓室窦口"),
    "听骨链": ("听骨链", "听小骨", "听小骨链", "锤骨", "砧骨", "镫骨", "听骨"),
    "乳突": ("乳突", "乳突气房", "乳突气化", "乳突蜂房", "气房", "气化"),
    "半规管": ("半规管",),
    "耳蜗": ("耳蜗",),
    "面神经管": ("面神经管",),
    "乙状窦": ("乙状窦",),
    "颈静脉球": ("颈静脉球",),
    "岩尖": ("岩尖", "岩部", "岩骨尖"),
}

_ATTRIBUTE_PATTERNS: dict[str, tuple[str, ...]] = {
    # bone change
    "骨质吸收": ("骨质吸收", "吸收"),
    "骨质破坏": ("骨质破坏", "破坏", "侵蚀", "侵蚀性破坏"),
    "骨质毛糙": ("骨皮质毛糙", "骨皮质粗糙", "骨质毛糙", "毛糙"),
    "膨胀性": ("膨胀性", "膨胀", "扩张"),
    "边缘光滑": ("边缘光滑", "边界光滑", "光滑"),
    "边界锐利": ("边界清楚锐利", "边界清楚", "锐利", "清楚"),
    # soft tissue / fluid / aeration
    "软组织影": ("软组织影", "软组织密度影", "肿块影", "占位"),
    "液平": ("液平", "气液平面"),
    "气泡": ("气泡", "气泡征"),
    "充填": ("充填", "填充"),
    "气化良好": ("气化良好", "气化型"),
    "气化减少": ("气化减少", "气化不良", "气化差"),
    "硬化": ("硬化", "硬化型"),
    "密度增高": ("密度增高", "密度影", "高密度"),
    # general status
    "完整": ("完整", "连续"),
}

_NEGATION_PREFIXES: tuple[str, ...] = ("未见", "无", "未", "不见", "未发现", "未见明显", "未见明确")

_LOC_BILAT: tuple[str, ...] = ("双侧", "双耳", "两侧", "双耳侧", "双侧耳")
_LOC_LEFT: tuple[str, ...] = ("左侧", "左耳", "左")
_LOC_RIGHT: tuple[str, ...] = ("右侧", "右耳", "右")


def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\u3000]+", " ", s)
    return s.strip()


def split_report_clauses(text: str) -> list[str]:
    """
    Split a report into coarse clauses.
    """
    s = _normalize_text(text)
    if not s:
        return []
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(s) if p.strip()]
    return parts


def _detect_location(clause: str) -> str:
    c = str(clause)
    if any(k in c for k in _LOC_BILAT):
        return "B"
    has_l = any(k in c for k in _LOC_LEFT)
    has_r = any(k in c for k in _LOC_RIGHT)
    if has_l and has_r:
        return "LR"
    if has_l:
        return "L"
    if has_r:
        return "R"
    return "U"


def _find_matches(clause: str, patterns: dict[str, tuple[str, ...]], *, kind: str) -> list[dict]:
    out: list[dict] = []
    for canonical, pats in patterns.items():
        for p in pats:
            if not p:
                continue
            start = 0
            while True:
                idx = clause.find(p, start)
                if idx < 0:
                    break
                out.append(
                    {
                        "kind": str(kind),
                        "canonical": str(canonical),
                        "pattern": str(p),
                        "start": int(idx),
                        "end": int(idx + len(p)),
                    }
                )
                start = idx + max(1, len(p))
    out.sort(key=lambda d: (int(d["start"]), -int(d["end"])))
    return out


def _is_negated(clause: str, attr_start: int) -> bool:
    # Look slightly before the attribute span for common negation prefixes.
    left = max(0, int(attr_start) - 6)
    window = clause[left:int(attr_start)]
    return any(n in window for n in _NEGATION_PREFIXES)


def _choose_entity(entities: list[dict], attr: dict, *, window: int = 16) -> dict | None:
    if not entities:
        return None
    a0 = int(attr["start"])
    a1 = int(attr["end"])

    before = [e for e in entities if int(e["end"]) <= a0 and (a0 - int(e["end"])) <= int(window)]
    if before:
        return max(before, key=lambda e: int(e["end"]))  # nearest before

    after = [e for e in entities if int(e["start"]) >= a1 and (int(e["start"]) - a1) <= int(window)]
    if after:
        return min(after, key=lambda e: int(e["start"]))  # nearest after

    # fallback: global nearest
    def dist(e: dict) -> int:
        return min(abs(a0 - int(e["start"])), abs(a0 - int(e["end"])))

    nearest = min(entities, key=dist)
    if dist(nearest) <= int(window) * 2:
        return nearest
    return None


def extract_entity_attribute_triplets(text: str) -> list[Triplet]:
    """
    Extract (location, entity, attribute) triplets from a report_text.

    This is a lightweight baseline for check.md 4.2.2. It is intentionally
    rule-based and dependency-free (no external NLP model required).
    """
    out: list[Triplet] = []
    for clause in split_report_clauses(text):
        loc = _detect_location(clause)
        entities = _find_matches(clause, _ENTITY_PATTERNS, kind="entity")
        attrs = _find_matches(clause, _ATTRIBUTE_PATTERNS, kind="attribute")
        if not entities or not attrs:
            continue

        for a in attrs:
            ent = _choose_entity(entities, a, window=16)
            if ent is None:
                continue
            neg = _is_negated(clause, int(a["start"]))
            out.append(
                Triplet(
                    location=str(loc),
                    entity=str(ent["canonical"]),
                    attribute=str(a["canonical"]),
                    negated=bool(neg),
                    clause=str(clause),
                    entity_span=(int(ent["start"]), int(ent["end"])),
                    attribute_span=(int(a["start"]), int(a["end"])),
                )
            )
    return out


def triplets_to_strings(triplets: Iterable[Triplet]) -> list[str]:
    """
    Compact string form for logging / quick inspection.
    """
    out: list[str] = []
    for t in triplets:
        neg = "NEG:" if bool(t.negated) else ""
        out.append(f"{t.location}:{t.entity}:{neg}{t.attribute}")
    return out

