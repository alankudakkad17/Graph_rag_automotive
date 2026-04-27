"""
entity_extractor.py
────────────────────
Extracts automotive domain entities from text chunks using:
  1. Rule-based patterns (BMW-specific – fast, high precision)
  2. spaCy NER (general NLP)

Entities extracted: Vehicle, Engine, Feature, Specification, Recall, DTC
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import spacy

from backend.utils import logger

# ── Load spaCy model (small, free) ────────────────────────────────────────────
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy 'en_core_web_sm' not found – run: python -m spacy download en_core_web_sm")
    _nlp = None


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class ExtractedEntities:
    vehicles: list[dict] = field(default_factory=list)
    engines: list[dict] = field(default_factory=list)
    features: list[dict] = field(default_factory=list)
    specifications: list[dict] = field(default_factory=list)
    recalls: list[dict] = field(default_factory=list)
    dtc_codes: list[dict] = field(default_factory=list)


# ── Regex patterns ────────────────────────────────────────────────────────────
_VARIANT_RE = re.compile(
    r"\b(7[0-9][0-9][iedxX]{0,2}(?:\s+(?:xDrive|eDrive|M\s*Sport)?)?)\b"
)
_ENGINE_HP_RE = re.compile(r"(\d{2,4})\s*(?:hp|horsepower|bhp|PS)", re.IGNORECASE)
_ENGINE_TORQUE_RE = re.compile(r"(\d{2,4})\s*(?:lb-ft|lbft|Nm|newton)", re.IGNORECASE)
_ENGINE_DISP_RE = re.compile(r"(\d\.\d)\s*(?:L|liter|litre)", re.IGNORECASE)
_SPEC_DIM_RE = re.compile(
    r"(wheelbase|length|width|height|curb weight|gross weight|"
    r"cargo|trunk|fuel tank|turning radius)[:\s]+([0-9,.]+)\s*"
    r"(mm|cm|m|in|inches|kg|lbs?|liters?|L|gallons?|ft)?",
    re.IGNORECASE,
)
_DTC_RE = re.compile(r"\b([PBCU][0-9]{4}(?:-[0-9]{2})?)\b")
_RECALL_RE = re.compile(r"NHTSA\s+(?:Campaign|Recall)\s+#?([0-9A-Z\-]+)", re.IGNORECASE)
_FEATURE_KEYWORDS = [
    "Driving Assistant", "Active Cruise Control", "Lane Change Warning",
    "Parking Assistant", "Reversing Camera", "Head-Up Display",
    "iDrive", "BMW Live Cockpit", "Surround View", "Night Vision",
    "Gesture Control", "Wireless Charging", "Harman Kardon", "Bowers & Wilkins",
    "M Sport", "xDrive", "Adaptive Suspension", "Air Suspension",
    "Executive Drive Pro", "Integral Active Steering", "Automatic Emergency Call",
    "Remote Software Upgrade", "BMW Digital Key",
]
_FEATURE_RE = re.compile(
    "|".join(re.escape(k) for k in _FEATURE_KEYWORDS),
    re.IGNORECASE,
)


class EntityExtractor:
    """Extracts structured entities from raw text for graph ingestion."""

    def extract(self, text: str) -> ExtractedEntities:
        entities = ExtractedEntities()
        self._extract_vehicles(text, entities)
        self._extract_engines(text, entities)
        self._extract_features(text, entities)
        self._extract_specifications(text, entities)
        self._extract_dtc(text, entities)
        self._extract_recalls(text, entities)
        if _nlp:
            self._spacy_enhance(text, entities)
        return entities

    # ── Rule-based extractors ─────────────────────────────────────────────────
    def _extract_vehicles(self, text: str, e: ExtractedEntities) -> None:
        for m in _VARIANT_RE.finditer(text):
            variant = m.group(1).strip()
            e.vehicles.append({
                "id": f"BMW_{variant.replace(' ', '_')}",
                "make": "BMW",
                "model": "7 Series",
                "variant": variant,
                "year": 2023,
                "body_style": "Sedan",
            })
        # deduplicate
        seen = set()
        e.vehicles = [v for v in e.vehicles
                      if not (v["id"] in seen or seen.add(v["id"]))]

    def _extract_engines(self, text: str, e: ExtractedEntities) -> None:
        hp_matches = _ENGINE_HP_RE.findall(text)
        torque_matches = _ENGINE_TORQUE_RE.findall(text)
        disp_matches = _ENGINE_DISP_RE.findall(text)
        if hp_matches or disp_matches:
            engine = {
                "id": f"engine_{disp_matches[0] if disp_matches else 'unknown'}",
                "horsepower": int(hp_matches[0]) if hp_matches else None,
                "torque": int(torque_matches[0]) if torque_matches else None,
                "displacement": f"{disp_matches[0]}L" if disp_matches else None,
            }
            # Infer engine type from context
            if "turbo" in text.lower():
                engine["type"] = "TwinPower Turbo"
            if "hybrid" in text.lower() or "phev" in text.lower():
                engine["fuel_type"] = "Plug-in Hybrid"
                engine["type"] = "PHEV"
            elif "electric" in text.lower() or "ev" in text.lower():
                engine["fuel_type"] = "Electric"
            else:
                engine["fuel_type"] = "Gasoline"
            e.engines.append(engine)

    def _extract_features(self, text: str, e: ExtractedEntities) -> None:
        for m in _FEATURE_RE.finditer(text):
            name = m.group(0).strip()
            category = self._classify_feature(name)
            e.features.append({
                "id": f"feat_{name.lower().replace(' ', '_')[:40]}",
                "name": name,
                "category": category,
                "description": self._extract_sentence(text, m.start()),
                "standard": "standard" in text[max(0, m.start()-50):m.end()+50].lower(),
            })
        # deduplicate by name
        seen = set()
        e.features = [f for f in e.features
                      if not (f["name"] in seen or seen.add(f["name"]))]

    def _extract_specifications(self, text: str, e: ExtractedEntities) -> None:
        for m in _SPEC_DIM_RE.finditer(text):
            attr = m.group(1).strip()
            val = m.group(2).strip()
            unit = (m.group(3) or "").strip()
            e.specifications.append({
                "id": f"spec_{attr.lower().replace(' ', '_')}_{val}",
                "attribute": attr,
                "value": val,
                "unit": unit,
                "category": self._classify_spec(attr),
            })

    def _extract_dtc(self, text: str, e: ExtractedEntities) -> None:
        for m in _DTC_RE.finditer(text):
            code = m.group(1)
            e.dtc_codes.append({
                "id": f"dtc_{code}",
                "code": code,
                "description": self._extract_sentence(text, m.start()),
                "severity": "Unknown",
                "system": self._infer_dtc_system(code),
            })

    def _extract_recalls(self, text: str, e: ExtractedEntities) -> None:
        for m in _RECALL_RE.finditer(text):
            recall_id = m.group(1)
            e.recalls.append({
                "id": f"recall_{recall_id}",
                "title": f"NHTSA Recall {recall_id}",
                "description": self._extract_sentence(text, m.start()),
                "date": "2023",
                "remedy": "",
            })

    # ── spaCy enhancement ─────────────────────────────────────────────────────
    def _spacy_enhance(self, text: str, e: ExtractedEntities) -> None:
        doc = _nlp(text[:5000])  # Limit for performance
        for ent in doc.ents:
            if ent.label_ in ("QUANTITY", "CARDINAL") and any(
                u in ent.text for u in ["hp", "Nm", "kg", "mm", "L"]
            ):
                pass  # already captured by regex
            elif ent.label_ == "PRODUCT" and "bmw" not in ent.text.lower():
                e.features.append({
                    "id": f"feat_ner_{ent.text[:30].replace(' ', '_').lower()}",
                    "name": ent.text,
                    "category": "Technology",
                    "description": "",
                    "standard": False,
                })

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _classify_feature(name: str) -> str:
        name_l = name.lower()
        if any(w in name_l for w in ["assist", "safety", "emergency", "warning",
                                      "brake", "collision", "lane"]):
            return "Safety & ADAS"
        if any(w in name_l for w in ["idrive", "cockpit", "display", "gesture",
                                      "voice", "audio", "harman", "bowers"]):
            return "Infotainment"
        if any(w in name_l for w in ["air", "suspension", "drive", "steering",
                                      "adaptive", "sport", "executive"]):
            return "Driving Dynamics"
        if any(w in name_l for w in ["seat", "massage", "ambient", "panorama",
                                      "night vision", "comfort", "charging"]):
            return "Comfort & Convenience"
        return "General"

    @staticmethod
    def _classify_spec(attr: str) -> str:
        attr_l = attr.lower()
        if any(w in attr_l for w in ["length", "width", "height", "wheelbase",
                                      "turning", "ground"]):
            return "Dimensions"
        if any(w in attr_l for w in ["weight", "payload", "gross"]):
            return "Weight"
        if any(w in attr_l for w in ["cargo", "trunk", "tank", "fuel"]):
            return "Capacity"
        if any(w in attr_l for w in ["hp", "torque", "speed", "0-60"]):
            return "Performance"
        return "General"

    @staticmethod
    def _infer_dtc_system(code: str) -> str:
        prefix = code[0].upper()
        return {"P": "Powertrain", "B": "Body", "C": "Chassis", "U": "Network"}.get(
            prefix, "Unknown"
        )

    @staticmethod
    def _extract_sentence(text: str, pos: int, radius: int = 150) -> str:
        start = max(0, pos - radius)
        end = min(len(text), pos + radius)
        return text[start:end].replace("\n", " ").strip()
