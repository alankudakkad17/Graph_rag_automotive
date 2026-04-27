"""
automotive_schema.py
────────────────────
Defines the Neo4j graph schema for the Automotive / BMW domain.

Node labels  →  Vehicle, Engine, Feature, System, Component,
                Specification, Recall, DiagnosticCode, Document, Chunk

Edge types   →  HAS_ENGINE, HAS_FEATURE, BELONGS_TO_SYSTEM,
                HAS_COMPONENT, HAS_SPECIFICATION, SUBJECT_TO_RECALL,
                HAS_DTC, SOURCED_FROM, NEXT_CHUNK
"""

# ── Cypher: create constraints & indexes ──────────────────────────────────────
SCHEMA_SETUP_QUERIES = [
    # ── Uniqueness constraints ─────────────────────────────────────────────
    "CREATE CONSTRAINT vehicle_id IF NOT EXISTS FOR (v:Vehicle) REQUIRE v.id IS UNIQUE",
    "CREATE CONSTRAINT engine_id  IF NOT EXISTS FOR (e:Engine)  REQUIRE e.id IS UNIQUE",
    "CREATE CONSTRAINT feature_id IF NOT EXISTS FOR (f:Feature) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT system_id  IF NOT EXISTS FOR (s:System)  REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT comp_id    IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT spec_id    IF NOT EXISTS FOR (s:Specification) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT recall_id  IF NOT EXISTS FOR (r:Recall)  REQUIRE r.id IS UNIQUE",
    "CREATE CONSTRAINT dtc_id     IF NOT EXISTS FOR (d:DiagnosticCode) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT doc_id     IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id   IF NOT EXISTS FOR (c:Chunk)   REQUIRE c.id IS UNIQUE",

    # ── Full-text indexes for keyword search ──────────────────────────────
    """CREATE FULLTEXT INDEX chunk_text_idx IF NOT EXISTS
       FOR (c:Chunk) ON EACH [c.text]""",
    """CREATE FULLTEXT INDEX feature_idx IF NOT EXISTS
       FOR (f:Feature) ON EACH [f.name, f.description]""",
    """CREATE FULLTEXT INDEX spec_idx IF NOT EXISTS
       FOR (s:Specification) ON EACH [s.attribute, s.value]""",

    # ── Vector index (Neo4j 5.11+ native vector index) ────────────────────
    """CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS
       FOR (c:Chunk) ON c.embedding
       OPTIONS {indexConfig: {
         `vector.dimensions`: 384,
         `vector.similarity_function`: 'cosine'
       }}""",
]

# ── Node property schemas (for validation / documentation) ───────────────────
NODE_SCHEMAS = {
    "Vehicle": {
        "id":           "str  – BMW_{model}_{year}",
        "make":         "str  – e.g. BMW",
        "model":        "str  – e.g. 7 Series",
        "variant":      "str  – e.g. 740i, 760i xDrive",
        "year":         "int  – e.g. 2023",
        "body_style":   "str  – Sedan / SAV / etc.",
        "drivetrain":   "str  – RWD / AWD",
        "msrp":         "str  – starting price",
    },
    "Engine": {
        "id":             "str",
        "type":           "str  – TwinPower Turbo Inline-6 / V8 etc.",
        "displacement":   "str  – 3.0L / 4.4L",
        "cylinders":      "int",
        "horsepower":     "int  – hp",
        "torque":         "int  – lb-ft",
        "fuel_type":      "str  – Gasoline / PHEV / EV",
        "transmission":   "str  – 8-speed Steptronic",
    },
    "Feature": {
        "id":          "str",
        "name":        "str  – e.g. Driving Assistance Professional",
        "category":    "str  – Safety / Infotainment / Comfort / Performance",
        "description": "str",
        "standard":    "bool – True if standard, False if optional",
    },
    "System": {
        "id":          "str",
        "name":        "str  – e.g. iDrive 8.5, M Sport Brakes",
        "type":        "str  – Infotainment / Braking / Suspension / ADAS",
        "description": "str",
    },
    "Component": {
        "id":          "str",
        "name":        "str",
        "part_number": "str",
        "category":    "str",
    },
    "Specification": {
        "id":        "str",
        "attribute": "str  – e.g. Wheelbase, Curb Weight",
        "value":     "str",
        "unit":      "str  – mm / kg / hp / Nm",
        "category":  "str  – Dimensions / Performance / Capacity",
    },
    "Recall": {
        "id":          "str  – NHTSA campaign number",
        "title":       "str",
        "description": "str",
        "date":        "str",
        "remedy":      "str",
    },
    "DiagnosticCode": {
        "id":          "str  – e.g. P0300",
        "code":        "str",
        "description": "str",
        "severity":    "str  – Critical / High / Medium / Low",
        "system":      "str",
    },
    "Document": {
        "id":       "str",
        "title":    "str",
        "source":   "str  – file path",
        "type":     "str  – Spec Sheet / Owner Manual / Service Manual",
        "pages":    "int",
    },
    "Chunk": {
        "id":         "str",
        "text":       "str",
        "page":       "int",
        "chunk_idx":  "int",
        "embedding":  "list[float] – 384-d vector",
        "doc_id":     "str",
    },
}

# ── Relationship types ────────────────────────────────────────────────────────
RELATIONSHIPS = {
    "HAS_ENGINE":           ("Vehicle",     "Engine"),
    "HAS_FEATURE":          ("Vehicle",     "Feature"),
    "BELONGS_TO_SYSTEM":    ("Feature",     "System"),
    "HAS_COMPONENT":        ("System",      "Component"),
    "HAS_SPECIFICATION":    ("Vehicle",     "Specification"),
    "ENGINE_SPEC":          ("Engine",      "Specification"),
    "SUBJECT_TO_RECALL":    ("Vehicle",     "Recall"),
    "HAS_DTC":              ("Component",   "DiagnosticCode"),
    "SOURCED_FROM":         ("Chunk",       "Document"),
    "MENTIONS_VEHICLE":     ("Chunk",       "Vehicle"),
    "MENTIONS_FEATURE":     ("Chunk",       "Feature"),
    "MENTIONS_SPEC":        ("Chunk",       "Specification"),
    "NEXT_CHUNK":           ("Chunk",       "Chunk"),
}
