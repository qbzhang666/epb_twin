"""Standard per-ring record of an EPB drive. One row per built ring; rings are numbered within a drive.

Every project adapter maps its raw columns onto these names. Units are fixed; an adapter converts if needed.
kind: 'id' identifiers, 'context' ground and geometry at the ring, 'op' operating record, 'grout' volumes,
'label' text classes normalised through the adapter vocabulary."""
from dataclasses import dataclass

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Field:
    name: str
    unit: str
    kind: str
    required: bool
    description: str


FIELDS = [
    Field("drive", "-", "id", True, "drive identifier (one bore)"),
    Field("ring_no", "-", "id", True, "ring number within the drive"),
    Field("source_rev", "-", "id", False, "revision of the source sheet; the latest revision of a ring is kept"),
    Field("cover_m", "m", "context", True, "cover above the crown; a range 'a-b' is read as its mean, the spread kept"),
    Field("cover_spread_m", "m", "context", False, "spread of the cover estimate at the ring"),
    Field("cover_source", "-", "context", False, "how the cover was obtained (per ring, range, interpolated)"),
    Field("rock_cover_m", "m", "context", False, "rock above the crown"),
    Field("soil_thickness_m", "m", "context", False, "soil above the rock head"),
    Field("water_bar", "bar", "context", False, "groundwater pressure at the crown"),
    Field("ground_class", "-", "label", False, "ground class text at the face (normalised to ground_set)"),
    Field("operating_mode", "-", "label", False, "operating mode text (normalised to mode)"),
    Field("face_crown_bar", "bar", "op", True, "chamber pressure used as the crown value"),
    Field("face_axis_bar", "bar", "op", True, "chamber pressure at the axis"),
    Field("thrust_kn", "kN", "op", True, "total thrust of the cylinders"),
    Field("contact_force_kn", "kN", "op", False, "cutterhead contact force, if logged"),
    Field("cutterhead_torque_mnm", "MN m", "op", True, "cutterhead torque"),
    Field("advance_speed_mm_min", "mm/min", "op", True, "advance rate"),
    Field("cutterhead_speed_rpm", "rpm", "op", True, "cutterhead speed"),
    Field("penetration_mm_rev", "mm/rev", "op", False, "penetration per revolution"),
    Field("total_grout_m3", "m3", "grout", False, "tail grout injected for the ring"),
    Field("theoretical_grout_m3", "m3", "grout", False, "theoretical annulus volume of the ring"),
    Field("over_excavation_post_grout_pct", "%", "grout", False, "excavated volume over theoretical, after grout"),
]
DERIVED = {
    "ground_set": "ordered set of ground classes at the face, e.g. 'S1/S2'",
    "ground_mixed": "more than one ground class at the face",
    "mode": "operating mode in the adapter's canonical names",
    "has_context": "ring has a cover value",
    "water_filled": "water pressure filled from the nearest sheet of the drive",
    "z_axis_m": "depth of the axis, cover + excavated radius",
    "gw_depth_m": "depth of the water table implied by water_bar (>= 0.5 m)",
    "face_axis_kpa": "face_axis_bar in kPa",
    "face_crown_kpa": "face_crown_bar in kPa",
    "face_grad_kpa_m": "face pressure gradient between crown and axis values",
    "grout_ratio": "total over theoretical grout",
    "cover_ok": "context present and cover spread within the adapter limit",
    "gap_before": "rings missing in the record before this ring",
}
NAMES = [f.name for f in FIELDS]
REQUIRED = [f.name for f in FIELDS if f.required]
NUMERIC = [f.name for f in FIELDS if f.unit != "-"] + ["ring_no", "source_rev"]


def validate(table):
    """Counts only: required fields missing, unknown columns, rows without a ring number."""
    cols = set(table.columns)
    return {"missing_required": [c for c in REQUIRED if c not in cols],
            "unknown_columns": sorted(cols - set(NAMES) - set(DERIVED)),
            "rows": int(len(table)),
            "rows_without_ring": int(table["ring_no"].isna().sum()) if "ring_no" in cols else int(len(table))}


# unit conversion: factor from a given unit to the standard unit of a field
_G = 9.80665
FACTORS = {
    "bar": {"bar": 1, "kpa": 0.01, "mpa": 10, "pa": 1e-5, "psi": 0.0689476},
    "kN": {"kn": 1, "mn": 1000, "n": 1e-3, "tf": _G},
    "MN m": {"mn m": 1, "mnm": 1, "kn m": 1e-3, "knm": 1e-3, "n m": 1e-6, "nm": 1e-6, "tf m": _G * 1e-3, "tfm": _G * 1e-3},
    "m": {"m": 1, "mm": 1e-3, "cm": 1e-2, "ft": 0.3048},
    "mm/min": {"mm/min": 1, "m/h": 1000 / 60, "mm/s": 60, "m/min": 1000},
    "m3": {"m3": 1, "l": 1e-3},
    "rpm": {"rpm": 1, "rad/s": 60 / (2 * 3.141592653589793)},
    "mm/rev": {"mm/rev": 1},
    "%": {"%": 1, "fraction": 100},
}


def unit_factor(field_name, unit):
    """Factor that converts a value of field_name given in unit to the field's standard unit."""
    f = next((x for x in FIELDS if x.name == field_name), None)
    if f is None:
        raise ValueError(f"unknown field {field_name}")
    table = FACTORS.get(f.unit)
    key = str(unit).strip().lower().replace("·", " ").replace("³", "3")
    if table is None or key not in table:
        raise ValueError(f"{field_name}: cannot convert {unit!r} to {f.unit} (known: {sorted(table or [])})")
    return float(table[key])
