"""Normalise a raw per-ring record through a project adapter.

Absorbs deviations of form met in permit sheets and logger exports: ground classes repeated or merged ('S1, S-1',
'S2/S3'), operating modes in several spellings, cover as a range ('17.2-21.8'), numbers with units or decimal
commas, missing grout, and overlapping revisions of the same ring (the latest is kept). Returns the standard table
with derived columns and a report that holds counts only."""
import re
import numpy as np
import pandas as pd
from .schema import NUMERIC, unit_factor

_RANGE = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*(?:-|–|—|to)\s*(-?\d+(?:[.,]\d+)?)")
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def _plain(v):
    try:
        float(v); return True
    except ValueError:
        return False


def to_number(x):
    """'2.54 bar' -> 2.54; '2,54' -> 2.54; '17.2-21.8' -> mean of the range; '' / NaN -> NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    m = _RANGE.match(s)
    if m:
        a, b = (float(v.replace(",", ".")) for v in m.groups())
        return (a + b) / 2
    m = _NUM.search(s)
    return float(m.group(0).replace(",", ".")) if m else np.nan


def range_spread(x):
    if isinstance(x, str):
        m = _RANGE.match(x.strip())
        if m:
            a, b = (float(v.replace(",", ".")) for v in m.groups())
            return abs(b - a)
    return np.nan


def normalise(raw: pd.DataFrame, adapter, r_exc: float):
    """raw: project table; adapter: Adapter; r_exc: excavated radius (m). Returns (table, report)."""
    t = adapter.rename(raw.copy()); rep = {}
    if "cover_m" in t:
        spread = t["cover_m"].map(range_spread)
        rep["cover_ranges"] = int(spread.notna().sum())
        if rep["cover_ranges"]:
            t["cover_spread_m"] = np.where(spread.notna(), spread, t.get("cover_spread_m", np.nan))
            if "cover_source" in t:
                t.loc[spread.notna(), "cover_source"] = "range"
    bad = 0
    for c in NUMERIC:
        if c in t and not pd.api.types.is_numeric_dtype(t[c]):          # object (pandas < 3) or str (pandas >= 3)
            bad += int(t[c].map(lambda v: isinstance(v, str) and not _plain(v)).sum())
            t[c] = t[c].map(to_number)
    rep["text_numbers"] = bad
    conv = []
    for c, u in (adapter.units or {}).items():                          # units that differ from the schema
        f = unit_factor(c, u)
        if c in t and f != 1.0:
            t[c] = t[c] * f
            if c == "cover_m" and "cover_spread_m" in t:
                t["cover_spread_m"] = t["cover_spread_m"] * f
            conv.append(c)
    rep["converted_units"] = len(conv)
    n0 = len(t)
    keys = ["drive", "ring_no"]
    t = (t.sort_values(keys + ["source_rev"]) if "source_rev" in t else t).drop_duplicates(keys, keep="last")
    rep["duplicate_revisions"] = n0 - len(t)
    t["ring_no"] = t["ring_no"].astype(int)
    if "ground_class" in t:
        rep["ground_strings"] = int(t["ground_class"].notna().sum())
        t["ground_set"] = t.pop("ground_class").map(adapter.ground_set)
    else:
        before = t["ground_set"].fillna("").astype(str) if "ground_set" in t else pd.Series("", index=t.index)
        t["ground_set"] = before.map(adapter.ground_set)
        rep["ground_strings"] = int((before != t["ground_set"]).sum())
    if "operating_mode" in t:
        rep["mode_strings"] = int(t["operating_mode"].notna().sum())
        t["mode"] = t.pop("operating_mode").map(adapter.mode)
    else:
        before = t["mode"].fillna("").astype(str) if "mode" in t else pd.Series("", index=t.index)
        t["mode"] = before.map(lambda s: s if s in adapter.modes else adapter.mode(s))
        rep["mode_strings"] = int((before != t["mode"]).sum())
    t["ground_mixed"] = t["ground_set"].str.contains("/")
    if "has_context" not in t:
        t["has_context"] = t["cover_m"].notna()
    t["has_context"] = t["has_context"].astype(bool) & t["cover_m"].notna()
    t = t.sort_values(keys)
    if "water_bar" not in t:
        t["water_bar"] = np.nan
    miss = t["has_context"] & t["water_bar"].isna()
    t["water_filled"] = (t["water_filled"].astype(bool) if "water_filled" in t else False) | miss
    t["water_bar"] = t.groupby("drive")["water_bar"].transform(lambda s: s.interpolate(limit_direction="both"))
    rep["water_filled"] = int(miss.sum())
    t["z_axis_m"] = t["cover_m"] + r_exc
    t["gw_depth_m"] = (t["cover_m"] - t["water_bar"] * 100 / 9.81).clip(lower=0.5)
    t["face_axis_kpa"] = t["face_axis_bar"] * 100
    t["face_crown_kpa"] = t["face_crown_bar"] * 100
    t["face_grad_kpa_m"] = (t["face_axis_kpa"] - t["face_crown_kpa"]) / r_exc
    for c in ("total_grout_m3", "theoretical_grout_m3", "cover_spread_m"):
        if c not in t:
            t[c] = np.nan
    t["grout_ratio"] = t["total_grout_m3"] / t["theoretical_grout_m3"]
    rep["grout_missing"] = int(t["grout_ratio"].isna().sum())
    t["cover_ok"] = t["has_context"] & (t["cover_spread_m"].fillna(0) <= adapter.max_cover_spread_m)
    t["gap_before"] = t.groupby("drive")["ring_no"].diff().fillna(1).ne(1)
    rep["rings"] = int(len(t)); rep["rings_with_context"] = int(t["has_context"].sum())
    return t.reset_index(drop=True), rep
