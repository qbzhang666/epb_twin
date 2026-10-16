"""Seeded deviations of form (layer 3): the same record written the way permit sheets and logger exports write it.
Values are unchanged in substance. Applied to a clean standard table, returned in the adapter's raw column names."""
import re
import numpy as np
import pandas as pd

KINDS = ("ground", "mode", "cover_range", "units", "revisions", "grout")
MODE_SPELLINGS = {"full_EPB": ["EPB", "Full EPB", "EPB (full)", "closed EPB mode"],
                  "semi_EPB": ["Semi-EPB", "semi EPB", "Semi closed EPB"],
                  "EPB_air_bubble": ["EPB air bubble", "Air-bubble EPB"]}
DERIVED = ("z_axis_m", "gw_depth_m", "face_axis_kpa", "face_crown_kpa", "face_grad_kpa_m", "grout_ratio", "cover_ok",
           "gap_before", "ground_mixed", "water_filled")


def ground_spellings(code):
    """'S1' -> ['S1', 'S-1', 's1', 'S 1', 'S1, S1']; single letters -> ['R', 'r', 'R, R']."""
    out = [code, code.lower(), f"{code}, {code}"]
    m = re.match(r"^([A-Za-z]+)(\w)$", code)
    if m and len(code) > 1:
        out += [f"{m.group(1)}-{m.group(2)}", f"{m.group(1)} {m.group(2)}"]
    return out


def make_variant(clean: pd.DataFrame, adapter, seed: int, kinds=KINDS, frac=0.3):
    """clean: standard table (as from normalise or the synthetic generator). Returns a raw table for the adapter."""
    rng = np.random.default_rng(seed); t = clean.copy().reset_index(drop=True)
    pick = lambda: rng.random(len(t)) < frac
    if "ground" in kinds and "ground_set" in t:
        sp = {c: ground_spellings(c) for c in adapter.ground_order}
        t["ground_class"] = [s if not isinstance(s, str) or not s else
                             rng.choice([" + ", "/", ", ", " and "]).join(rng.choice(sp.get(p, [p])) for p in s.split("/"))
                             for s in t.pop("ground_set")]
    if "mode" in kinds and "mode" in t:
        t["operating_mode"] = [rng.choice(MODE_SPELLINGS[m]) if m in MODE_SPELLINGS else m for m in t.pop("mode")]
    t = t.astype({c: object for c in ("cover_m", "face_crown_bar", "face_axis_bar") if c in t})
    if "cover_range" in kinds:
        m = pick() & t["has_context"].astype(bool).to_numpy() & t["cover_m"].notna().to_numpy()
        for i in np.flatnonzero(m):
            c = float(t.at[i, "cover_m"]); sp_ = t.at[i, "cover_spread_m"] if "cover_spread_m" in t else np.nan
            h = (float(sp_) if pd.notna(sp_) else 0.4) / 2
            t.at[i, "cover_m"] = f"{c - h!r}-{c + h!r}"
    if "units" in kinds:
        for col in ("face_crown_bar", "face_axis_bar"):
            m = pick() & t[col].notna().to_numpy()
            for i in np.flatnonzero(m):
                v = repr(float(t.at[i, col])); t.at[i, col] = rng.choice([v + " bar", v.replace(".", ","), " " + v])
    if "grout" in kinds and "total_grout_m3" in t:
        m = pick(); t.loc[m, "total_grout_m3"] = np.nan
    if "revisions" in kinds:
        m = pick(); dup = t[m].copy()
        dup["source_rev"] = pd.to_numeric(dup.get("source_rev", 0), errors="coerce").fillna(1) - 1
        for col in ("thrust_kn", "total_grout_m3"):
            if col in dup:
                dup[col] = pd.to_numeric(dup[col], errors="coerce") * 1.15
        dup["face_crown_bar"] = [repr(float(v) + 0.3) if pd.notna(pd.to_numeric(v, errors="coerce")) and not isinstance(v, str) else v
                                 for v in dup["face_crown_bar"]]
        t = pd.concat([dup, t]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    t = t.drop(columns=[c for c in DERIVED if c in t])
    return t.rename(columns={s: r for s, r in adapter.columns.items() if s in t.columns})
