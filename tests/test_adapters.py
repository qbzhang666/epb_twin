"""The same drive written in three record styles (example, a logger export in SI sub-units, Chinese headers) gives
the same standard table after normalisation."""
import sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epb_twin import load_adapter, normalise, load_machine
from epb_twin.schema import unit_factor
from epb_twin.synthetic import drive_table

A = Path(__file__).resolve().parents[1] / "epb_twin" / "adapters"
S = Path(__file__).resolve().parents[1] / "epb_twin" / "specs"
WORDS = {"logger_si": {"F": "fill", "S": "clay", "W": "weathered", "R": "rock"},
         "chinese_headers": {"F": "填土", "S": "黏土", "W": "强风化", "R": "中风化"}}
MODES = {"logger_si": {"full_EPB": "EPB closed", "semi_EPB": "semi open", "EPB_air_bubble": "air bubble"},
         "chinese_headers": {"full_EPB": "土压平衡", "semi_EPB": "半土压", "EPB_air_bubble": "气压平衡"}}


def as_style(clean, ad, style):
    t = clean.copy()
    for c, u in ad.units.items():                                   # back to the record's units
        t[c] = t[c] / unit_factor(c, u)
    t["ground_class"] = [" / ".join(WORDS[style][x] for x in s.split("/")) if s else "" for s in t.pop("ground_set")]
    t["operating_mode"] = [MODES[style][m] for m in t.pop("mode")]
    return t.rename(columns={s: r for s, r in ad.columns.items() if s in t})


def test_styles_agree():
    m = load_machine(S / "machine_example_7m.yaml"); clean = drive_table(m, seed=3)
    ref, _ = normalise(clean.rename(columns={"ground_set": "ground_class", "mode": "operating_mode"}),
                       load_adapter(A / "example.yaml"), m.radius_m)
    for style in ("logger_si", "chinese_headers"):
        ad = load_adapter(A / f"{style}.yaml")
        raw = as_style(clean, ad, style)
        p = Path(tempfile.mkdtemp()) / "r.csv"; raw.to_csv(p, index=False, encoding="utf-8")
        t, rep = normalise(pd.read_csv(p, encoding="utf-8"), ad, m.radius_m)
        assert list(t.ground_set) == list(ref.ground_set), style
        assert list(t["mode"]) == list(ref["mode"]), style
        for c in ("cover_m", "face_crown_bar", "face_axis_bar", "thrust_kn", "cutterhead_torque_mnm", "advance_speed_mm_min",
                  "water_bar", "z_axis_m", "face_grad_kpa_m", "grout_ratio"):
            assert np.allclose(t[c].astype(float), ref[c].astype(float), rtol=1e-9, equal_nan=True), (style, c)
        assert rep["converted_units"] == len([c for c in ad.units if c in t])


if __name__ == "__main__":
    test_styles_agree(); print("adapter styles agree")
