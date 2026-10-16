"""Layer-1 tests on a small synthetic record (run: python -m pytest tests, or python tests/test_normalise.py)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epb_twin import load_adapter, normalise, validate, to_number

AD = load_adapter(Path(__file__).resolve().parents[1] / "epb_twin" / "adapters" / "example.yaml")
R = 7.0


def raw():
    return pd.DataFrame({
        "drive": ["D1"] * 5, "ring_no": [1, 2, 2, 3, 5], "source_rev": [0, 0, 1, 0, 0],
        "cover_m": ["12.0", "12.4", "12.5", "11.0-13.0", 12.8], "water_bar": [0.8, None, 0.9, 0.9, 1.0],
        "ground_class": ["Fill", "S, s", "S/W", "weathered (W) over R", None],
        "operating_mode": ["Full EPB", "semi-EPB", "air bubble", "", "EPB"],
        "p_crown_bar": ["1,8", "1.9 bar", 1.9, 2.0, 2.0], "p_axis_bar": [2.2, 2.3, 2.3, 2.4, 2.4],
        "thrust_kN": [5e4] * 5, "torque_MNm": [10.0] * 5, "advance_speed_mm_min": [40] * 5,
        "cutterhead_speed_rpm": [1.5] * 5, "total_grout_m3": [20, None, 21, 22, 22], "theoretical_grout_m3": [20] * 5})


def test_to_number():
    assert to_number("2,54") == 2.54 and to_number("2.54 bar") == 2.54 and to_number("17.2-21.8") == 19.5
    assert np.isnan(to_number(""))


def test_normalise_example():
    t, rep = normalise(raw(), AD, R)
    assert list(t.ring_no) == [1, 2, 3, 5] and rep["duplicate_revisions"] == 1
    assert t.loc[t.ring_no == 2, "source_rev"].item() == 1                 # latest revision kept
    assert list(t.ground_set) == ["F", "S/W", "W/R", ""]
    assert list(t["mode"]) == ["full_EPB", "EPB_air_bubble", "", "full_EPB"]
    assert t.loc[t.ring_no == 3, "cover_m"].item() == 12.0 and t.loc[t.ring_no == 3, "cover_spread_m"].item() == 2.0
    assert not t.loc[t.ring_no == 3, "cover_ok"].item()                     # spread above the adapter limit
    assert t.face_crown_bar.tolist()[0] == 1.8 and rep["text_numbers"] >= 3
    assert t.gap_before.tolist() == [False, False, False, True]
    assert np.isclose(t.z_axis_m.iloc[0], 12.0 + R)
    v = validate(t)
    assert v["missing_required"] == [] and v["rows_without_ring"] == 0


if __name__ == "__main__":
    test_to_number(); test_normalise_example(); print("epb_twin layer-1 tests passed")
