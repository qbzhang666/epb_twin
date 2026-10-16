"""A synthetic EPB drive (layer 3): a complete demonstration project that needs no project data.

    python -m epb_twin demo ./demo_project

writes project.yaml, adapter.yaml, machine.yaml, ground.yaml, data/record.csv (raw, in the adapter's column names)
and results/*.json (synthetic run summaries for the summary tool). Everything is generated from a seed; the drive,
its ground and its numbers are fictitious."""
import json, math, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from . import loads as L
from .spec import load_machine

HERE = Path(__file__).resolve().parent
SPECS = HERE / "specs"

GROUND_YAML = """# Synthetic ground: soil over weathered rock over rock. Values are illustrative.
name: demo_ground
materials:
  Soil: {E_MPa: 20, nu: 0.35, gamma: 18.5, K0: 0.5, c_kPa: 5, phi_deg: 30}
  Rock: {E_MPa: 150, nu: 0.28, gamma: 23.0, K0: 0.7, c_kPa: 100, phi_deg: 35}
default_layers: [Soil, Rock]
load_model: {kind: rock, gamma: 23.0, K0: 0.45, ucs_MPa: 25}
"""
PROJECT_YAML = """# Demonstration project (synthetic drive). Copy this folder to start a real project and replace the files.
name: demo_drive
record: data/record.csv
adapter: adapter.yaml
machine: machine.yaml
ground: ground.yaml
window: {n_rings: 30, grout_overpressure_bar: 0.5}
runner:
  kind: local
  model_script: epb_twin.fe.advance
  results_dir: results
  summary_pattern: "adv_{tag}_summary.json"
guard: {min_rings: 5, max_series: 20}
"""


def _smooth(rng, n, lo, hi, waves=3):
    x = np.linspace(0, 1, n); y = np.zeros(n)
    for k in range(1, waves + 1):
        y += rng.normal(0, 1 / k) * np.sin(2 * math.pi * k * x + rng.uniform(0, 2 * math.pi))
    y = (y - y.min()) / (np.ptp(y) or 1)
    return lo + (hi - lo) * y


def drive_table(machine, seed=7, drives=(("D1", 1, 260), ("D2", 1, 220))):
    """Standard-name table of a synthetic drive (clean)."""
    rng = np.random.default_rng(seed); R = machine.radius_m; rows = []
    lg = L.LoadGround("rock", 23.0, 0.45, ucs_MPa=25.0)
    geo = L.MachineGeometry(**machine.load_geometry()); P = L.ResponseParameters().at("mid")
    a_ring = math.pi / 4 * (machine.excavated_diameter_m ** 2 - machine.lining_extrados_diameter_m ** 2) * machine.ring_length_m
    for drive, r0, n in drives:
        cover = _smooth(rng, n, 9.0, 22.0); soil = _smooth(rng, n, 3.0, 20.0); gw = 3.0
        mode = np.where(_smooth(rng, n, 0, 1) > 0.8, "semi_EPB", "full_EPB")
        for i in range(n):
            c, s = cover[i], soil[i]; crown, invert = c, c + 2 * R
            if s <= crown:
                gset = "R" if s < crown - 3 else "W/R"
            elif s >= invert:
                gset = "S"
            else:
                gset = "S/W" if s < c + R else "S/R"
            water = max(c - gw, 0) * 9.81 / 100
            sv = 19.0 * c - 9.81 * max(c - gw, 0)
            p_crown = water + 0.35 * sv / 100 + rng.normal(0, 0.05)
            p_axis = p_crown + 11.0 * R / 100 + rng.normal(0, 0.03)
            adv = rng.uniform(25, 45); rpm = rng.uniform(1.3, 2.0)
            terms = L.ring_terms(c + R, lg, geo, adv, rpm, p_axis * 100, P, gw_depth_m=gw)
            thrust = sum(v for k, v in terms.items() if k.startswith("thrust_")) * rng.uniform(0.75, 1.0)
            torque = sum(v for k, v in terms.items() if k.startswith("torque_")) / 1000 * rng.uniform(0.8, 1.3)
            contact = thrust * rng.uniform(0.25, 0.40)
            grout = a_ring * rng.uniform(1.0, 1.4) if rng.random() > 0.08 else np.nan
            rows.append({"drive": drive, "ring_no": r0 + i, "source_rev": 0, "cover_m": round(c, 2),
                         "cover_spread_m": 2.4 if rng.random() < 0.04 else 0.2, "cover_source": "tabulated",
                         "rock_cover_m": round(max(c - s, 0.0), 2), "soil_thickness_m": round(s, 2),
                         "water_bar": round(water, 2), "ground_set": gset, "mode": str(mode[i]),
                         "face_crown_bar": round(p_crown, 3), "face_axis_bar": round(p_axis, 3), "thrust_kn": round(thrust, 0),
                         "contact_force_kn": round(contact, 0), "cutterhead_torque_mnm": round(torque, 2),
                         "advance_speed_mm_min": round(adv, 1), "cutterhead_speed_rpm": round(rpm, 2),
                         "penetration_mm_rev": round(adv / rpm, 1), "total_grout_m3": round(grout, 2) if np.isfinite(grout) else np.nan,
                         "theoretical_grout_m3": round(a_ring, 2), "over_excavation_post_grout_pct": round(rng.normal(0.5, 1.5), 2),
                         "has_context": True})
    t = pd.DataFrame(rows)
    t.loc[(t.drive == "D1") & (t.ring_no <= 12), ["cover_m", "has_context"]] = [np.nan, False]   # no context at the start
    t = t[~((t.drive == "D1") & t.ring_no.between(150, 155))].reset_index(drop=True)            # a gap in the record
    return t


def _summary(tag, S, u, N, M, a):
    rings = {f"ring{k}": {"N_hoop_kN_m": [-N * 1.1, -N, -N * 0.9], "M_hoop_kNm_m": [-M * 0.2, M]} for k in range(1, 21)}
    return {"tag": tag, "synthetic": True, "assumed": a, "surface_settlement_max_mm": round(S, 2),
            "shield_ground_force": {"ring30": {"convergence_mm": {"mean": round(u * 0.4, 1), "max": round(u, 1)}, "contact_area_frac": 0.0}},
            "lining_forces": {"ring30": rings}, "seconds": 0}


def write_demo(out_dir, seed=7):
    out = Path(out_dir); (out / "data").mkdir(parents=True, exist_ok=True); (out / "results").mkdir(exist_ok=True)
    shutil.copy(SPECS / "machine_example_7m.yaml", out / "machine.yaml")
    shutil.copy(HERE / "adapters" / "example.yaml", out / "adapter.yaml")
    (out / "ground.yaml").write_text(GROUND_YAML); (out / "project.yaml").write_text(PROJECT_YAML)
    m = load_machine(out / "machine.yaml")
    t = drive_table(m, seed)
    raw = t.rename(columns={"face_crown_bar": "p_crown_bar", "face_axis_bar": "p_axis_bar", "thrust_kn": "thrust_kN",
                            "cutterhead_torque_mnm": "torque_MNm", "ground_set": "ground_class", "mode": "operating_mode"})
    raw["ground_class"] = raw["ground_class"].map({"S": "S", "R": "R", "S/W": "S/W", "S/R": "S/R", "W/R": "W/R"})
    raw.to_csv(out / "data" / "record.csv", index=False)
    rng = np.random.default_rng(seed + 1)
    runs = {"A_base": 1.0, "A_re75": 1.55, "A_re300": 0.72, "A_gp05": 1.9, "A_gp0": 2.6, "A_j03": 1.04, "A_fe40": 0.93}
    for tag, f in runs.items():
        S = 9.0 * f; u = 15.0 * f ** 0.8; N = 900 / f ** 0.3; M = 30 * f ** 0.5
        a = {"grout_crown_bar": 2.2, "gap_pressure_frac": {"A_gp05": 0.5, "A_gp0": 0.0}.get(tag, 1.0), "lining_E_eff_GPa": 21.0}
        json.dump(_summary(tag, S * rng.uniform(0.99, 1.01), u, N, M, a), open(out / "results" / f"adv_{tag}_summary.json", "w"), indent=1)
    return out
