"""The six tools (layer 7). Each is a plain function of JSON-serialisable input and output, bound to a project.
Outputs are window-level aggregates only: the guard rejects numeric series longer than guard.max_series, and no
aggregate is formed over fewer than guard.min_rings rings, so per-ring values never leave the analyst's computer.

    from epb_twin.tools import Toolset
    ts = Toolset("project.yaml")
    ts.list_windows(n_rings=45, sort_by="soil_thick_mean")
"""
import hashlib, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from .project import load_project
from .normalise import normalise
from . import loads as L


class ToolError(ValueError):
    pass


def _py(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _sig4(x):
    return float("%.4g" % x)


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "%g" % v if abs(v) >= 1e5 else repr(float(v))
    return str(v)


BASE = [("tag", "ADV_TAG"), ("rings", None), ("face_crown_bar", "ADV_FACE_CROWN_BAR"), ("face_grad_kpa_m", "ADV_FACE_GRAD"),
        ("grout_crown_bar", "ADV_GROUT_CROWN_BAR"), ("gap_p_frac", "ADV_GAP_P_FRAC"), ("joint_f", "ADV_JOINT_F"),
        ("escale", "ADV_ESCALE"), ("z_axis", "ADV_Z_AXIS"), ("z_rock", "ADV_Z_ROCK"), ("mesh", "ADV_MESH")]
EXTRA = [("shield_w_kn", "ADV_SHIELD_W_KN"), ("backup_w_kn", "ADV_BACKUP_W_KN"), ("mc", "ADV_MC"), ("layers", "ADV_LAYERS")]
SHIELDS = {"model": "model", "declared": "declared", "adv15": "model", "record_decl": "declared"}
WINDOW_KEYS = ["drive", "ring_from", "ring_to", "cover_mean", "soil_thick_mean", "rock_cover_mean", "water_bar_mean",
               "face_crown_bar_mean", "cover_bad_rings", "grout_rings", "mixed_ground_frac", "modes"]


class Toolset:
    def __init__(self, project):
        self.p = load_project(project) if isinstance(project, (str, Path)) else project
        self._tables = {}
        raw = {"normalise_record": self.normalise_record, "list_windows": self.list_windows,
               "build_window": self.build_window, "decompose_loads": self.decompose_loads,
               "write_job": self.write_job, "summarise_results": self.summarise_results}
        self.TOOLS = {k: self._audited(k, f) for k, f in raw.items()}

    def _audited(self, name, f):
        """Tool calls through TOOLS (agents, CLI, MCP) are logged: arguments, outcome, output size and hash."""
        def call(**kw):
            rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool": name, "args": kw}
            try:
                out = f(**kw)
                txt = json.dumps(out, sort_keys=True, default=str)
                rec.update(ok=True, bytes=len(txt), sha256=hashlib.sha256(txt.encode()).hexdigest()[:16])
                return out
            except Exception as e:
                rec.update(ok=False, error=str(e)[:300]); raise
            finally:
                if self.p.audit:
                    try:
                        d = self.p.root / "_audit"; d.mkdir(exist_ok=True)
                        with open(d / "tool_calls.jsonl", "a", encoding="utf-8") as fh:
                            fh.write(json.dumps(rec, default=str) + "\n")
                    except OSError:
                        pass
        call.__name__ = name; call.__doc__ = f.__doc__
        return call

    # ---------------------------------------------------------------- helpers
    @property
    def r_exc(self):
        return self.p.machine.radius_m

    def guard(self, out, path="out"):
        mx = int(self.p.guard["max_series"])
        if isinstance(out, dict):
            for k, v in out.items():
                self.guard(v, f"{path}.{k}")
        elif isinstance(out, list):
            if len(out) > 100:
                raise ToolError(f"{path}: {len(out)} rows exceed the output limit")
            if sum(isinstance(v, (int, float)) for v in out) > mx:
                raise ToolError(f"{path}: numeric series longer than {mx}")
            for i, v in enumerate(out):
                self.guard(v, f"{path}[{i}]")
        return out

    def _min_rings(self, n):
        if n < int(self.p.guard["min_rings"]):
            raise ToolError(f"n_rings must be at least {self.p.guard['min_rings']}")

    def table(self, record=None):
        key = str(record or self.p.record)
        if key not in self._tables:
            path = Path(key)
            raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
            self._tables[key] = normalise(raw, self.p.adapter, self.r_exc)
        return self._tables[key]

    @staticmethod
    def window_stats(s):
        return {"cover_mean": s.cover_m.mean(), "cover_min": s.cover_m.min(), "cover_max": s.cover_m.max(),
                "cover_tabulated_frac": (s.cover_source == "tabulated").mean() if "cover_source" in s else np.nan,
                "cover_bad_rings": int((~s.cover_ok).sum()),
                "rock_cover_mean": s.rock_cover_m.mean() if "rock_cover_m" in s else np.nan,
                "soil_thick_mean": s.soil_thickness_m.mean(), "soil_thick_min": s.soil_thickness_m.min(),
                "soil_thick_max": s.soil_thickness_m.max(), "water_bar_mean": s.water_bar.mean(),
                "face_crown_bar_mean": s.face_crown_bar.mean(), "face_axis_bar_mean": s.face_axis_bar.mean(),
                "face_grad_kpa_m": s.face_grad_kpa_m.median(),
                "grout_ratio_mean": s.grout_ratio.mean(), "grout_rings": int(s.grout_ratio.notna().sum()),
                "overexc_post_grout_pct_mean": s.over_excavation_post_grout_pct.mean() if "over_excavation_post_grout_pct" in s else np.nan,
                "thrust_mean_kn": s.thrust_kn.mean(), "torque_mean_mnm": s.cutterhead_torque_mnm.mean(),
                "modes": "+".join(sorted(m for m in s["mode"].unique() if m)),
                "mixed_ground_frac": s.ground_mixed.mean()}

    def windows(self, t, n):
        rows = []
        for dr, g in t[t["has_context"]].groupby("drive"):
            g = g.sort_values("ring_no").reset_index(drop=True); rings = g.ring_no.to_numpy()
            for i in range(len(g) - n + 1):
                if rings[i + n - 1] - rings[i] != n - 1:
                    continue
                rows.append({"drive": dr, "ring_from": int(rings[i]), "ring_to": int(rings[i + n - 1]),
                             **self.window_stats(g.iloc[i:i + n])})
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- tools
    def normalise_record(self, record=None):
        t, rep = self.table(record)
        drives = {str(d): {"rings": int(len(g)), "ring_from": int(g.ring_no.min()), "ring_to": int(g.ring_no.max()),
                           "with_context": int(g.has_context.sum())} for d, g in t.groupby("drive")}
        return self.guard({"record": Path(record or self.p.record).name, "deviations": rep, "drives": drives})

    def list_windows(self, n_rings=None, sort_by="soil_thick_mean", descending=True, top=10, drive=None,
                     max_bad_cover_rings=0, record=None):
        n_rings = int(n_rings or self.p.window["n_rings"]); self._min_rings(n_rings)
        t, _ = self.table(record)
        w = self.windows(t, n_rings)
        if w.empty:
            return self.guard({"n_candidates": 0, "windows": []})
        if drive:
            w = w[w.drive == str(drive).strip().upper()]
        w = w[w.cover_bad_rings <= max_bad_cover_rings]
        if sort_by not in w:
            raise ToolError(f"sort_by must be one of {sorted(c for c in w.columns if c not in ('drive', 'modes'))}")
        w = w.sort_values(sort_by, ascending=not descending)
        rows = [{k: _py(round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
                for r in w[WINDOW_KEYS].head(int(top)).to_dict("records")]
        return self.guard({"n_candidates": int(len(w)), "sorted_by": sort_by, "windows": rows})

    def build_window(self, drive, ring_from, n_rings=None, name="W", grout_overpressure_bar=None, record=None):
        """Case line for the advance model: crown face pressure (window mean), face gradient (median), grout = face +
        overpressure, axis depth = cover + R, rock head = soil thickness; 4 significant figures before rounding."""
        n_rings = int(n_rings or self.p.window["n_rings"]); self._min_rings(n_rings)
        gop = float(self.p.window["grout_overpressure_bar"] if grout_overpressure_bar is None else grout_overpressure_bar)
        t, _ = self.table(record)
        name = str(name).strip().upper(); drive = str(drive).strip().upper(); ring_from = int(ring_from)
        g = t[(t.drive.astype(str).str.upper() == drive) & t.has_context & t.ring_no.between(ring_from, ring_from + n_rings - 1)]
        if len(g) != n_rings or g.ring_no.diff().fillna(1).ne(1).any():
            raise ToolError(f"{drive} {ring_from}-{ring_from + n_rings - 1}: {len(g)} of {n_rings} rings have context "
                            "(window must be contiguous)")
        s = self.window_stats(g)
        crown = round(_sig4(s["face_crown_bar_mean"]), 2)
        case = {"tag": f"{name}_r{n_rings}", "rings": n_rings, "face_crown_bar": crown,
                "face_grad_kpa_m": round(_sig4(s["face_grad_kpa_m"]), 1), "grout_crown_bar": round(crown + gop, 2),
                "gap_p_frac": 1.0, "joint_f": self.p.machine.joint_factor, "escale": 1.0,
                "z_axis": -round(_sig4(s["cover_mean"]) + self.r_exc, 3), "z_rock": -_sig4(s["soil_thick_mean"]),
                "mesh": f"_{name}"}
        flags = []
        if s["cover_bad_rings"]:
            flags.append(f"{s['cover_bad_rings']} rings with cover spread > {self.p.adapter.max_cover_spread_m} m")
        if s["grout_rings"] < 0.7 * n_rings:
            flags.append(f"grout logged on {s['grout_rings']} of {n_rings} rings")
        if s["mixed_ground_frac"] > 0.5:
            flags.append("mixed face on most rings")
        if "+" in s["modes"]:
            flags.append(f"operating mode changes within the window ({s['modes']})")
        if int(g.ring_no.max()) == int(t[t.drive.astype(str).str.upper() == drive].ring_no.max()):
            flags.append("window ends at the last ring of the drive in the record")
        stats = {k: _py(round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
        return self.guard({"window": {"drive": drive, "ring_from": ring_from, "ring_to": ring_from + n_rings - 1},
                           "case": case, "stats": stats, "flags": flags,
                           "mesh_command": f"ADV_Z_AXIS={case['z_axis']} ADV_Z_ROCK={case['z_rock']} ADV_MESH={case['mesh']} "
                                           f"{self.model_cmd()} mesh --rings {n_rings}"})

    def ring_terms_frame(self, t, shield="model"):
        if shield not in SHIELDS:
            raise ToolError(f"shield must be one of {list(SHIELDS)}")
        M = self.p.machine
        length = M.declared_shield_length_m if SHIELDS[shield] == "declared" else None
        if SHIELDS[shield] == "declared" and length is None:
            raise ToolError("the machine file has no shield.declared_length_m")
        g = L.MachineGeometry(**M.load_geometry(length)); P = L.ResponseParameters(); rows = []
        for _, r in t[t["cover_ok"] & (t["thrust_kn"] > 0)].iterrows():
            out = {"drive": r["drive"], "ring_no": int(r["ring_no"])}
            for lev in L.LEVELS:
                terms = L.ring_terms(float(r["z_axis_m"]), self.p.load_ground, g, float(r["advance_speed_mm_min"]),
                                     float(r["cutterhead_speed_rpm"]), float(r["face_axis_kpa"]), P.at(lev),
                                     gw_depth_m=float(r["gw_depth_m"]))
                out[f"thrust_{lev}_kn"] = sum(v for k, v in terms.items() if k.startswith("thrust_"))
                out[f"torque_{lev}_knm"] = sum(v for k, v in terms.items() if k.startswith("torque_"))
                if lev == "mid":
                    out.update(terms)
            out["thrust_meas_kn"] = r["thrust_kn"]
            out["contact_meas_kn"] = r["contact_force_kn"] if "contact_force_kn" in r else np.nan
            out["torque_meas_knm"] = r["cutterhead_torque_mnm"] * 1000
            rows.append(out)
        return pd.DataFrame(rows)

    def decompose_loads(self, drive, ring_from=None, ring_to=None, shield="model", record=None, geometry=None):
        """Thrust and torque terms of the load model against the logged thrust, contact force and torque, aggregated
        over the rings (of the window, or of the drive) with a usable cover. shield: 'model' or 'declared'."""
        shield = geometry or shield
        t, _ = self.table(record)
        drive = str(drive).strip().upper()
        s = t[t.drive.astype(str).str.upper() == drive]
        if ring_from is not None:
            s = s[s.ring_no.between(int(ring_from), int(ring_to) if ring_to is not None else int(ring_from) + self.p.window["n_rings"] - 1)]
        d = self.ring_terms_frame(s, shield)
        if len(d) < int(self.p.guard["min_rings"]):
            raise ToolError(f"only {len(d)} rings with usable cover (min {self.p.guard['min_rings']})")
        tr = d.thrust_mid_kn / d.thrust_meas_kn; qr = d.torque_mid_knm / d.torque_meas_knm
        fric = d[["thrust_skin_kn", "thrust_shield_kn", "thrust_seal_kn", "thrust_backup_kn"]].sum(axis=1)
        out = {"drive": drive, "shield": SHIELDS[shield], "rings": int(len(d)),
               "thrust_ratio_mean": round(float(tr.mean()), 3),
               "thrust_in_band": int(d.thrust_meas_kn.between(d.thrust_low_kn, d.thrust_high_kn).sum()),
               "torque_ratio_mean": round(float(qr.mean()), 3),
               "torque_in_band": int(d.torque_meas_knm.between(d.torque_low_knm, d.torque_high_knm).sum()),
               "share_mid": {k: round(float((d[k] / d.thrust_mid_kn).mean()), 3) for k in
                             ["thrust_face_kn", "thrust_cutting_kn", "thrust_shield_kn", "thrust_skin_kn",
                              "thrust_seal_kn", "thrust_backup_kn"]},
               "thrust_meas_mean_MN": round(float(d.thrust_meas_kn.mean()) / 1e3, 1),
               "torque_meas_mean_MNm": round(float(d.torque_meas_knm.mean()) / 1e3, 1)}
        if d.contact_meas_kn.notna().any():
            out.update({"meas_contact_share": round(float((d.contact_meas_kn / d.thrust_meas_kn).mean()), 3),
                        "model_cutting_over_contact": round(float((d.thrust_cutting_kn / d.contact_meas_kn).mean()), 3),
                        "model_face_over_meas_noncontact": round(float((d.thrust_face_kn / (d.thrust_meas_kn - d.contact_meas_kn)).mean()), 3),
                        "meas_remainder_mean_kn": round(float((d.thrust_meas_kn - d.contact_meas_kn - d.thrust_face_kn).mean()), 0)})
        out["model_friction_mean_kn"] = round(float(fric.mean()), 0)
        if ring_from is not None:
            out["window"] = {"ring_from": int(s.ring_no.min()), "ring_to": int(s.ring_no.max())}
        return self.guard(out)

    def model_cmd(self):
        """'python -u advance.py' for a script, 'python -u -m epb_twin.fe.advance' for a module (the default)."""
        ms = self.p.runner["model_script"]
        return f"python -u {ms}" if ms.endswith(".py") else f"python -u -m {ms}"

    def case_columns(self):
        mats = list(self.p.ground.materials)
        per_mat = []
        for m in mats:
            u = m.upper()
            per_mat += [(f"e_{m.lower()}_pa", f"ADV_E_{u}"), (f"c_{m.lower()}_kpa", f"ADV_C_{u}"), (f"phi_{m.lower()}_deg", f"ADV_PHI_{u}")]
        return BASE, EXTRA + per_mat

    def case_defaults(self):
        M, G = self.p.machine, self.p.ground
        d = {"gap_p_frac": 1.0, "joint_f": M.joint_factor, "escale": 1.0, "mesh": "", "shield_w_kn": M.shield_weight_kn,
             "backup_w_kn": M.backup_weight_kn, "mc": 0, "layers": ""}
        for n, m in G.materials.items():
            d[f"e_{n.lower()}_pa"] = m.E_MPa * 1e6; d[f"c_{n.lower()}_kpa"] = m.c_kPa; d[f"phi_{n.lower()}_deg"] = m.phi_deg
        return d

    def write_job(self, cases, name, comment="", hours=6, out_dir=None):
        """cases: list of case dicts. Writes cases_<name>.csv and run_<name>.sh (slurm array or local loop) when
        out_dir is given; always returns their text and the submit line."""
        if not cases:
            raise ToolError("no cases")
        base, extra = self.case_columns(); defaults = self.case_defaults()
        keys = set().union(*cases)
        known = {c for c, _ in base} | {c for c, _ in extra}
        unknown = keys - known
        if unknown:
            raise ToolError(f"unknown case keys {sorted(unknown)}; known: {sorted(known)}")
        cols = list(base) + [c for c in extra if c[0] in keys]
        tags = [c.get("tag") for c in cases]
        if any(not t for t in tags) or len(set(tags)) != len(tags):
            raise ToolError("every case needs a unique tag")
        for c in cases:
            for k in ("rings", "face_crown_bar", "face_grad_kpa_m", "grout_crown_bar", "z_axis", "z_rock"):
                if c.get(k) is None:
                    raise ToolError(f"{c['tag']}: {k} missing")
            if not (0 < float(c["face_crown_bar"]) < 6 and float(c["grout_crown_bar"]) >= float(c["face_crown_bar"]) - 1e-9):
                raise ToolError(f"{c['tag']}: face {c['face_crown_bar']} / grout {c['grout_crown_bar']} bar out of range")
            if float(c["z_axis"]) > -self.r_exc:
                raise ToolError(f"{c['tag']}: axis {c['z_axis']} m leaves less than R of cover")
            if int(c.get("mc", 0) or 0) not in (0, 1):
                raise ToolError(f"{c['tag']}: mc must be 0 (elastic) or 1 (Mohr-Coulomb)")
        csv_name, sh_name = f"cases_{name}.csv", f"run_{name}.sh"
        lines = [",".join(c for c, _ in cols)] + [",".join(fmt(c.get(k, defaults.get(k))) for k, _ in cols) for c in cases]
        csv_text = "\n".join(lines) + "\n"
        vars_ = [f"C{i}" for i in range(len(cols))]
        export = "export " + " ".join(f'{e}="${v}"' for (k, e), v in zip(cols, vars_) if e)
        export += ' ADV_MACHINE="$HERE/machine.yaml" ADV_GROUND="$HERE/ground.yaml" ADV_SHIELD_MODE=contact ADV_LOADS=sum ADV_LINING=solid'
        rings_var = vars_[[c for c, _ in cols].index("rings")]; tag_var = vars_[0]
        run = f'{self.model_cmd()} run --rings "${rings_var}" --steps 4'
        pat = self.p.runner["summary_pattern"].replace("{tag}", "${" + tag_var + "}")
        body = [f'IFS=, read -r {" ".join(vars_)} <<< "$LINE"', export, 'cd "$HERE"',
                f'if [ -f "{self.p.runner["results_dir"]}/{pat}" ]; then echo "${tag_var} done"; exit 0; fi',
                f'echo "case ${tag_var}: ${rings_var} rings on $(hostname)"', run]
        n = len(cases); r = self.p.runner
        if r["kind"] == "slurm":
            s = r["slurm"]
            head = ["#!/bin/bash", f"# {comment or name}: one array task = one line of {csv_name} ({n} cases)",
                    f"#   sbatch --array=1-{n} {sh_name}", f"#SBATCH --job-name={name[:12]}"]
            head += [f"#SBATCH --account={s['account']}"] if s.get("account") else []
            head += [f"#SBATCH --partition={s['partition']}"] if s.get("partition") else []
            head += ["#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={s['cpus']}", f"#SBATCH --mem={s['mem']}",
                     f"#SBATCH --time={int(hours):02d}:00:00", "#SBATCH --output=logs/%x_%A_%a.out", "set -e",
                     'HERE="${SLURM_SUBMIT_DIR}"']
            head += [f"source {s['python_env']}"] if s.get("python_env") else []
            head += ["export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_WAIT_POLICY=passive",
                     f'LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$HERE/{csv_name}")']
            script = "\n".join(head + body) + "\n"; submit = f"sbatch --array=1-{n} {sh_name}"
        else:
            script = "\n".join(["#!/bin/bash", f"# {comment or name}: runs the {n} cases of {csv_name} one after another",
                                'HERE="$(cd "$(dirname "$0")" && pwd)"', f'tail -n +2 "$HERE/{csv_name}" | while read -r LINE; do (',
                                "set -e"] + body + [") || echo \"case failed\"; done"]) + "\n"
            submit = f"bash {sh_name}"
        warnings = [f"{t}: results exist already ({self.p.runner['summary_pattern'].format(tag=t)}); the task would skip this case (use a new tag)"
                    for t in tags if (self.p.results_dir / self.p.runner["summary_pattern"].format(tag=t)).exists()]
        meshes = sorted({(c["rings"], c.get("mesh", ""), c["z_axis"], c["z_rock"]) for c in cases})
        out = {"csv_name": csv_name, "script_name": sh_name, "columns": [c for c, _ in cols], "warnings": warnings,
               "n_cases": n, "csv": csv_text, "script": script, "submit": submit,
               "mesh_commands": [f"ADV_Z_AXIS={za} ADV_Z_ROCK={zr} ADV_MESH={m} {self.model_cmd()} mesh --rings {rg}"
                                 for rg, m, za, zr in meshes],
               "run_dir_needs": [csv_name, sh_name, "machine.yaml", "ground.yaml"] + ([self.p.runner["model_script"]] if self.p.runner["model_script"].endswith(".py") else []),
               "environment": "epb_twin installed with the [fe] extra (pip or the container in containers/)"}
        if out_dir:
            od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
            for fn, txt in ((csv_name, csv_text), (sh_name, script)):
                p = od / fn
                if p.exists() and p.read_text() != txt:
                    raise ToolError(f"{p} exists with different content; choose another name")
                p.write_text(txt, newline="\n")
            out["written_to"] = str(od)
        return out

    @staticmethod
    def metrics(path, old_rings=20):
        d = json.load(open(path)); a = d["assumed"]; lf = d["lining_forces"]; last = lf[list(lf)[-1]]
        old = [last["ring%d" % i] for i in range(1, old_rings + 1) if "ring%d" % i in last]
        sg = d["shield_ground_force"]; fin = sg[list(sg)[-1]]
        return {"tag": d["tag"], "grout_crown_bar": a.get("grout_crown_bar"), "gap_pressure_frac": a.get("gap_pressure_frac"),
                "lining_E_eff_GPa": a.get("lining_E_eff_GPa"), "S_max_mm": d["surface_settlement_max_mm"],
                "u_shield_mean_mm": fin["convergence_mm"]["mean"], "u_shield_max_mm": fin["convergence_mm"]["max"],
                "u_shield_max_any_stage_mm": max(v["convergence_mm"]["max"] for v in sg.values()),
                "shield_contact_frac_max": max(v["contact_area_frac"] for v in sg.values()),
                "N_hoop_kN_m": round(float(np.mean([-r["N_hoop_kN_m"][1] for r in old])), 1),
                "M_max_kNm_m": max(max(abs(x) for x in r["M_hoop_kNm_m"]) for r in last.values()), "seconds": d.get("seconds")}

    def summarise_results(self, tags, base=None):
        """Metrics of finished runs: settlement, shield-zone convergence and contact, lining hoop force and moment;
        with base, the change of each in per cent."""
        rd = self.p.results_dir; pat = self.p.runner["summary_pattern"]
        out, missing = {}, []
        for t in ([base] if base else []) + [x for x in tags if x != base]:
            p = rd / pat.format(tag=t)
            if not p.exists():
                missing.append(t); continue
            try:
                out[t] = {k: _py(v) for k, v in self.metrics(p).items()}
            except (KeyError, IndexError) as e:
                out[t] = {"error": f"summary lacks {e}"}
        if base and base in out and "error" not in out[base]:
            b = out[base]
            for t, m in out.items():
                if t != base and "error" not in m:
                    m["change_pct"] = {k: round(100 * (m[k] - b[k]) / b[k], 1) for k in
                                       ("S_max_mm", "u_shield_max_mm", "N_hoop_kN_m", "M_max_kNm_m") if b[k]}
        return self.guard({"results": out, "missing": missing})
