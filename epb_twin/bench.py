"""Benchmark (layer 3): tasks on record variants with seeded deviations of form, scored against the clean record;
pass^k = share of tasks solved in all k trials (Yao et al. 2025). The tasks are built from the project itself, so the
same benchmark runs on the synthetic demo and, privately, on a project's own record."""
import csv, io, json, math, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from .tools import Toolset, ToolError
from .deviations import make_variant

CASE_KEYS = ("tag", "rings", "face_crown_bar", "face_grad_kpa_m", "grout_crown_bar", "z_axis", "z_rock", "mesh")
HARD_FORMAT = ('\nAnswer as {"status": "done" | "infeasible" | "needs_clarification", "result": {%s} (when done), '
               '"note": short explanation}.')


class NaiveToolset(Toolset):
    """Reads the record with plain parsing: column mapping only, no normalisation of the deviations."""

    def table(self, record=None):
        key = str(record or self.p.record)
        if key not in self._tables:
            raw = self.p.adapter.rename(pd.read_csv(key)); R = self.r_exc
            for c in ("cover_m", "face_crown_bar", "face_axis_bar", "soil_thickness_m", "total_grout_m3", "thrust_kn"):
                if c in raw:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
            raw["ground_set"] = raw.pop("ground_class") if "ground_class" in raw else raw.get("ground_set", "")
            raw["mode"] = raw.pop("operating_mode") if "operating_mode" in raw else raw.get("mode", "")
            raw["mode"] = raw["mode"].fillna("").astype(str); raw["ground_set"] = raw["ground_set"].fillna("").astype(str)
            raw["z_axis_m"] = raw.cover_m + R; raw["face_axis_kpa"] = raw.face_axis_bar * 100
            raw["face_crown_kpa"] = raw.face_crown_bar * 100; raw["face_grad_kpa_m"] = (raw.face_axis_kpa - raw.face_crown_kpa) / R
            raw["grout_ratio"] = raw.total_grout_m3 / raw.theoretical_grout_m3
            raw["has_context"] = raw.get("has_context", raw.cover_m.notna()).astype(bool)
            raw["cover_ok"] = raw.has_context; raw["ground_mixed"] = False; raw["gw_depth_m"] = 5.0
            self._tables[key] = (raw, {})
        return self._tables[key]


def _close(a, b, tol):
    try:
        return math.isclose(float(a), float(b), abs_tol=tol)
    except (TypeError, ValueError):
        return False


def _csv_rows(text):
    return list(csv.DictReader(io.StringIO(text or "")))


def _case_ok(a, r):
    a = a or {}
    bad = [k for k in CASE_KEYS if not (str(a.get(k)) == str(r[k]) if isinstance(r[k], str) else _close(a.get(k), r[k], 1e-6))]
    return not bad, f"wrong {bad}" if bad else "ok"


def _csv_ok(a, r, skip=()):
    a, r = _csv_rows(a), _csv_rows(r)
    if len(a) != len(r):
        return False, f"{len(a)} rows, expected {len(r)}"
    for ra, rr in zip(a, r):
        for k, v in rr.items():
            if k not in skip and not (ra.get(k) == v or _close(ra.get(k), v, 1e-6)):
                return False, f"{rr.get('tag')}.{k}: {ra.get(k)} != {v}"
    return True, "ok"


def make_tasks(ts):
    """Basic and hard task sets for a project, with windows chosen from its clean record."""
    t, _ = ts.table(); n = int(ts.p.window["n_rings"]); drives = sorted(map(str, t.drive.unique()))
    d1, d2 = drives[0], drives[-1]
    wa = ts.list_windows(n, "soil_thick_mean", True, 1, d1, 0)["windows"][0]
    wb_all = ts.list_windows(n, "cover_mean", False, 50, d2, 0)["windows"]
    wb = next(w for w in wb_all if d2 != d1 or w["ring_from"] > wa["ring_to"] or w["ring_to"] < wa["ring_from"])
    A, B = (d1, wa["ring_from"]), (d2, wb["ring_from"])
    M = ts.p.machine; mats = list(ts.p.ground.materials); rock = ts.p.ground.default_layers[-1]
    wts = {"shield_w_kn": M.shield_weight_kn, "backup_w_kn": M.backup_weight_kn, "mc": 0}
    caseA = lambda s: s.build_window(*A, n, "A")["case"]
    caseB = lambda s: s.build_window(*B, n, "B")["case"]
    rd = ts.p.results_dir; pat = ts.p.runner["summary_pattern"]
    tags = sorted(p.name[len(pat.split("{tag}")[0]):-len(pat.split("{tag}")[1])] for p in rd.glob(pat.replace("{tag}", "*"))) if rd.exists() else []
    base = next((x for x in tags if x.endswith("_base")), None); sweeps = [x for x in tags if base and x != base and x.split("_")[0] == base.split("_")[0]]
    basic = [
        {"id": "case_A", "fields": "case (object with the case-line keys)", "kind": "case",
         "prompts": [f"Prepare the advance-model case line for a {n}-ring window on drive {A[0]} starting at ring {A[1]}; name it A.",
                     f"Need the model input for {A[0]}, rings {A[1]} to {A[1] + n - 1}. Window name: a.",
                     f"drive {A[0].lower()}, {n} rings from ring {A[1]}: build the case, call the window A"],
         "solve": lambda s: {"case": caseA(s)}},
        {"id": "case_B", "fields": "case (object with the case-line keys)", "kind": "case",
         "prompts": [f"Build the {n}-ring case on drive {B[0]} from ring {B[1]}, name B.",
                     f"Case line please: drive {B[0]}, first ring {B[1]}, {n} rings, label B",
                     f"Window B = drive {B[0]} rings {B[1]}-{B[1] + n - 1}. Give its advance-model case."],
         "solve": lambda s: {"case": caseB(s)}},
        {"id": "thickest_soil", "fields": "drive, ring_from", "kind": "pick",
         "prompts": [f"Which {n}-ring window on drive {d1} has the thickest soil over rock, with no ring of doubtful cover? Give drive and first ring.",
                     f"On {d1}, find the {n}-ring stretch with the deepest soil above the rock head (exclude windows with cover-spread problems).",
                     f"Thickest soil over rock, drive {d1}, {n} rings, clean cover only: where does that window start?"],
         "solve": lambda s: {k: s.list_windows(n, "soil_thick_mean", True, 1, d1, 0)["windows"][0][k] for k in ("drive", "ring_from")}},
        {"id": "thrust_split", "fields": "meas_contact_share, thrust_ratio_mean", "kind": "num", "tol": 0.005,
         "prompts": [f"Drive {d1}, model shield: what share of the logged thrust is cutterhead contact force, and what is the mean predicted/measured thrust ratio?",
                     f"For {d1} with the model shield give the contact-force fraction of thrust and the mean thrust ratio (model over record).",
                     f"Decompose thrust on drive {d1} (model shield): report meas_contact_share and thrust_ratio_mean."],
         "solve": lambda s: {k: s.decompose_loads(d1, shield="model")[k] for k in ("meas_contact_share", "thrust_ratio_mean")}},
        {"id": "job_AB", "fields": "csv (the cases CSV text)", "kind": "csv",
         "prompts": [f"Write a job named agt for windows A ({A[0]} from ring {A[1]}) and B ({B[0]} from ring {B[1]}), {n} rings each, with the machine weights (shield {M.shield_weight_kn:g} kN, back-up {M.backup_weight_kn:g} kN), elastic ground.",
                     f"Job 'agt': cases A = {A[0]} ring {A[1]} and B = {B[0]} ring {B[1]} ({n} rings), shield weight {M.shield_weight_kn:g} kN and back-up {M.backup_weight_kn:g} kN, no Mohr-Coulomb.",
                     f"Prepare the run (name agt) for windows A and B: {A[0]} from {A[1]}, {B[0]} from {B[1]}, {n} rings, weights {M.shield_weight_kn:g}/{M.backup_weight_kn:g} kN, elastic."],
         "solve": lambda s: {"csv": s.write_job([{**caseA(s), **wts}, {**caseB(s), **wts}], "agt")["csv"]}},
    ]
    if base and sweeps:
        run = sweeps[0]
        basic.append({"id": "sweep_effect", "fields": "S_max_change_pct", "kind": "num", "tol": 0.15,
                      "prompts": [f"By how many per cent does the surface settlement change in run {run} against the base run {base}?",
                                  f"Settlement change, % : {run} relative to {base}.",
                                  f"Compare runs {run} and {base} (base): percentage change in maximum surface settlement?"],
                      "solve": lambda s: {"S_max_change_pct": s.summarise_results([run], base=base)["results"][run]["change_pct"]["S_max_mm"]}})
    hard = []
    wc = ts.list_windows(n, "cover_mean", True, 100, None, n)["windows"]
    wc = [w for w in wc if w["cover_bad_rings"] > 0]
    if wc:
        C = (wc[0]["drive"], wc[0]["ring_from"])
        hard.append({"id": "flag_C", "fields": "case (case-line object), warnings (list of strings)", "kind": "case_warn",
                     "prompts": [f"Build the {n}-ring case C on drive {C[0]} from ring {C[1]} and tell me if anything in the record makes it doubtful.",
                                 f"Case line for {C[0]} {C[1]}-{C[1] + n - 1} (call it C). Any data-quality concerns before it runs?",
                                 f"C = drive {C[0]}, first ring {C[1]}, {n} rings. Give the case and list the warnings."],
                     "solve": lambda s: {"status": "done", "result": {"case": s.build_window(*C, n, "C")["case"],
                                                                      "warnings": s.build_window(*C, n, "C")["flags"]}}})
    last = int(t[t.drive.astype(str) == d2].ring_no.max()); r9 = last - n // 2
    hard += [
        {"id": "infeasible", "fields": "case (case-line object)", "kind": "status", "status": "infeasible",
         "prompts": [f"Prepare a {n}-ring case on drive {d2} starting at ring {r9}, name Z.",
                     f"Need case Z: {d2} from ring {r9}, {n} rings.", f"Drive {d2}, rings {r9} onward, {n}-ring window Z: build the case line."],
         "solve": lambda s: _status_of(lambda: s.build_window(d2, r9, n, "Z"))},
        {"id": "override", "fields": "csv (the cases CSV text)", "kind": "csv_override",
         "prompts": [f"Write a job 'ovr' for A ({A[0]} from ring {A[1]}, {n} rings) but with the crown face pressure set to 250 kPa and the grout 50 kPa above it; everything else from the record.",
                     f"Job ovr: window A = {A[0]} ring {A[1]}, {n} rings; override face pressure at the crown to 0.25 MPa, grout 0.05 MPa higher. Keep the record values otherwise.",
                     f"Run 'ovr', one case: A ({A[0]}, ring {A[1]}, {n} rings) with p_crown = 250 kPa and grout = p_crown + 50 kPa."],
         "solve": lambda s: {"status": "done", "result": {"csv": s.write_job([{**caseA(s), "face_crown_bar": 2.5, "grout_crown_bar": 3.0}], "ovr")["csv"]}}},
        {"id": "compare", "fields": "higher (window name), difference (absolute difference of the mean ratios)", "kind": "compare",
         "prompts": [f"With the model shield, which window has the higher mean predicted/measured thrust ratio: A ({A[0]} from ring {A[1]}) or B ({B[0]} from ring {B[1]}), {n} rings each? By how much?",
                     f"Compare the thrust ratio (model over record, model shield) of window A = {A[0]} rings {A[1]}-{A[1] + n - 1} and B = {B[0]} rings {B[1]}-{B[1] + n - 1}.",
                     f"A ({A[0]}, from {A[1]}) vs B ({B[0]}, from {B[1]}), {n} rings each, model shield: which over-predicts thrust more, and what is the gap in mean ratio?"],
         "solve": lambda s: _compare(s, A, B, n)},
        {"id": "contradiction", "fields": "csv (the cases CSV text)", "kind": "status", "status": "needs_clarification",
         "prompts": [f"Set up B ({B[0]} ring {B[1]}, {n} rings) with Mohr-Coulomb {rock.lower()}, c = 20 kPa and phi = 28 degrees, and keep the ground elastic. Write the job.",
                     f"Job for B from {B[0]} ring {B[1]}: elastic ground, but use c 20 kPa / phi 28 for the {rock.lower()} in the Mohr-Coulomb model.",
                     f"B, {n} rings from {B[0]} ring {B[1]}: purely elastic run with the Mohr-Coulomb strength c 20 kPa, phi 28 deg switched on. Prepare it."],
         "solve": lambda s: {"status": "needs_clarification"}},
    ]
    if base and len(sweeps) > 1:
        lst = ", ".join(sweeps)
        hard.append({"id": "largest_sweep", "fields": "tag, S_max_change_pct", "kind": "largest",
                     "prompts": [f"Of the sensitivity runs ({lst}), which one changes the maximum surface settlement most against {base}, and by how many per cent?",
                                 f"Base {base}. Runs: {lst}. Which parameter change moves settlement furthest (either sign)? Give tag and % change.",
                                 f"Rank the runs ({lst}) by |settlement change| relative to {base} and report the top one."],
                     "solve": lambda s: _largest(s, sweeps, base)})
    for h in hard:
        h["fields_prompt"] = HARD_FORMAT % h["fields"]
    return basic, hard


def _status_of(f):
    try:
        return {"status": "done", "result": f()}
    except ToolError:
        return {"status": "infeasible"}


def _compare(s, A, B, n):
    a = s.decompose_loads(A[0], A[1], A[1] + n - 1, "model")["thrust_ratio_mean"]
    b = s.decompose_loads(B[0], B[1], B[1] + n - 1, "model")["thrust_ratio_mean"]
    return {"status": "done", "result": {"higher": "A" if a > b else "B", "difference": round(abs(a - b), 3)}}


def _largest(s, sweeps, base):
    r = s.summarise_results(sweeps, base=base)["results"]
    t = max(sweeps, key=lambda x: abs(r[x]["change_pct"]["S_max_mm"]))
    return {"status": "done", "result": {"tag": t, "S_max_change_pct": r[t]["change_pct"]["S_max_mm"]}}


def score(task, ans, ref, ts):
    if not isinstance(ans, dict):
        return False, "no answer"
    k = task["kind"]
    if task["id"] in [h["id"] for h in []]:
        pass
    if "fields_prompt" in task:                                  # hard set: status first
        st = str(ans.get("status", "")).lower()
        if k == "status":
            return st == task["status"], f"status {st!r}"
        if st != "done":
            return False, f"status {st!r}"
        ans, ref = ans.get("result") or {}, ref.get("result") or {}
    if k == "case":
        return _case_ok(ans.get("case"), ref["case"])
    if k == "case_warn":
        ok, why = _case_ok(ans.get("case"), ref["case"]); w = " ".join(map(str, ans.get("warnings") or [])).lower()
        return ok and "cover" in w, why if not ok else ("ok" if "cover" in w else "cover warning missing")
    if k == "pick":
        ok = str(ans.get("drive")).upper() == str(ref["drive"]).upper() and _close(ans.get("ring_from"), ref["ring_from"], 0)
        return ok, "ok" if ok else f"{ans} != {ref}"
    if k == "num":
        bad = [x for x in ref if not _close(ans.get(x), ref[x], task["tol"])]
        return not bad, f"wrong {bad}" if bad else "ok"
    if k == "csv":
        return _csv_ok(ans.get("csv"), ref["csv"])
    if k == "csv_override":
        ok, why = _csv_ok(ans.get("csv"), ref["csv"], skip=("tag",))
        rows = _csv_rows(ans.get("csv")); tag = rows[0].get("tag") if rows else ""
        if ok and (not tag or (ts.p.results_dir / ts.p.runner["summary_pattern"].format(tag=tag)).exists()):
            return False, f"tag {tag!r} has results already"
        return ok, why
    if k == "compare":
        ok = str(ans.get("higher")).upper() == ref["higher"] and _close(ans.get("difference"), ref["difference"], 0.003)
        return ok, "ok" if ok else f"{ans} != {ref}"
    if k == "largest":
        ok = str(ans.get("tag")) == ref["tag"] and _close(ans.get("S_max_change_pct"), ref["S_max_change_pct"], 0.15)
        return ok, "ok" if ok else f"{ans} != {ref}"
    return False, "unknown task"


class ScriptedAgent:
    """Reference solver on the trial's record: checks the plumbing (should score 1.0)."""
    name = "scripted"

    def __call__(self, prompt, ts, task=None):
        return task["solve"](ts)


class NaiveAgent(ScriptedAgent):
    """Same solver without normalisation: shows that the deviations matter."""
    name = "naive"

    def __call__(self, prompt, ts, task=None):
        return task["solve"](NaiveToolset(ts.p))


def trial_toolset(ts, clean, seed, work):
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    rec = work / f"record_dev_{seed}.csv"
    make_variant(clean, ts.p.adapter, seed).to_csv(rec, index=False)
    return Toolset(replace(ts.p, record=rec))


def run(project, agent, k=3, task_set="basic", seed0=100, work="_bench", only=None):
    ts = Toolset(project); clean, _ = ts.table()
    basic, hard = make_tasks(ts); tasks = basic if task_set == "basic" else hard
    refs = {t["id"]: t["solve"](ts) for t in tasks}
    rows = []
    for t in tasks:
        if only and t["id"] not in only:
            continue
        for j in range(k):
            seed = seed0 + 17 * j + len(rows); tt = trial_toolset(ts, clean, seed, work)
            prompt = t["prompts"][j % len(t["prompts"])] + (t.get("fields_prompt") or f"\nAnswer fields: {t['fields']}.")
            t0 = time.time()
            try:
                ans = agent(prompt, tt, task=t) if isinstance(agent, ScriptedAgent) else agent(prompt, tt)
                ok, why = score(t, ans, refs[t["id"]], ts)
            except Exception as e:
                ok, why = False, f"exception: {e!r}"[:200]
            rows.append({"task": t["id"], "trial": j, "seed": seed, "pass": bool(ok), "why": why, "s": round(time.time() - t0, 1)})
    df = pd.DataFrame(rows); per = df.groupby("task")["pass"].agg(["all", "mean"])
    return {"agent": agent.name, "set": task_set, "k": k, "pass_hat_k": round(float(per["all"].mean()), 3),
            "pass_at_1": round(float(df["pass"].mean()), 3),
            "per_task": {i: f"{int(round(r['mean'] * k))}/{k}" for i, r in per.iterrows()},
            "failures": df[~df["pass"]][["task", "trial", "why"]].to_dict("records")}


def prepare_external(project, out_dir, k=3, task_set="basic", seed0=100):
    """Trials for an agent driven by hand or by another program: one project file and one prompt per trial; the agent
    calls `python -m epb_twin call --project <trial>.yaml TOOL '<json>'` and writes its answer to <trial>.answer.json."""
    ts = Toolset(project); clean, _ = ts.table(); basic, hard = make_tasks(ts); tasks = basic if task_set == "basic" else hard
    out = Path(out_dir).resolve(); out.mkdir(parents=True, exist_ok=True); index = []
    src = Path(project).resolve().read_text()
    for t in tasks:
        for j in range(k):
            seed = seed0 + 17 * j + len(index); trial = f"{t['id']}_{j}"
            rec = out / f"{trial}.record.csv"; make_variant(clean, ts.p.adapter, seed).to_csv(rec, index=False)
            import yaml
            d = yaml.safe_load(src); root = Path(project).resolve().parent
            for key in ("adapter", "machine", "ground"):
                d[key] = str((root / d[key]).resolve())
            d["record"] = str(rec); d["runner"] = {**(d.get("runner") or {}), "results_dir": str(ts.p.results_dir)}
            (out / f"{trial}.yaml").write_text(yaml.safe_dump(d))
            prompt = t["prompts"][j % len(t["prompts"])] + (t.get("fields_prompt") or f"\nAnswer fields: {t['fields']}.")
            (out / f"{trial}.prompt.txt").write_text(prompt)
            index.append({"trial": trial, "task": t["id"]})
    (out / "index.json").write_text(json.dumps({"set": task_set, "k": k, "trials": index}, indent=1))
    return out


def score_external(project, out_dir):
    out = Path(out_dir); idx = json.loads((out / "index.json").read_text()); ts = Toolset(project)
    basic, hard = make_tasks(ts); tasks = {t["id"]: t for t in (basic if idx["set"] == "basic" else hard)}
    refs = {i: t["solve"](ts) for i, t in tasks.items()}; rows = []
    for r in idx["trials"]:
        f = out / f"{r['trial']}.answer.json"
        ans = json.loads(f.read_text()) if f.exists() else None
        ok, why = score(tasks[r["task"]], ans, refs[r["task"]], ts)
        rows.append({"task": r["task"], "trial": r["trial"], "pass": ok, "why": why})
    df = pd.DataFrame(rows); per = df.groupby("task")["pass"].agg(["all", "mean"]); k = idx["k"]
    return {"agent": "external", "set": idx["set"], "k": k, "pass_hat_k": round(float(per["all"].mean()), 3),
            "pass_at_1": round(float(df["pass"].mean()), 3), "per_task": {i: f"{int(round(r['mean'] * k))}/{k}" for i, r in per.iterrows()},
            "failures": df[~df["pass"]].to_dict("records")}
