"""Project file: everything a new project configures, in one place. Paths are relative to the project file.

    name: my_drive
    record: data/record.csv              # raw per-ring record (csv or xlsx), any column names
    adapter: adapter.yaml                # how the record maps onto the standard record
    machine: machine.yaml                # shield, lining, cutterhead
    ground: ground.yaml                  # materials, default layering, load-model ground class
    window: {n_rings: 45, grout_overpressure_bar: 0.5}
    runner:
      kind: slurm                        # slurm | local
      model_script: epb_twin.fe.advance  # advance model run once per case (a module, or a script path *.py)
      results_dir: results
      summary_pattern: "adv_{tag}_summary.json"
      slurm: {account: myproj, partition: comp, cpus: 8, mem: 48G, python_env: /path/to/venv/bin/activate}
    guard: {min_rings: 5, max_series: 20}  # data boundary of the tool outputs
    audit: true                            # log every tool call to _audit/tool_calls.jsonl (arguments, size, hash)
"""
from dataclasses import dataclass, field
from pathlib import Path
from .adapter import load_adapter
from .spec import load_machine, load_ground, _read
from .loads import LoadGround

RUNNER_DEFAULTS = {"kind": "local", "model_script": "epb_twin.fe.advance", "results_dir": "results",
                   "summary_pattern": "adv_{tag}_summary.json",
                   "slurm": {"account": "", "partition": "", "cpus": 8, "mem": "48G", "python_env": ""}}


@dataclass
class Project:
    name: str
    root: Path
    record: Path
    adapter: object
    machine: object
    ground: object
    load_ground: object
    window: dict = field(default_factory=dict)
    runner: dict = field(default_factory=dict)
    guard: dict = field(default_factory=dict)
    audit: bool = True

    @property
    def results_dir(self):
        return (self.root / self.runner["results_dir"]).resolve()


def load_project(path):
    p = Path(path).resolve(); d = _read(p); root = p.parent
    rel = lambda k: (root / d[k]).resolve()
    g_raw = _read(rel("ground"))
    if "load_model" not in g_raw:
        raise ValueError("ground file needs a load_model section (kind, gamma, K0, ucs_MPa or c_kPa)")
    runner = {**RUNNER_DEFAULTS, **(d.get("runner") or {})}
    runner["slurm"] = {**RUNNER_DEFAULTS["slurm"], **((d.get("runner") or {}).get("slurm") or {})}
    return Project(name=d.get("name", p.stem), root=root, record=rel("record"), adapter=load_adapter(rel("adapter")),
                   machine=load_machine(rel("machine")), ground=load_ground(rel("ground")),
                   load_ground=LoadGround.from_dict(g_raw["load_model"]),
                   window={"n_rings": 45, "grout_overpressure_bar": 0.5, **(d.get("window") or {})}, runner=runner,
                   guard={"min_rings": 5, "max_series": 20, **(d.get("guard") or {})}, audit=bool(d.get("audit", True)))
