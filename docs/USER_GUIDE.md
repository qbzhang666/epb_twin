# epb_twin user guide

epb_twin turns the per-ring record of an earth-pressure-balance (EPB) shield drive into finite element cases, a
machine-load decomposition and result summaries, through six tested tools. A language-model agent can call the tools,
but it never sees per-ring data: every tool returns window-level aggregates only.

A new project writes **four small text files and no code**: a project file, an adapter (how the record maps onto the
standard record), a machine file and a ground file.

## 1. Install
Python 3.9 or later, on Windows, macOS or Linux.
```
pip install -e .                 # from a clone of the repository: numpy, pandas, pyyaml
pip install -e ".[excel]"        # records in .xlsx
pip install -e ".[fe]" -c constraints-fe.txt   # advance model: Kratos 10.4 and gmsh 4, tested versions
```
Or use the container, which pins the whole stack: `docker build -t epb-twin -f containers/Dockerfile .`, or on a
cluster `apptainer build epb_twin.sif containers/epb_twin.def` (see the comments in each file).
The record tools, the load model, the agent and the benchmark need only the first line. The finite element model is
needed only where the cases are run (a workstation or a cluster).

## 2. Try the demonstration project (5 minutes)
```
python -m epb_twin demo demo_project                      # synthetic drive, no project data needed
python -m epb_twin check --project demo_project/project.yaml
python -m epb_twin call --project demo_project/project.yaml list_windows '{"top": 3}'
python -m epb_twin call --project demo_project/project.yaml build_window '{"drive": "D1", "ring_from": 60, "name": "A"}'
python -m epb_twin bench --project demo_project/project.yaml --agent scripted --k 3   # 1.00: the plumbing works
python -m epb_twin bench --project demo_project/project.yaml --agent naive --k 3      # low: the deviations matter
```
The demonstration drive, its ground and its numbers are fictitious and generated from a seed.

## 3. Set up your own project
```
my_project/
  project.yaml        # what is where, how cases are run, the data boundary
  adapter.yaml        # your record's column names and vocabularies
  machine.yaml        # your shield
  ground.yaml         # your ground
  data/record.csv     # your record (stays on your computer)
```
Start from the demo folder and replace the files. Keep `adapter.yaml` and `data/` out of any public repository if
your record or its vocabulary is confidential.

### 3.1 project.yaml
```yaml
name: my_drive
record: data/record.csv
adapter: adapter.yaml
machine: machine.yaml
ground: ground.yaml
window: {n_rings: 45, grout_overpressure_bar: 0.5}   # default window length; grout = face + overpressure
runner:
  kind: slurm                     # slurm (array job) or local (one case after another)
  model_script: epb_twin.fe.advance   # the packaged advance model (or a path to your own script, *.py)
  results_dir: results
  summary_pattern: "adv_{tag}_summary.json"
  slurm: {account: myproj, partition: comp, cpus: 8, mem: 48G, python_env: /path/to/venv/bin/activate}
guard: {min_rings: 5, max_series: 20}
audit: true                       # every tool call is logged to _audit/tool_calls.jsonl
```
`guard` is the data boundary: no tool output carries a numeric list longer than `max_series`, and no aggregate is
formed over fewer than `min_rings` rings. Lower values are stricter.

### 3.2 adapter.yaml: your record onto the standard record
The standard record has one row per built ring. Its fields, units and meaning are listed in `epb_twin/schema.py`
(`python -c "import epb_twin; [print(f) for f in epb_twin.FIELDS]"`). Required: `drive`, `ring_no`, `cover_m`,
`face_crown_bar`, `face_axis_bar`, `thrust_kn`, `cutterhead_torque_mnm`, `advance_speed_mm_min`,
`cutterhead_speed_rpm`. Everything else is optional and switches off the checks that need it.
```yaml
name: my_drive
columns:                          # standard name: your column name (only where they differ)
  face_crown_bar: EPB_sensor_1_12_mean
  face_axis_bar: EPB_sensor_4_9_mean
  thrust_kn: Thrust_total
  ground_class: Geology_at_face
  operating_mode: Mode
ground_classes:
  order: [S1, S2, S3, SF, QA]     # your class codes, in face order
  aliases: {FILL: F}              # optional: words in the text that mean a code
modes:
  default: full_EPB
  rules:                          # first match wins, case-insensitive substring
    - {mode: EPB_air_bubble, contains: [bubble, air]}
    - {mode: semi_EPB, contains: [semi, partial, half]}
limits: {max_cover_spread_m: 1.5} # rings whose cover estimate spreads more than this are flagged
units:                            # only where your record differs from the schema
  face_crown_bar: kPa
  thrust_kn: MN
  cutterhead_torque_mnm: kN m
```
Units are converted to the schema's (bar, kN, MN m, mm/min, rpm, m, m3, %): pressure in bar, kPa, MPa, Pa or psi;
force in kN, MN, N or tf; torque in MN m, kN m, N m or tf m; length in m, mm, cm or ft; speed in mm/min, m/h, mm/s
or m/min; volume in m3 or L; rotation in rpm or rad/s. Ready-made examples in `epb_twin/adapters/`:
`example.yaml` (generic), `logger_si.yaml` (a logger export with pressures in kPa, thrust in MN, torque in kN m,
advance in mm/s) and `chinese_headers.yaml` (Chinese column headers, geology in words, Chinese mode names). A test
writes the same drive in all three styles and checks that they normalise to the same table. What the normaliser
absorbs without help: numbers with units or decimal commas ("2,54 bar"), ranges ("17.2-21.8" read as the mean and its
spread), class codes written several ways ("S1, S-1", "s-f"), mode names in several spellings, missing grout, and
several revisions of the same ring (`source_rev`: the highest is kept).

Check the mapping: `python -m epb_twin check --project project.yaml` prints the counts of every deviation it handled,
missing required fields and unknown columns. It prints no ring values.

### 3.3 machine.yaml
```yaml
name: my_shield
excavated_diameter_m: 7.0
ring_length_m: 1.5
lining: {extrados_diameter_m: 6.7, thickness_m: 0.35, E_GPa: 35.0, joint_factor: 0.6}
shield: {length_m: 9.0, declared_length_m: 8.5, overcut_m: 0.03, taper_m: 0.02, weight_kn: 5000}
backup: {length_m: 60.0, weight_kn: 3000}
cutterhead: {opening_ratio: 0.30, rim_length_m: 0.3, cutter_radius_m: 0.216, cutter_tip_width_m: 0.015,
             cutter_spacing_m: 0.08, cutter_rated_load_kn: 250, bearing_torque_knm: 150, seal_contact_length_m: 0.5}
```
`declared_length_m` is optional: it lets `decompose_loads(shield="declared")` compare a declared and a modelled shield.

### 3.4 ground.yaml
```yaml
name: my_ground
materials:                        # E (MPa), nu, unit weight (kN/m3), K0, cohesion (kPa), friction angle (deg)
  Fill: {E_MPa: 15, nu: 0.35, gamma: 18.0, K0: 0.5, c_kPa: 5, phi_deg: 28}
  Rock: {E_MPa: 120, nu: 0.28, gamma: 22.6, K0: 0.7, c_kPa: 100, phi_deg: 32}
default_layers: [Fill, Rock]      # two-layer model split at each window's rock head
load_model: {kind: rock, gamma: 22.6, K0: 0.43, ucs_MPa: 15}   # or {kind: soil, gamma, K0, c_kPa, water_separate}
```
`load_model` is the ground class of the machine-load model: `rock` lets the disc cutters carry the face (CSM model,
needs `ucs_MPa`); `soil` is a pressurised face with skin friction (`water_separate: true`, sand) or adhesion
(`false`, clay).

## 4. Use the tools
| Tool | Returns |
|---|---|
| `normalise_record` | deviation counts; drives with ring ranges |
| `list_windows` | contiguous windows of n rings with ground context, ranked by a statistic (`soil_thick_mean`, `cover_mean`, ...) |
| `build_window` | the case line of a window, its statistics and data-quality flags |
| `decompose_loads` | load-model terms against the logged thrust, contact force and torque |
| `write_job` | cases CSV and run script (Slurm array or local loop), mesh commands, warnings |
| `summarise_results` | settlement, shield convergence and lining forces of finished runs; changes against a base |

From Python:
```python
from epb_twin.tools import Toolset
ts = Toolset("project.yaml")
w = ts.list_windows(sort_by="cover_mean", descending=False, top=5)
case = ts.build_window("D1", 120, name="W1")["case"]
job = ts.write_job([case], "first", out_dir="runs")
```
From the shell: `python -m epb_twin call --project project.yaml TOOL '{json arguments}'`.

### Running the cases
`write_job` writes `cases_<name>.csv` and `run_<name>.sh`. Copy them with `machine.yaml` and `ground.yaml` to a run
directory on a machine where epb_twin is installed with `[fe]` (or use the container); run the printed
`mesh_commands` once per window (`python -m epb_twin.fe.advance mesh ...`), then submit
(`sbatch --array=1-N run_<name>.sh`, or `bash run_<name>.sh` for `kind: local`). A case whose summary already exists
is skipped, so finished cases are never re-run; the tool warns when a tag would be skipped.

## 5. Use a language model (optional)
The model chooses tools and arguments; the tools run on your computer and return aggregates only.

| Back-end | How |
|---|---|
| Local open-weight model (Ollama) | `ollama pull qwen3.6:27b`, then `python -m epb_twin bench --project P --agent openai --model qwen3.6:27b` |
| Any OpenAI-compatible server (vLLM, LM Studio, llama.cpp, hosted) | `--agent openai --model NAME --base-url http://host:port/v1`; key in `OPENAI_API_KEY` |
| Claude through the API | `pip install anthropic`, `ANTHROPIC_API_KEY`, `--agent anthropic --model MODEL` |
| MCP clients (Claude Desktop, IDE assistants, other MCP hosts) | add a server with command `python -m epb_twin mcp --project /path/project.yaml` |
| Anything else, or by hand | `--prepare-external DIR`: one prompt and one project file per trial; the agent calls `python -m epb_twin call` and writes `<trial>.answer.json`; then `--score-external DIR` |

Fix the model version (an Ollama digest, a dated model id) and report it with the scores.

## 6. Benchmark
`bench` builds two task sets from the project itself: **basic** (case lines, window search, load split, job writing,
a sweep effect) and **hard** (data-quality warnings, an infeasible window, an override, a comparison, a contradictory
request, a sweep ranking). Each trial runs on its own copy of the record with seeded deviations of form (ground
spellings, mode names, cover ranges, units and decimal commas, missing grout, overlapping revisions); answers are
scored against the clean record. pass^k is the share of tasks solved in all k trials; use k = 10 for reported numbers.
On your own record the benchmark runs privately: nothing but the agent's tool calls leaves the tools.

## 7. Data boundary in short
- The record, the adapter and the machine and ground files stay where you keep them.
- Tool outputs are window aggregates, counts and case lines; the guard rejects longer numeric series.
- Every tool call made through an agent, the command line or MCP is logged in `_audit/tool_calls.jsonl` (time, tool,
  arguments, success, output size and hash), so a reviewer or data owner can check what left the tools.
- A cloud model sees the task text and tool outputs only. For a record that must not leave the computer at all, use a
  local model (Section 5).

## 8. Troubleshooting
| Message | Meaning |
|---|---|
| `x of n rings have context (window must be contiguous)` | the window crosses a gap in the record or rings without cover |
| `rings with cover spread > ... m` (flag) | cover estimates disagree on some rings; check before running |
| `unknown case keys` | a case key the advance model does not take; `write_job` lists the known keys |
| `machine and ground files need PyYAML` | `pip install pyyaml` in the environment that runs the model |
| `numeric series longer than ...` | a tool output would carry per-ring values; ask for an aggregate instead |

## 9. Compatibility
Tested for release 0.1.0:

| Item | Versions |
|---|---|
| Python | 3.9, 3.11, 3.12, 3.13 (CI: 3.9 to 3.13 on Linux, Windows and macOS) |
| numpy / pandas | numpy 2.0 to 2.5; pandas 2.3 and 3.0 |
| Finite element stack | Kratos Multiphysics 10.4.3 and 10.4.4 (GeoMechanics, StructuralMechanics, LinearSolvers), gmsh 4.15 |
| Records | CSV (UTF-8, any column names), XLSX with `[excel]` |
| Runners | Slurm array jobs; local runs on Windows (WSL or Git Bash), macOS, Linux; Apptainer and Docker images |
| Language models | OpenAI-compatible chat with tool calls (Ollama, vLLM, LM Studio, llama.cpp, hosted), Anthropic API, MCP hosts |

The standard record carries a version (`epb_twin.schema.SCHEMA_VERSION`, now 1.0). Adapters and records written for
1.x keep working within 1.x; changes are listed in CHANGELOG.md.

## 10. Licence and citation
BSD 3-Clause (LICENSE). Third-party licences, including gmsh (GPL, used only when meshing), are listed in
THIRD_PARTY.md. Cite the software and the paper as given in CITATION.cff.
