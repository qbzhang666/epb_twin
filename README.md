# epb_twin (working name)

[User guide](docs/USER_GUIDE.md) · [快速入门](docs/QUICKSTART.zh.md) · BSD 3-Clause

Tested tools for ring-by-ring analysis of earth-pressure-balance (EPB) shield drives: from a machine record to
finite element cases, load decomposition and result summaries, with an optional language-model agent that calls the
tools and never sees per-ring data.

A new project supplies three files and no code: an **adapter** (how its record maps onto the standard record),
a **machine** file and a **ground** file. Everything else is shared.

## Status
| Step | Content | State |
|---|---|---|
| 1 | Standard per-ring record (`schema.py`), adapters (`adapter.py`, `adapters/example.yaml`), normaliser | done, tested |
| 2 | Machine and ground files (`spec.py`, `specs/*.yaml`) read by the advance and load models | done, tested |
| 3 | Synthetic drive (`synthetic.py`), deviations, tools (`tools.py`), load model (`loads.py`), project file, agents (OpenAI-compatible, Anthropic, MCP, external), benchmark, CLI | done, tested |
| 4 | Advance model in the package (`fe/`), units in adapters, adapter examples (logger, Chinese headers), audit log, licence, citation, CI (Python 3.9-3.13, three OS, FE smoke test), containers, docs (English, Chinese quick start) | done, tested |
| 5 | Release v0.1.0 on GitHub with a Zenodo DOI (after the IP check) | next |

## Quick start
```
pip install -e .
python -m epb_twin demo demo_project
python -m epb_twin bench --project demo_project/project.yaml --agent scripted
```
Full guide: `docs/USER_GUIDE.md` (installation, project set-up, adapters, machine and ground files, tools, language
models, benchmark, data boundary, compatibility).

## Layer 1: record and adapter
```python
from epb_twin import load_adapter, normalise, validate
ad = load_adapter("my_project.yaml")          # kept with the project data
table, report = normalise(raw_dataframe, ad, r_exc=7.8)
validate(table)                               # required fields present, counts only
```
`report` holds counts only (cover ranges, text numbers, duplicate revisions, ground and mode strings, missing grout).
The standard fields, units and derived columns are listed in `epb_twin/schema.py`.

## Layer 2: machine and ground files
```python
from epb_twin import load_machine, load_ground
m = load_machine("machine.yaml")      # diameter, ring, lining, shield, back-up, cutterhead
g = load_ground("ground.yaml")        # materials (E, nu, gamma, K0, c, phi) and the default layering
```
Examples: `epb_twin/specs/machine_example_7m.yaml`, `epb_twin/specs/ground_example_two_layer.yaml`. The advance model
reads `ADV_MACHINE` and `ADV_GROUND` (paths); single values can still be overridden by environment variables for sweeps.
Needs PyYAML (`pip install pyyaml`).

## Tests
`python -m pytest tests` (the finite element smoke test runs when `[fe]` is installed, about 1 min)

## Publishing (maintainers)
1. Create the repository and move `ci/github-workflow-tests.yml` to `.github/workflows/tests.yml` (tests on Python
   3.9 to 3.13, Linux, Windows and macOS, and the finite element smoke test).
2. Connect the repository to Zenodo, then tag `v0.1.0`; add the DOI to CITATION.cff and README.
3. Optional: build the container images in `containers/` and publish them with the release.
