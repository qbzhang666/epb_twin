# epb_twin

Tested tools that turn the per-ring record of an earth-pressure-balance shield drive into finite element cases, a
machine-load decomposition and result summaries, with an optional language-model agent that never sees per-ring
data. A new project writes four small text files and no code.

- [User guide](USER_GUIDE.md): installation, project set-up, adapters, machine and ground files, tools, language
  models, benchmark, data boundary, compatibility.
- [快速入门（中文）](QUICKSTART.zh.md)

```
pip install -e .
python -m epb_twin demo demo_project
python -m epb_twin bench --project demo_project/project.yaml --agent scripted
```
