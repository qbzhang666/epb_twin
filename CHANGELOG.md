# Changelog
Versions follow semantic versioning. The standard record has its own version (`epb_twin.schema.SCHEMA_VERSION`);
a record or adapter written for schema 1.x keeps working within 1.x.

## 0.1.0 (unreleased)
- Standard per-ring record (schema 1.0), project adapters with unit conversion, normaliser for deviations of form.
- Machine and ground files; finite element advance model (`epb_twin.fe.advance`, Kratos GeoMechanics 10.4, gmsh 4).
- Machine-load model (JSCE thrust terms, CSM disc forces, cutterhead torque terms).
- Six tools with a data guard and an audit log; project file; Slurm and local runners.
- Synthetic demonstration drive; seeded deviations; basic and hard task sets; pass^k benchmark.
- Agents: OpenAI-compatible endpoints (Ollama, vLLM, LM Studio, llama.cpp, hosted), Anthropic API, MCP server, external.
- Adapter examples: generic, logger export in SI sub-units, Chinese headers.
- pandas is needed only by the normaliser: the specs and the FE model import without it (e.g. an HPC Kratos environment).
