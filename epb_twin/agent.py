"""Language-model agents (layer 7, optional). The model sees the task and the tool outputs only; every tool runs on
the analyst's computer. Two back-ends:
  OpenAIAgent     any OpenAI-compatible chat endpoint with tool calls: Ollama (http://localhost:11434/v1),
                  vLLM, LM Studio, llama.cpp server, or a hosted service
  AnthropicAgent  Claude through the Anthropic API (pip install anthropic; ANTHROPIC_API_KEY)
"""
import json, os, urllib.request

PROPS = {
    "normalise_record": ("Read and clean the drive record; returns deviation counts and the drives with their ring ranges.", {}, []),
    "list_windows": ("Contiguous windows of n rings with ground context, ranked by a window statistic.",
                     {"n_rings": {"type": "integer"}, "sort_by": {"type": "string", "description": "e.g. soil_thick_mean, cover_mean, rock_cover_mean, water_bar_mean, face_crown_bar_mean"},
                      "descending": {"type": "boolean"}, "top": {"type": "integer"}, "drive": {"type": "string"},
                      "max_bad_cover_rings": {"type": "integer", "description": "allowed rings with doubtful cover (default 0)"}}, []),
    "build_window": ("Case line of the advance model for one window, with window statistics and data-quality flags.",
                     {"drive": {"type": "string"}, "ring_from": {"type": "integer"}, "n_rings": {"type": "integer"},
                      "name": {"type": "string"}, "grout_overpressure_bar": {"type": "number"}}, ["drive", "ring_from"]),
    "decompose_loads": ("Machine-load model against the logged thrust, contact force and torque over a drive or a window.",
                        {"drive": {"type": "string"}, "ring_from": {"type": "integer"}, "ring_to": {"type": "integer"},
                         "shield": {"type": "string", "enum": ["model", "declared"]}}, ["drive"]),
    "write_job": ("Cases file and run script for a list of case lines. Case keys: those of build_window plus optional "
                  "shield_w_kn, backup_w_kn, mc (0 elastic, 1 Mohr-Coulomb), layers, e_<material>_pa, c_<material>_kpa, phi_<material>_deg.",
                  {"cases": {"type": "array", "items": {"type": "object"}}, "name": {"type": "string"}, "comment": {"type": "string"}},
                  ["cases", "name"]),
    "summarise_results": ("Metrics of finished runs (settlement, shield convergence, lining forces); with base, the change in per cent.",
                          {"tags": {"type": "array", "items": {"type": "string"}}, "base": {"type": "string"}}, ["tags"]),
    "submit_answer": ("Submit the final answer as a JSON object with the requested fields.", {"answer": {"type": "object"}}, ["answer"]),
}


def tool_schemas(style="anthropic"):
    out = []
    for name, (desc, props, req) in PROPS.items():
        params = {"type": "object", "properties": props, "required": req}
        out.append({"name": name, "description": desc, "input_schema": params} if style == "anthropic" else
                   {"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
    return out


def system_prompt(ts):
    t, _ = ts.table()
    drives = ", ".join(sorted(map(str, t.drive.unique())))
    return (f"You prepare finite element cases and summaries for an EPB tunnel drive (project {ts.p.name}). Drives: {drives}. "
            f"Default window length: {ts.p.window['n_rings']} rings. Use only the tools and do not invent numbers. "
            "If a request cannot be done or contradicts itself, say so in the answer instead of guessing. "
            "Finish by calling submit_answer.")


def _run_tool(ts, name, args):
    out = ts.TOOLS[name](**args)
    return {k: v for k, v in out.items() if k != "script"} if name == "write_job" else out


class OpenAIAgent:
    def __init__(self, model, base_url=None, api_key=None, max_turns=12, temperature=0.0):
        self.name = f"openai:{model}"; self.model = model; self.max_turns = max_turns; self.temperature = temperature
        self.url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")).rstrip("/") + "/chat/completions"
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")

    def _post(self, payload):
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())

    def __call__(self, prompt, ts):
        msgs = [{"role": "system", "content": system_prompt(ts)}, {"role": "user", "content": prompt}]
        for _ in range(self.max_turns):
            m = self._post({"model": self.model, "messages": msgs, "tools": tool_schemas("openai"),
                            "temperature": self.temperature})["choices"][0]["message"]
            msgs.append({k: v for k, v in m.items() if k in ("role", "content", "tool_calls")})
            calls = m.get("tool_calls") or []
            if not calls:
                try:
                    return json.loads(m.get("content") or "")
                except ValueError:
                    return None
            for c in calls:
                fn = c["function"]; args = fn.get("arguments") or "{}"
                args = json.loads(args) if isinstance(args, str) else args
                if fn["name"] == "submit_answer":
                    return args.get("answer", args)
                try:
                    res = json.dumps(_run_tool(ts, fn["name"], args))[:20000]
                except Exception as e:
                    res = f"error: {e}"
                msgs.append({"role": "tool", "tool_call_id": c.get("id", fn["name"]), "content": res})
        return None


class AnthropicAgent:
    def __init__(self, model=None, max_turns=12):
        import anthropic
        self.client = anthropic.Anthropic(); self.model = model or os.environ["AGENT_MODEL"]
        self.name = f"anthropic:{self.model}"; self.max_turns = max_turns

    def __call__(self, prompt, ts):
        msgs = [{"role": "user", "content": prompt}]
        for _ in range(self.max_turns):
            r = self.client.messages.create(model=self.model, max_tokens=2000, system=system_prompt(ts),
                                            tools=tool_schemas("anthropic"), messages=msgs)
            msgs.append({"role": "assistant", "content": r.content})
            uses = [b for b in r.content if b.type == "tool_use"]
            if not uses:
                return None
            results = []
            for u in uses:
                if u.name == "submit_answer":
                    return u.input.get("answer")
                try:
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(_run_tool(ts, u.name, u.input))[:20000]})
                except Exception as e:
                    results.append({"type": "tool_result", "tool_use_id": u.id, "content": f"error: {e}", "is_error": True})
            msgs.append({"role": "user", "content": results})
        return None
