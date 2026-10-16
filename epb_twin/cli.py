"""Command line: python -m epb_twin <command> ...

  demo DIR                               write the synthetic demonstration project
  check --project P                      read, normalise and validate the record; print the report
  tools                                  list the tools and their arguments (JSON schema)
  call --project P TOOL 'JSON'           run one tool, print its output
  bench --project P [--set basic|hard] [--agent scripted|naive|openai|anthropic] [--model M] [--base-url U] [--k 3]
  bench --project P --prepare-external DIR [--set ...] [--k 3]   /   --score-external DIR
  mcp --project P                        serve the tools to an MCP client over stdio
"""
import argparse, json, sys
from pathlib import Path


def _print(x):
    print(json.dumps(x, indent=1, default=str))


def mcp_serve(project):
    """Minimal Model Context Protocol server (JSON-RPC 2.0 over stdio): initialize, tools/list, tools/call."""
    from .tools import Toolset
    from .agent import tool_schemas
    ts = Toolset(project)
    tools = [{"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]}
             for t in tool_schemas("anthropic") if t["name"] != "submit_answer"]
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line); mid = msg.get("id"); method = msg.get("method"); res = None; err = None
        if method == "initialize":
            res = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                   "capabilities": {"tools": {}}, "serverInfo": {"name": "epb_twin", "version": __import__("epb_twin").__version__}}
        elif method == "tools/list":
            res = {"tools": tools}
        elif method == "tools/call":
            p = msg.get("params", {})
            try:
                out = ts.TOOLS[p["name"]](**(p.get("arguments") or {}))
                res = {"content": [{"type": "text", "text": json.dumps(out, default=str)}]}
            except Exception as e:
                res = {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}
        elif mid is None:
            continue                                            # notifications
        else:
            err = {"code": -32601, "message": f"unknown method {method}"}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, **({"error": err} if err else {"result": res})}) + "\n")
        sys.stdout.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m epb_twin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("dir"); d.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("check"); c.add_argument("--project", required=True)
    sub.add_parser("tools")
    k = sub.add_parser("call"); k.add_argument("--project", required=True); k.add_argument("tool"); k.add_argument("args", nargs="?", default="{}")
    b = sub.add_parser("bench"); b.add_argument("--project", required=True); b.add_argument("--set", default="basic", choices=["basic", "hard"])
    b.add_argument("--agent", default="scripted", choices=["scripted", "naive", "openai", "anthropic"])
    b.add_argument("--model"); b.add_argument("--base-url"); b.add_argument("--k", type=int, default=3)
    b.add_argument("--prepare-external"); b.add_argument("--score-external"); b.add_argument("--out")
    m = sub.add_parser("mcp"); m.add_argument("--project", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        from .synthetic import write_demo
        out = write_demo(a.dir, a.seed); print(f"demo project written to {out}; try: python -m epb_twin check --project {out}/project.yaml")
    elif a.cmd == "check":
        from .tools import Toolset
        from .schema import validate
        ts = Toolset(a.project); t, rep = ts.table()
        _print({"project": ts.p.name, "deviations": rep, "validate": validate(t),
                "drives": ts.normalise_record()["drives"], "machine": ts.p.machine.name, "ground": ts.p.ground.name})
    elif a.cmd == "tools":
        from .agent import tool_schemas
        _print(tool_schemas("anthropic"))
    elif a.cmd == "call":
        from .tools import Toolset
        _print(Toolset(a.project).TOOLS[a.tool](**json.loads(a.args)))
    elif a.cmd == "bench":
        from . import bench as B
        if a.prepare_external:
            print(B.prepare_external(a.project, a.prepare_external, a.k, a.set)); return
        if a.score_external:
            res = B.score_external(a.project, a.score_external)
        else:
            if a.agent == "scripted":
                ag = B.ScriptedAgent()
            elif a.agent == "naive":
                ag = B.NaiveAgent()
            elif a.agent == "openai":
                from .agent import OpenAIAgent
                ag = OpenAIAgent(a.model, a.base_url)
            else:
                from .agent import AnthropicAgent
                ag = AnthropicAgent(a.model)
            res = B.run(a.project, ag, a.k, a.set, work=Path(a.project).resolve().parent / "_bench")
        _print(res)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    elif a.cmd == "mcp":
        mcp_serve(a.project)


if __name__ == "__main__":
    main()
