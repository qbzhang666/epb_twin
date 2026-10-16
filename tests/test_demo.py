"""End-to-end on the synthetic demo: tools, deviations, benchmark baselines and the MCP server."""
import json, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epb_twin.synthetic import write_demo
from epb_twin.tools import Toolset, ToolError
from epb_twin import bench as B


def test_demo_end_to_end():
    d = Path(tempfile.mkdtemp()) / "demo"; write_demo(d)
    ts = Toolset(d / "project.yaml")
    w = ts.list_windows(top=1)["windows"][0]
    c = ts.build_window(w["drive"], w["ring_from"], name="t")["case"]
    assert c["tag"] == "T_r30" and c["z_axis"] < -ts.r_exc
    job = ts.write_job([c], "t1"); assert job["n_cases"] == 1 and "python -u -m epb_twin.fe.advance run" in job["script"]
    assert ts.decompose_loads("D1")["rings"] >= 5
    try:
        ts.build_window("D1", 140, name="gap"); raise AssertionError("gap window accepted")
    except ToolError:
        pass
    try:
        ts.guard({"x": list(range(50))}); raise AssertionError("guard did not fire")
    except ToolError:
        pass
    r1 = B.run(d / "project.yaml", B.ScriptedAgent(), k=1, work=d / "_b")
    r0 = B.run(d / "project.yaml", B.NaiveAgent(), k=1, work=d / "_b")
    assert r1["pass_hat_k"] == 1.0 and r0["pass_hat_k"] < 0.5
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
    out = subprocess.run([sys.executable, "-m", "epb_twin", "mcp", "--project", str(d / "project.yaml")], input="\n".join(map(json.dumps, msgs)) + "\n",
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1]).stdout.splitlines()
    assert len(json.loads(out[1])["result"]["tools"]) == 6


if __name__ == "__main__":
    test_demo_end_to_end(); print("epb_twin demo tests passed")
