"""Smoke test of the finite element advance model: 9 rings of the demo machine on a coarse mesh (about 1 min).
Skipped when gmsh or Kratos is not installed (pip install -e ".[fe]")."""
import json, os, subprocess, sys, tempfile, shutil
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytest.importorskip("gmsh"); pytest.importorskip("KratosMultiphysics.GeoMechanicsApplication")


def test_advance_smoke():
    sys.path.insert(0, str(ROOT))
    from epb_twin.synthetic import write_demo
    from epb_twin.tools import Toolset
    d = Path(tempfile.mkdtemp()); demo = write_demo(d / "demo")
    run = d / "run"; run.mkdir(); shutil.copy(demo / "machine.yaml", run); shutil.copy(demo / "ground.yaml", run)
    env = {**os.environ, "ADV_Z_AXIS": "-14", "ADV_Z_ROCK": "-7", "ADV_TAG": "smoke", "ADV_SHIELD_W_KN": "5000",
           "ADV_BACKUP_W_KN": "3000", "OMP_NUM_THREADS": "2", "PYTHONPATH": str(ROOT)}
    for args in (["mesh", "--rings", "9", "--h", "2.5"], ["run", "--rings", "9", "--steps", "1"]):
        subprocess.run([sys.executable, "-m", "epb_twin.fe.advance", *args], cwd=run, env=env, check=True,
                       stdout=subprocess.DEVNULL, timeout=900)
    m = Toolset.metrics(run / "results" / "adv_smoke_summary.json", old_rings=3)
    assert 0 < m["S_max_mm"] < 50 and m["N_hoop_kN_m"] > 0 and m["shield_contact_frac_max"] >= 0
