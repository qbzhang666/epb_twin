"""Layer-2 tests: machine and ground files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epb_twin import load_machine, load_ground

S = Path(__file__).resolve().parents[1] / "epb_twin" / "specs"


def test_machine():
    m = load_machine(S / "machine_example_7m.yaml")
    assert m.radius_m == 3.5 and m.ring_length_m == 1.5 and m.declared_shield_length_m is None
    g = m.load_geometry(shield_length_m=8.0)
    assert g["shield_length_m"] == 8.0 and g["cutter_spacing_m"] == 0.08 and g["head_rim_length_m"] == 0.3


def test_ground():
    g = load_ground(S / "ground_example_two_layer.yaml")
    assert g.default_layers == ["Soil", "Rock"] and g.materials["Rock"].E_MPa == 150 and g.materials["Soil"].phi_deg == 30


if __name__ == "__main__":
    test_machine(); test_ground(); print("epb_twin layer-2 tests passed")
