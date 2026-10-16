"""Machine and ground files (layer 2). A project describes its shield and its ground in two small YAML files; the
advance model and the load model read them instead of constants.

machine.yaml
    name: ...
    excavated_diameter_m: 7.0
    ring_length_m: 1.5
    lining: {extrados_diameter_m: 6.7, thickness_m: 0.35, E_GPa: 35.0, joint_factor: 0.6}
    shield: {length_m: 9.0, declared_length_m: 8.5, overcut_m: 0.03, taper_m: 0.02, weight_kn: 5000}
    backup: {length_m: 60.0, weight_kn: 3000}
    cutterhead: {opening_ratio: 0.30, rim_length_m: 0.3, cutter_radius_m: 0.216, cutter_tip_width_m: 0.015,
                 cutter_spacing_m: 0.08, cutter_rated_load_kn: 250, bearing_torque_knm: 150,
                 seal_contact_length_m: 0.5}

ground.yaml
    name: ...
    materials:                      # E (MPa), nu, unit weight (kN/m3), K0, cohesion (kPa), friction angle (deg)
      Fill: {E_MPa: 15, nu: 0.35, gamma: 18.0, K0: 0.5, c_kPa: 5, phi_deg: 28}
      Rock: {E_MPa: 120, nu: 0.28, gamma: 22.6, K0: 0.7, c_kPa: 100, phi_deg: 32}
    default_layers: [Fill, Rock]    # two-layer model split at the rock head of each case
"""
from dataclasses import dataclass, field
from pathlib import Path


def _read(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        import json
        return json.loads(text)
    try:
        import yaml
    except ImportError as e:
        raise ImportError("machine and ground files need PyYAML: pip install pyyaml") from e
    return yaml.safe_load(text)


@dataclass(frozen=True)
class Machine:
    name: str
    excavated_diameter_m: float
    ring_length_m: float
    lining_extrados_diameter_m: float
    lining_thickness_m: float
    lining_E_GPa: float
    joint_factor: float
    shield_length_m: float
    overcut_m: float
    taper_m: float
    shield_weight_kn: float = 0.0
    backup_weight_kn: float = 0.0
    backup_length_m: float = 0.0
    declared_shield_length_m: float = None
    cutterhead: dict = field(default_factory=dict)

    @property
    def radius_m(self):
        return self.excavated_diameter_m / 2

    def load_geometry(self, shield_length_m=None):
        """Keyword arguments for the load model's machine geometry."""
        c = self.cutterhead
        kw = dict(diameter_m=self.excavated_diameter_m, segment_outer_diameter_m=self.lining_extrados_diameter_m,
                  shield_length_m=shield_length_m or self.shield_length_m, machine_weight_kn=self.shield_weight_kn,
                  backup_weight_kn=self.backup_weight_kn)
        for k_spec, k_geo in (("opening_ratio", "opening_ratio"), ("rim_length_m", "head_rim_length_m"),
                              ("cutter_radius_m", "cutter_radius_m"), ("cutter_tip_width_m", "cutter_tip_width_m"),
                              ("cutter_spacing_m", "cutter_spacing_m"), ("cutter_rated_load_kn", "cutter_rated_load_kn"),
                              ("bearing_torque_knm", "bearing_torque_knm"), ("seal_contact_length_m", "seal_contact_length_m")):
            if k_spec in c:
                kw[k_geo] = float(c[k_spec])
        return kw


@dataclass(frozen=True)
class Material:
    E_MPa: float
    nu: float
    gamma: float
    K0: float
    c_kPa: float = 0.0
    phi_deg: float = 30.0


@dataclass(frozen=True)
class Ground:
    name: str
    materials: dict
    default_layers: list


def load_machine(path):
    d = _read(path); ln, sh, bu = d.get("lining", {}), d.get("shield", {}), d.get("backup", {})
    return Machine(name=d.get("name", Path(path).stem), excavated_diameter_m=float(d["excavated_diameter_m"]),
                   ring_length_m=float(d["ring_length_m"]), lining_extrados_diameter_m=float(ln["extrados_diameter_m"]),
                   lining_thickness_m=float(ln["thickness_m"]), lining_E_GPa=float(ln.get("E_GPa", 35.0)),
                   joint_factor=float(ln.get("joint_factor", 1.0)), shield_length_m=float(sh["length_m"]),
                   overcut_m=float(sh.get("overcut_m", 0.0)), taper_m=float(sh.get("taper_m", 0.0)),
                   shield_weight_kn=float(sh.get("weight_kn", 0.0)), backup_weight_kn=float(bu.get("weight_kn", 0.0)),
                   backup_length_m=float(bu.get("length_m", 0.0)),
                   declared_shield_length_m=float(sh["declared_length_m"]) if "declared_length_m" in sh else None,
                   cutterhead=dict(d.get("cutterhead", {}) or {}))


def load_ground(path):
    d = _read(path)
    mats = {n: Material(**{k: float(v) for k, v in m.items()}) for n, m in d["materials"].items()}
    lay = list(d.get("default_layers", list(mats)[:2]))
    for n in lay:
        if n not in mats:
            raise ValueError(f"default layer {n} has no material")
    return Ground(name=d.get("name", Path(path).stem), materials=mats, default_layers=lay)
