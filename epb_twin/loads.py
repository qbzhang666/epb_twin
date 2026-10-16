"""Machine-load model (layer 5): thrust and torque terms at one ring, for replaying a record.

Thrust after the Japanese shield-tunnel specification (face, cutting, shield skin, tail seal, back-up), disc forces
after the Colorado School of Mines model (Rostami and Ozdemir 1993; Rostami 1997) and cutterhead torque as a sum of
cutting, face friction, stirring, periphery and bearing terms (after Shi et al. 2011). Each coefficient has a low,
middle and high value (ResponseParameters). The chamber pressure comes from the record.

The ground at the face is one load class from the ground file:
    load_model: {kind: rock, gamma: 22.6, K0: 0.43, ucs_MPa: 15}                    # cutters carry the face
    load_model: {kind: soil, gamma: 18.9, K0: 0.40, c_kPa: 5, water_separate: true}  # pressurised face
"""
import math
from dataclasses import dataclass, fields

LEVELS = ("low", "mid", "high")
GAMMA_W = 9.81
_CSM_C, _PSI_PER_MPA, _INCH_PER_M = 2.12, 145.0377, 39.3701


@dataclass(frozen=True)
class Range:
    low: float
    mid: float
    high: float

    def at(self, level):
        return getattr(self, level)


@dataclass(frozen=True)
class MachineGeometry:
    diameter_m: float = 7.0
    shield_length_m: float = 9.0
    machine_weight_kn: float = 5000.0
    backup_weight_kn: float = 3000.0
    segment_outer_diameter_m: float = 6.7
    seal_contact_length_m: float = 0.5
    opening_ratio: float = 0.30
    head_rim_length_m: float = 0.3
    cutter_radius_m: float = 0.216
    cutter_tip_width_m: float = 0.015
    cutter_spacing_m: float = 0.08
    cutter_rated_load_kn: float = 250.0
    bearing_torque_knm: float = 150.0

    @property
    def radius_m(self):
        return self.diameter_m / 2.0

    @property
    def face_area_m2(self):
        return math.pi * self.diameter_m ** 2 / 4.0

    @property
    def n_face_cutters(self):
        return max(int(round(self.radius_m / self.cutter_spacing_m)), 1)

    @property
    def cutter_radius_sum_m(self):
        n = self.n_face_cutters
        return self.cutter_spacing_m * n * (n + 1) / 2.0


@dataclass(frozen=True)
class ResponseParameters:
    mu_skin: Range = Range(0.2, 0.3, 0.4)                 # JSCE; skin friction in sandy ground
    adhesion_factor: Range = Range(0.5, 1.0, 1.5)         # adhesion in clayey ground, multiple of cohesion
    mu_seal: Range = Range(0.2, 0.25, 0.3)                # JSCE
    mu_backup: Range = Range(0.08, 0.1, 0.15)
    mu_rock_skin: Range = Range(0.2, 0.3, 0.4)            # shield sliding on a rock invert
    mu_rock_shield: Range = Range(0.02, 0.10, 0.20)       # ground bearing on the skin of a closed rock drive
    face_friction: Range = Range(0.10, 0.15, 0.25)
    periphery_friction: Range = Range(0.10, 0.15, 0.25)
    muck_shear_kpa: Range = Range(5.0, 10.0, 20.0)
    cutting_energy_kj_m3: Range = Range(100.0, 300.0, 600.0)   # Teale (1965)
    tensile_ratio: Range = Range(0.08, 0.10, 0.12)

    def at(self, level):
        return {f.name: getattr(self, f.name).at(level) for f in fields(self)}


@dataclass(frozen=True)
class LoadGround:
    kind: str                     # "rock" or "soil"
    gamma: float                  # kN/m3
    K0: float
    ucs_MPa: float = 0.0
    c_kPa: float = 0.0
    water_separate: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(kind=str(d["kind"]), gamma=float(d["gamma"]), K0=float(d["K0"]), ucs_MPa=float(d.get("ucs_MPa", 0.0)),
                   c_kPa=float(d.get("c_kPa", 0.0)), water_separate=bool(d.get("water_separate", True)))


def mean_shield_pressure_kpa(z0_m, diameter_m, gamma, gw_depth_m, k_lat):
    """JSCE P_m = (P1 + P2 + Q1 + Q2) / 4 at crown and invert."""
    r = diameter_m / 2.0; out = []
    for depth in (z0_m - r, z0_m + r):
        total = gamma * depth; u = GAMMA_W * max(depth - gw_depth_m, 0.0); eff = max(total - u, 0.0)
        out.append((total, k_lat * eff + u))
    (p1, q1), (p2, q2) = out
    return (p1 + p2 + q1 + q2) / 4.0


def csm_cutter_forces_kn(ucs_mpa, penetration_mm, cutter_radius_m=0.216, tip_width_m=0.015, spacing_m=0.08, tensile_ratio=0.10):
    """(F_normal, F_rolling, contact angle) of one disc (CSM model, constant cross-section)."""
    if ucs_mpa <= 0 or penetration_mm <= 0:
        return 0.0, 0.0, 0.0
    radius_in = cutter_radius_m * _INCH_PER_M; tip_in = tip_width_m * _INCH_PER_M; spacing_in = spacing_m * _INCH_PER_M
    p_in = min(penetration_mm / 1000.0, 0.9 * cutter_radius_m) * _INCH_PER_M
    phi = math.acos((radius_in - p_in) / radius_in)
    sc = ucs_mpa * _PSI_PER_MPA; st = tensile_ratio * sc
    pp = _CSM_C * (spacing_in * sc ** 2 * st / (phi * math.sqrt(radius_in * tip_in))) ** (1.0 / 3.0)
    f_kn = tip_in * radius_in * phi * pp * 4.4482216 / 1000.0
    return f_kn * math.cos(phi / 2.0), f_kn * math.sin(phi / 2.0), phi


def ring_terms(z0_m, ground, g, advance_mm_min, rpm, chamber_kpa, params, gw_depth_m=5.0, closed_face=True):
    """All thrust (kN) and torque (kN m) terms at one ring. ground: LoadGround; g: MachineGeometry."""
    pen_mm = advance_mm_min / rpm; p_rev_m = pen_mm / 1000.0
    k_face = (1.0 - g.opening_ratio) * (2.0 * math.pi / 3.0) * g.radius_m ** 3
    skin_area = math.pi * g.diameter_m * g.shield_length_m
    seal_area = math.pi * g.segment_outer_diameter_m * g.seal_contact_length_m
    p_ch = float(chamber_kpa)
    if ground.kind == "soil":
        pm = mean_shield_pressure_kpa(z0_m, g.diameter_m, ground.gamma, gw_depth_m, ground.K0)
        skin = (params["mu_skin"] * (skin_area * pm + g.machine_weight_kn) if ground.water_separate
                else params["adhesion_factor"] * ground.c_kPa * skin_area)
        return {"thrust_skin_kn": skin, "thrust_shield_kn": 0.0, "thrust_face_kn": p_ch * g.face_area_m2,
                "thrust_cutting_kn": 0.0, "thrust_seal_kn": params["mu_seal"] * pm * seal_area,
                "thrust_backup_kn": params["mu_backup"] * g.backup_weight_kn,
                "torque_cutting_knm": params["cutting_energy_kj_m3"] * g.face_area_m2 * p_rev_m / (2.0 * math.pi),
                "torque_face_knm": params["face_friction"] * p_ch * k_face,
                "torque_stirring_knm": params["muck_shear_kpa"] * k_face,
                "torque_periphery_knm": params["periphery_friction"] * pm * math.pi * g.diameter_m * g.head_rim_length_m * g.radius_m,
                "torque_bearing_knm": g.bearing_torque_knm}
    pm_rock = mean_shield_pressure_kpa(z0_m, g.diameter_m, ground.gamma, gw_depth_m, ground.K0) if closed_face else 0.0
    f_n, f_r, _ = csm_cutter_forces_kn(ground.ucs_MPa, pen_mm, g.cutter_radius_m, g.cutter_tip_width_m, g.cutter_spacing_m,
                                       params["tensile_ratio"])
    u_axis = GAMMA_W * max(z0_m - gw_depth_m, 0.0)
    return {"thrust_skin_kn": params["mu_rock_skin"] * g.machine_weight_kn,
            "thrust_shield_kn": params["mu_rock_shield"] * skin_area * pm_rock,
            "thrust_face_kn": p_ch * g.face_area_m2, "thrust_cutting_kn": g.n_face_cutters * f_n,
            "thrust_seal_kn": params["mu_seal"] * u_axis * seal_area,
            "thrust_backup_kn": params["mu_backup"] * g.backup_weight_kn,
            "torque_cutting_knm": f_r * g.cutter_radius_sum_m, "torque_face_knm": params["face_friction"] * p_ch * k_face,
            "torque_stirring_knm": 0.0,
            "torque_periphery_knm": params["periphery_friction"] * u_axis * math.pi * g.diameter_m * g.head_rim_length_m * g.radius_m,
            "torque_bearing_knm": g.bearing_torque_knm}
