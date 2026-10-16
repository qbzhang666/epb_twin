"""Project adapter: how one project's raw per-ring record maps onto the standard record (schema.py).

An adapter is a small YAML (or JSON) file, kept with the project and not with this package when its vocabulary is
confidential:

    name: example
    columns: {face_crown_bar: p_upper, face_axis_bar: p_axis}     # standard name: raw column
    ground_classes: {order: [F, S, W, R], aliases: {FILL: F}}       # canonical codes in face order, optional aliases
    modes:
      default: full_EPB
      rules:                                                       # first match wins, on lower-case text
        - {mode: EPB_air_bubble, contains: [bubble, air]}
        - {mode: semi_EPB, contains: [semi, partial, half]}
    limits: {max_cover_spread_m: 1.5}
    units: {face_crown_bar: kPa, thrust_kn: MN, cutterhead_torque_mnm: kN m}   # only where the record differs

Units the normaliser converts (to the standard unit of each field, see schema.py):
    pressure  bar, kPa, MPa, Pa, psi        force   kN, MN, N, tf (tonne-force)
    torque    MN m, kN m, N m, tf m         length  m, mm, cm, ft
    speed     mm/min, m/h, mm/s, m/min      volume  m3, L
    rotation  rpm, rad/s                    ratio   %, fraction
"""
import json, re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Adapter:
    name: str
    columns: dict = field(default_factory=dict)
    ground_order: list = field(default_factory=list)
    ground_aliases: dict = field(default_factory=dict)
    mode_default: str = "full_EPB"
    mode_rules: list = field(default_factory=list)
    max_cover_spread_m: float = 1.5
    units: dict = field(default_factory=dict)

    def __post_init__(self):
        sep = r"\s*-?\s*"
        pats = [sep.join(re.escape(ch) for ch in code.upper()) for code in sorted(self.ground_order, key=len, reverse=True)]
        self._ground_re = re.compile(r"(?<![A-Z0-9])(?:" + "|".join(pats) + r")(?![A-Z0-9])") if pats else None

    @property
    def modes(self):
        return [self.mode_default] + [r["mode"] for r in self.mode_rules]

    def rename(self, raw):
        """Raw table -> standard column names (columns not in the map are kept as they are)."""
        inv = {r: s for s, r in self.columns.items() if r in raw.columns and r != s}
        return raw.rename(columns=inv)

    def ground_set(self, s):
        """Free text of ground classes -> canonical codes in face order joined by '/' ('' if none)."""
        if not isinstance(s, str) or self._ground_re is None:
            return ""
        up = s.upper()
        toks = {re.sub(r"[\s-]", "", t) for t in self._ground_re.findall(up)}
        toks |= {code for alias, code in self.ground_aliases.items() if alias.upper() in up}
        return "/".join(c for c in self.ground_order if c.upper() in {t.upper() for t in toks})

    def mode(self, s):
        """Free text of an operating mode -> canonical mode ('' if empty)."""
        if not isinstance(s, str) or not s.strip():
            return ""
        low = s.lower()
        for r in self.mode_rules:
            if any(w in low for w in r["contains"]):
                return r["mode"]
        return self.mode_default


def load_adapter(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        d = yaml.safe_load(text)
    else:
        d = json.loads(text)
    g, m = d.get("ground_classes", {}), d.get("modes", {})
    return Adapter(name=d.get("name", p.stem), columns=d.get("columns", {}) or {}, ground_order=g.get("order", []),
                   ground_aliases=g.get("aliases", {}) or {}, mode_default=m.get("default", "full_EPB"),
                   mode_rules=m.get("rules", []) or [], max_cover_spread_m=float(d.get("limits", {}).get("max_cover_spread_m", 1.5)),
                   units=d.get("units", {}) or {})
