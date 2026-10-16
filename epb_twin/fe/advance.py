# -*- coding: utf-8 -*-
"""Ring-by-ring advance of an EPB shield in a staged 3D finite element model (Kratos GeoMechanicsApplication,
drained, no groundwater): face pressure, shield and gap, tail grout, lining, machine weight.

Run in a case directory that holds machine.yaml and ground.yaml (or set ADV_MACHINE, ADV_GROUND):
    python -m epb_twin.fe.advance mesh [--rings 10] [--h 1.2]   -> data/<prefix>_r<N><mesh>_sl.npz       (gmsh)
    python -m epb_twin.fe.advance run  [--rings 10] [--steps 4] -> results/<prefix>_<tag>_summary.json   (Kratos)

Frame: x across the tunnel (axis at x = 0), y along the drive (the drive starts at y = 0, the boundary, and goes
towards +y), z up (ground surface z = 0). One bore.

Stages: 1 initial (K0). Then one stage per ring k = 1..N, the face moving from y = L_ring (k - 1) to L_ring k:
  - the ring's core is excavated and its face carries the chamber pressure p(z) = p_crown + g_face (z_crown - z);
    the previous face's load is switched off;
  - contact mode (default): the wall converges freely until it meets the shield, whose radius falls from the overcut
    at the face to the overcut plus half the taper at the tail; the gap carries bentonite at a fraction of the
    chamber pressure;
  - the ring that leaves the tail is loaded by the grout pressure p_g(z); one stage later the lining is activated
    stress-free and the grout load removed, so the lining takes the load the grout carried;
  - the shield weight rests on the invert under the shield and the back-up weight on the lined rings behind it.
Machine and ground come from machine.yaml and ground.yaml (epb_twin.spec); the case values (window depth, face and
grout pressures, gap support, stiffness factors) are environment variables, so that a cluster array can vary them.
"""
import json, math, os, sys, time
import numpy as np

WORK = os.environ.get("ADV_WORKDIR", os.getcwd()); DATA = os.path.join(WORK, "data"); RES = os.path.join(WORK, "results")
os.makedirs(DATA, exist_ok=True); os.makedirs(RES, exist_ok=True)
MODE = sys.argv[1] if len(sys.argv) > 1 else "help"
ARGS = dict(zip(sys.argv[2::2], sys.argv[3::2]))

ENV = os.environ.get
# machine and ground files (epb_twin.spec); environment variables still override single values for sweeps
try:
    from ..spec import load_machine, load_ground
except ImportError:                                                    # run as a plain script
    from epb_twin.spec import load_machine, load_ground
MACH = load_machine(ENV("ADV_MACHINE", os.path.join(WORK, "machine.yaml")))
GRND = load_ground(ENV("ADV_GROUND", os.path.join(WORK, "ground.yaml")))
PREFIX = ENV("ADV_PREFIX", "adv")
D_EXC = MACH.excavated_diameter_m; R_EXC = D_EXC / 2; RING = MACH.ring_length_m
T_LIN = MACH.lining_thickness_m; R_LIN_MID = MACH.lining_extrados_diameter_m / 2 - T_LIN / 2   # shell lining radius
OVERCUT = MACH.overcut_m; TAPER = float(ENV("ADV_TAPER", MACH.taper_m)); L_SHIELD = float(ENV("ADV_SHIELD_L", MACH.shield_length_m))
GAP_F = float(ENV("ADV_GAP_F", 0.5))                                  # prescribed mode: fraction of the gap imposed
SHIELD_MODE = ENV("ADV_SHIELD_MODE", "contact")                       # contact: the wall converges freely until it meets
#   the shield (gap = overcut + taper/2 x s/L), then is held while pressing on it (released if it pulls); prescribed: old
#   model, GAP_F x overcut imposed on the new ring's wall
GAP_P = float(ENV("ADV_GAP_P_FRAC", 1.0))                             # gap filled with bentonite at this fraction of the
#   chamber pressure (contact mode); the record injects about 9.6 m3 of bentonite per ring against a 6.5 m3 gap
Z_AXIS = float(ENV("ADV_Z_AXIS", -2.0 * D_EXC))                      # axis level; each window sets its own
Z_ROCK = float(ENV("ADV_Z_ROCK", -1.0 * D_EXC))                      # rock head (base of the first default layer)
MESH = ENV("ADV_MESH", "")                                            # mesh suffix, one per geometry (window)
N_RINGS = int(ARGS.get("--rings", ENV("ADV_RINGS", 10)))
FACE_CROWN = float(ENV("ADV_FACE_CROWN_BAR", 2.0)) * 1e5              # record: mean 2.05 bar at the upper sensor pair
FACE_GRAD = float(ENV("ADV_FACE_GRAD", 12.0)) * 1e3                   # Pa/m; 3.0 bar near the axis (record 3.02, lower pair)
GROUT_CROWN = float(ENV("ADV_GROUT_CROWN_BAR", 2.5)) * 1e5            # assumed
GROUT_GRAD = float(ENV("ADV_GROUT_GRAD", 12.0)) * 1e3
E_LIN = float(ENV("ADV_E_LINING", MACH.lining_E_GPa * 1e9)) * float(ENV("ADV_JOINT_F", MACH.joint_factor))   # joints as a reduced stiffness
ESCALE = float(ENV("ADV_ESCALE", 1.0))
TAG = ENV("ADV_TAG", f"base_r{N_RINGS}")
# materials from the ground file: E (Pa), nu, gamma (kN/m3), K0; E overridable per material with ADV_E_<NAME> (Pa),
# and for the two-layer names also with the earlier ADV_FILL_E / ADV_ROCK_E
_LEGACY_E = {GRND.default_layers[0]: "ADV_FILL_E", GRND.default_layers[-1]: "ADV_ROCK_E"}   # upper / lower default layer
GROUND = {n: (float(ENV(_LEGACY_E.get(n, ""), "") or ENV(f"ADV_E_{n.upper()}", m.E_MPa * 1e6)), m.nu, m.gamma, m.K0)
          for n, m in GRND.materials.items()}
# horizontal layers, "Name:z_base;Name:z_base;...;Name" (z of each base, m, ground at 0, top down; the last layer runs
# to the model base; names from the ground file). Empty keeps the two default layers split at the rock head.
LAYERS = [(t.split(":")[0], float(t.split(":")[1]) if ":" in t else None)
          for t in (ENV("ADV_LAYERS", "") or f"{GRND.default_layers[0]}:{Z_ROCK};{GRND.default_layers[1]}").split(";") if t]
# Mohr-Coulomb (drained, no dilatancy, tension cut-off 0.25 c) with the strengths of the ground file; MC off is elastic.
MC = ENV("ADV_MC", "0") == "1"
# machine weight. The shield's weight rests on the invert under the shield (moving with it); the back-up
# gantries' weight goes through their wheels onto the lined rings behind the tail. Zero keeps the earlier cases.
W_SHIELD = float(ENV("ADV_SHIELD_W_KN", 0.0)) * 1e3                  # N; set per case (machine.yaml weight)
W_BACKUP = float(ENV("ADV_BACKUP_W_KN", 0.0)) * 1e3                  # N; set per case (machine.yaml weight)
L_BACKUP = float(ENV("ADV_BACKUP_L", MACH.backup_length_m))           # m; back-up behind the shield
INVERT_HALF_DEG = 45.0                                                # bearing sector each side of the invert
SUM_LOADS = ENV("ADV_LOADS", "sum") == "sum"                          # "groups": the earlier per-group point loads
# Lining: "solid" (default) = an annulus of continuum elements (thickness T_LIN inside the excavated radius) activated
# with zero stress; "shell" = the earlier shells on the wall. GeoMechanics resets DISPLACEMENT at every stage, and the
# structural shells compute their strain from it, so a shell lining sheds about 5 % of its load to the ground at every
# stage (checked: N of a ring falls 2.6 -> 1.9 MN/m over 7 stages with no new load). Continuum elements keep their stress.
LINING = ENV("ADV_LINING", "solid")
H_LIN = float(ENV("ADV_H_LIN", 0.5))                                  # element size in the lining annulus
_LEGACY_C = {GRND.default_layers[0]: ("ADV_FILL_C", "ADV_FILL_PHI"), GRND.default_layers[-1]: ("ADV_ROCK_C", "ADV_ROCK_PHI")}
STRENGTH = {n: (float(ENV(_LEGACY_C.get(n, ("", ""))[0], "") or ENV(f"ADV_C_{n.upper()}", m.c_kPa)) * 1e3,
                float(ENV(_LEGACY_C.get(n, ("", ""))[1], "") or ENV(f"ADV_PHI_{n.upper()}", m.phi_deg)))
            for n, m in GRND.materials.items()}                       # c (Pa), phi (deg); used with ADV_MC=1
SIGN_R = float(ENV("ADV_REACTION_SIGN", 1.0))                         # REACTION = support force on the ground (base sum = +weight)
N_SHIELD = int(math.ceil(L_SHIELD / RING))                            # rings inside the shield (7 for 15 m)
N_BACKUP = int(math.ceil(L_BACKUP / RING))                            # back-up length in rings (32 for 75 m)
MESH_FILE = f"{PREFIX}_r{N_RINGS}{MESH}{'_sl' if LINING == 'solid' else ''}.npz"


def mesh():
    import gmsh
    h = float(ARGS.get("--h", 1.2))
    X0, X1 = -5 * D_EXC, 5 * D_EXC; Y0, Y1 = 0.0, N_RINGS * RING + 3 * D_EXC; ZB = Z_AXIS - 3 * D_EXC
    gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0); gmsh.model.add(PREFIX)
    occ = gmsh.model.occ; tagged = []
    ztop = 0.0
    for name, zbase in LAYERS:                                         # one box per layer, top down
        zb_ = ZB if zbase is None else zbase
        tagged.append(((3, occ.addBox(X0, Y0, zb_, X1 - X0, Y1 - Y0, ztop - zb_)), "soil", 0, name)); ztop = zb_
    for k in range(N_RINGS):
        tagged.append(((3, occ.addCylinder(0, k * RING, Z_AXIS, 0, RING, 0, R_EXC)), f"ring{k + 1}", 3, None))
        if LINING == "solid":                                          # the ring's core inside the lining annulus
            tagged.append(((3, occ.addCylinder(0, k * RING, Z_AXIS, 0, RING, 0, R_EXC - T_LIN)), f"core{k + 1}", 4, None))
    tagged.append(((3, occ.addCylinder(0, N_RINGS * RING, Z_AXIS, 0, Y1 - N_RINGS * RING, 0, R_EXC)), "bore_rest", 3, None))
    occ.synchronize()
    out, omap = occ.fragment([d for d, *_ in tagged], [])
    occ.synchronize()
    best = {}
    for (d, k, p, u), pieces in zip(tagged, omap):
        for dim, tg in pieces:
            b = best.setdefault(tg, dict(kind="soil", p=-1, layer=None))
            if u is not None:
                b["layer"] = u
            if p > b["p"]:
                b["kind"], b["p"] = k, p
    for tg, b in best.items():
        if b["layer"] is None:
            zc = occ.getCenterOfMass(3, tg)[2]
            b["layer"] = next(n for n, zbase in LAYERS if zbase is None or zc > zbase)
    bs = sorted({s for tg, b in best.items() if b["kind"].startswith(("ring", "bore", "core")) for _, s in gmsh.model.getBoundary([(3, tg)], oriented=False)})
    f1 = gmsh.model.mesh.field.add("Distance"); gmsh.model.mesh.field.setNumbers(f1, "SurfacesList", bs)
    gmsh.model.mesh.field.setNumber(f1, "Sampling", 40)
    f2 = gmsh.model.mesh.field.add("Threshold"); gmsh.model.mesh.field.setNumber(f2, "InField", f1)
    for k_, v_ in (("SizeMin", h), ("SizeMax", 10.0), ("DistMin", 1.5), ("DistMax", 35.0)):
        gmsh.model.mesh.field.setNumber(f2, k_, v_)
    fb = f2
    if LINING == "solid":                                              # finer in the annulus: two to three elements through
        cs = sorted({s for tg, b in best.items() if b["kind"].startswith("ring") for _, s in gmsh.model.getBoundary([(3, tg)], oriented=False)})
        f3 = gmsh.model.mesh.field.add("Distance"); gmsh.model.mesh.field.setNumbers(f3, "SurfacesList", cs)
        gmsh.model.mesh.field.setNumber(f3, "Sampling", 40)
        f4 = gmsh.model.mesh.field.add("Threshold"); gmsh.model.mesh.field.setNumber(f4, "InField", f3)
        for k_, v_ in (("SizeMin", H_LIN), ("SizeMax", 10.0), ("DistMin", T_LIN), ("DistMax", 6.0)):
            gmsh.model.mesh.field.setNumber(f4, k_, v_)
        fb = gmsh.model.mesh.field.add("Min"); gmsh.model.mesh.field.setNumbers(fb, "FieldsList", [f2, f4])
    gmsh.model.mesh.field.setAsBackgroundMesh(fb)
    for k_, v_ in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0), ("Mesh.MeshSizeFromCurvature", 0),
                   ("Mesh.MeshSizeMax", 10.0), ("Mesh.Algorithm3D", 10), ("Mesh.Optimize", 1)):
        gmsh.option.setNumber(k_, v_)
    t0 = time.time()
    try:
        gmsh.model.mesh.generate(3)
    except Exception:
        gmsh.model.mesh.clear(); gmsh.option.setNumber("Mesh.Algorithm3D", 1); gmsh.model.mesh.generate(3)
    nt, xyz, _ = gmsh.model.mesh.getNodes(); xyz = xyz.reshape(-1, 3); idx = np.full(int(nt.max()) + 1, -1); idx[nt.astype(int)] = np.arange(len(nt))
    names = ["soil"] + [f"ring{k + 1}" for k in range(N_RINGS)] + ["bore_rest"] + ([f"core{k + 1}" for k in range(N_RINGS)] if LINING == "solid" else [])
    lnames = list(GROUND)
    tets, tk, tl = [], [], []
    for tg, b in best.items():
        et, _, en = gmsh.model.mesh.getElements(3, tg)
        for e_t, conn in zip(et, en):
            if e_t == 4:
                c = idx[np.asarray(conn, int).reshape(-1, 4)]; tets.append(c)
                tk += [names.index(b["kind"])] * len(c); tl += [lnames.index(b["layer"])] * len(c)
    gmsh.finalize()
    tets = np.vstack(tets); used = np.unique(tets); re = np.full(len(xyz), -1); re[used] = np.arange(len(used))
    xyz, tets = xyz[used], re[tets]
    v = np.einsum("ij,ij->i", np.cross(xyz[tets[:, 1]] - xyz[tets[:, 0]], xyz[tets[:, 2]] - xyz[tets[:, 0]]), xyz[tets[:, 3]] - xyz[tets[:, 0]])
    tets[v < 0] = tets[v < 0][:, [0, 2, 1, 3]]
    np.savez_compressed(os.path.join(DATA, MESH_FILE), xyz=xyz, tets=tets, kind=np.array(tk), layer=np.array(tl),
                        names=np.array(names), layer_names=np.array(lnames), domain=np.array([X0, X1, Y0, Y1, ZB, 0.0]))
    print(f"mesh: {len(xyz)} nodes, {len(tets)} tets, {time.time() - t0:.0f} s")


def tet_faces(T):
    return np.sort(np.r_[T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]], 1)


def run():
    try:
        from . import geo_kratos as GK
    except ImportError:
        import geo_kratos as GK
    t0 = time.time(); steps = int(ARGS.get("--steps", 4 if SHIELD_MODE == "contact" else 2))
    M = dict(np.load(os.path.join(DATA, MESH_FILE), allow_pickle=False))
    zb = Z_AXIS - 3 * D_EXC
    if abs(float(M["domain"][4]) - zb) > 1e-6:
        raise SystemExit(f"mesh {MESH_FILE} was built for another axis level (base {M['domain'][4]} vs {zb})")
    names = [str(n) for n in M["names"]]; lnames = [str(n) for n in M["layer_names"]]
    xyz, tets, kind, layer = M["xyz"], M["tets"], M["kind"], M["layer"]
    X0, X1, Y0, Y1, ZB, _ = M["domain"]; e = 1e-6
    P = GK.Project(os.path.join(RES, f"{PREFIX}_{TAG}_kratos"), xyz); parts = {}
    for i, u in enumerate(lnames):
        for j, k in enumerate(names):
            sel = (kind == j) & (layer == i)
            if sel.any():
                nm = f"{k}_{u}"; P.part(nm, GK.UPW4, tets[sel]); parts[nm] = (k, u)
    sides_x = np.where((xyz[:, 0] < X0 + e) | (xyz[:, 0] > X1 - e))[0]
    sides_y = np.where((xyz[:, 1] < Y0 + e) | (xyz[:, 1] > Y1 - e))[0]
    base = np.where(xyz[:, 2] < ZB + e)[0]
    P.nodes("sides_x", sides_x); P.nodes("sides_y", sides_y); P.nodes("base", base)
    boundary = set(sides_x.tolist()) | set(sides_y.tolist()) | set(base.tolist())
    bore_k = [names.index(f"ring{k + 1}") for k in range(N_RINGS)] + [names.index("bore_rest")]
    ring_k = [[names.index(f"ring{k + 1}")] + ([names.index(f"core{k + 1}")] if f"core{k + 1}" in names else []) for k in range(N_RINGS)] \
        + [[names.index("bore_rest")]]
    if LINING == "solid" and "core1" not in names:
        raise SystemExit(f"{MESH_FILE} has no lining annulus: build it with ADV_LINING=solid ... mesh")
    all_bore = [i for g_ in ring_k for i in g_]
    soil_nodes = np.unique(tets[~np.isin(kind, all_bore)])
    rings = []
    for k in range(N_RINGS):
        T = tets[np.isin(kind, ring_k[k])]; F = tet_faces(T)
        wallF = F[np.isin(F, soil_nodes).all(1)]                       # the ring's faces on the ground: the excavated wall
        nxt = np.unique(tets[np.isin(kind, ring_k[k + 1])])
        faceF = F[np.isin(F, nxt).all(1)]                              # shared with the next ring: the face
        faceF = faceF[~np.isin(faceF, soil_nodes).all(1)] if len(faceF) else faceF
        wall = np.unique(wallF)
        rings.append(dict(wallF=wallF, wall=wall, faceF=faceF, face=np.setdiff1d(np.unique(faceF), wall)))

    def nodal_pressure(F, p_of_z, outward):
        """Nodal forces of a pressure p(z) on triangles F: towards +y (face) or radially outwards (grout)."""
        a, b, c = xyz[F[:, 0]], xyz[F[:, 1]], xyz[F[:, 2]]
        A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1); zc = (a[:, 2] + b[:, 2] + c[:, 2]) / 3
        f = np.zeros((len(xyz), 3))
        if outward:
            cen = (a + b + c) / 3; rv = cen - np.array([0.0, 0.0, Z_AXIS]); rv[:, 1] = 0.0
            nrm = rv / np.linalg.norm(rv, axis=1)[:, None]
        else:
            nrm = np.tile([0.0, 1.0, 0.0], (len(F), 1))
        fv = (p_of_z(zc) * A)[:, None] * nrm / 3
        for j in range(3):
            np.add.at(f, F[:, j], fv)
        idx = np.unique(F); return idx, f[idx]

    def invert_weight(r, W):
        """Downward nodal forces (N) totalling W on the ring's wall within INVERT_HALF_DEG of the invert, cos-weighted."""
        F = r["wallF"]; a, b, c = xyz[F[:, 0]], xyz[F[:, 1]], xyz[F[:, 2]]
        A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1); cen = (a + b + c) / 3
        ang = np.degrees(np.arccos(np.clip(-(cen[:, 2] - Z_AXIS) / R_EXC, -1, 1)))      # 0 at the invert
        w = np.where(ang < INVERT_HALF_DEG, np.cos(np.radians(ang)), 0.0) * A
        f = np.zeros((len(xyz), 3)); fz = -W * w / w.sum() / 3
        for j in range(3):
            np.add.at(f[:, 2], F[:, j], fz)
        return f

    z_crown = Z_AXIS + R_EXC
    p_face = lambda z: FACE_CROWN + FACE_GRAD * (z_crown - z)
    p_grout = lambda z: GROUT_CROWN + GROUT_GRAD * (z_crown - z)
    loads = {}
    for k, r in enumerate(rings):
        if len(r["faceF"]):
            fi, ff = nodal_pressure(r["faceF"], p_face, False); fi = np.setdiff1d(fi, r["wall"])
            keep = np.isin(nodal_pressure(r["faceF"], p_face, False)[0], fi)
            loads[f"face{k + 1}"] = (fi, nodal_pressure(r["faceF"], p_face, False)[1][keep]); P.point_loads(f"face{k + 1}", fi)
        wv = invert_weight(r, W_SHIELD / N_SHIELD) if W_SHIELD > 0 else None
        if SHIELD_MODE == "contact" and (GAP_P > 0 or wv is not None):
            bi, bf = nodal_pressure(r["wallF"], lambda z: GAP_P * p_face(z), True)
            if wv is not None:
                bf = bf + wv[bi]
            bi_ok = np.array([i for i in bi if i not in boundary], int); keep = np.isin(bi, bi_ok)
            loads[f"gap{k + 1}"] = (bi_ok, bf[keep]); P.point_loads(f"gap{k + 1}", bi_ok)
        if W_BACKUP > 0:
            wb = invert_weight(r, W_BACKUP / N_BACKUP); bk = np.array([i for i in np.flatnonzero(np.abs(wb[:, 2]) > 0) if i not in boundary], int)
            loads[f"bk{k + 1}"] = (bk, wb[bk]); P.point_loads(f"bk{k + 1}", bk)
        gi, gf = nodal_pressure(r["wallF"], p_grout, True)
        gi_ok = np.array([i for i in gi if i not in boundary], int); keep = np.isin(gi, gi_ok)
        loads[f"grout{k + 1}"] = (gi_ok, gf[keep]); P.point_loads(f"grout{k + 1}", gi_ok)
    if SUM_LOADS:   # one point-load condition per node carrying the sum of the active loads (groups share boundary nodes)
        for nm_ in list(loads):
            P.conds.pop(nm_, None); P.groups.pop(nm_, None)
        all_idx = np.unique(np.concatenate([v[0] for v in loads.values()])); P.point_loads("allload", all_idx)
        T_prev = np.zeros((len(xyz), 3))
    # the lining: shells on each ring's wall (a part of its own, activated when the ring is lined)
    lin_cen = {}
    for k, r in enumerate(rings):
        if LINING == "solid":
            Tl = tets[kind == names.index(f"ring{k + 1}")]               # the annulus: same nodes, concrete, inactive until lined
            P.part(f"lining{k + 1}", GK.UPW4, Tl); lin_cen[k] = xyz[Tl].mean(axis=1)
        else:
            P.part(f"lining{k + 1}", GK.SHELL3, r["wallF"])
        parts[f"lining{k + 1}"] = ("lining", None)
    surf = np.where(xyz[:, 2] > -e)[0]
    trib = np.zeros(len(xyz))                                          # tributary wall area of each node
    for r in rings:
        F = r["wallF"]; A = 0.5 * np.linalg.norm(np.cross(xyz[F[:, 1]] - xyz[F[:, 0]], xyz[F[:, 2]] - xyz[F[:, 0]]), axis=1)
        for j in range(3):
            np.add.at(trib, F[:, j], A / 3)
    nrm = np.zeros((len(xyz), 3)); exc = np.full(len(xyz), 10 ** 6)  # outward normal; ring in which a wall node is cut
    for k, r in enumerate(rings):
        cc = xyz[r["wall"]] - np.array([0.0, 0.0, Z_AXIS]); cc[:, 1] = 0.0; nrm[r["wall"]] = cc / np.linalg.norm(cc, axis=1)[:, None]
        exc[r["wall"]] = np.minimum(exc[r["wall"]], k)
    urad = np.zeros(len(xyz)); contact = set(); zone_at = {}; on_step_at = {}; contact_log = {}
    P.write_mesh()
    print(f"model: {len(xyz)} nodes, {len(tets)} tets, {N_RINGS} rings, shield {N_SHIELD} rings; written {time.time() - t0:.0f} s", flush=True)

    soil = {u: GK.soil_material(GROUND[u][0] * ESCALE, GROUND[u][1], GROUND[u][2], STRENGTH[u][0] if MC else 0.0,
                                STRENGTH[u][1] if MC else 30.0, k0=GROUND[u][3], mc=MC) for u in GROUND}
    lin_mat = GK.concrete_material(E_LIN, 0.2, 24.0) if LINING == "solid" else GK.shell_material(E_LIN, T_LIN)
    mats = {nm: (soil[u] if k != "lining" else lin_mat) for nm, (k, u) in parts.items()}
    fix = {"sides_x": (1, 0, 0), "sides_y": (0, 1, 0), "base": (1, 1, 1)}
    ground_parts = [nm for nm, (k, u) in parts.items() if k != "lining"]
    linings = [f"lining{k + 1}" for k in range(N_RINGS)]
    def make_on_step(k, zone):
        """Unilateral shield contact, updated after every converged step (lagged active set)."""
        import KratosMultiphysics as KM
        sb = (k + 1) * RING - xyz[zone, 1]; gap = OVERCUT + TAPER / 2 * np.clip(sb / L_SHIELD, 0.0, 1.0)
        dofs = (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z)

        def on_step(st):
            nd = [st._nodes[int(i)] for i in zone]
            D = np.array([list(n_.GetSolutionStepValue(KM.DISPLACEMENT)) for n_ in nd])
            u = urad[zone] - np.einsum("ij,ij->i", D, nrm[zone])                  # convergence since cut, inwards > 0
            Rn = SIGN_R * np.einsum("ij,ij->i", np.array([list(n_.GetSolutionStepValue(KM.REACTION)) for n_ in nd]), nrm[zone])
            on_, off_ = 0, 0
            for j, i in enumerate(zone):
                i = int(i)
                if i in contact and Rn[j] < 0:
                    for v in dofs:
                        nd[j].Free(v)
                    contact.discard(i); off_ += 1
                elif i not in contact and u[j] >= gap[j]:
                    for v in dofs:
                        nd[j].Fix(v)
                    contact.add(i); on_ += 1
            contact_log.setdefault(k, []).append((on_, off_, int(np.isin(zone, list(contact)).sum()) if contact else 0,
                                                  round(float(u.max()) * 1e3, 1)))
        return on_step

    P.stage("initial", {nm: mats[nm] for nm in parts}, active=ground_parts, inactive=linings, k0=ground_parts, bc={"fix": fix})
    active = list(ground_parts); released = []; lined = []; records = []; held = set(); held_at = {}; shield = {}; lined_at = {}; lining_forces = {}
    lining_cen = {k: xyz[r["wallF"]].mean(axis=1) for k, r in enumerate(rings)}   # element centroids (wallF order = element order)
    for k in range(N_RINGS):
        r = rings[k]
        ring_parts = [nm for nm, (kk, u) in parts.items() if kk in (f"ring{k + 1}", f"core{k + 1}")]
        active = [p for p in active if p not in ring_parts]
        # shield: the new ring's wall converges onto the shield (front gap), then is held while the shield passes
        w = np.array([i for i in r["wall"] if i not in boundary and i not in held], int); held |= set(w.tolist())
        if SHIELD_MODE == "contact":
            bc = {"fix": fix}
        else:
            c = xyz[w] - np.array([0.0, 0.0, Z_AXIS]); c[:, 1] = 0.0; rr = np.linalg.norm(c, axis=1)
            U = -GAP_F * OVERCUT * c / rr[:, None]
            bc = {"fix": fix, "prescribed": (w, U)}
        load_idx, load_F, on, off = [], [], [], []
        if f"face{k + 1}" in loads:
            fi, ff = loads[f"face{k + 1}"]; load_idx.append(fi); load_F.append(ff); on.append(f"face{k + 1}")
        if k > 0:
            off.append(f"face{k}")
        if f"gap{k + 1}" in loads:
            bi, bf = loads[f"gap{k + 1}"]; load_idx.append(bi); load_F.append(bf); on.append(f"gap{k + 1}")
        # the ring leaving the tail: released, grout pressure on the ground
        jt = k - N_SHIELD
        free = []
        if jt >= 0:
            nxt_wall = set(rings[jt + 1]["wall"].tolist()) if jt + 1 < N_RINGS else set()
            free = [i for i in rings[jt]["wall"] if i not in boundary and i not in nxt_wall]
            held -= set(free)
            gi, gf = loads[f"grout{jt + 1}"]; load_idx.append(gi); load_F.append(gf); on.append(f"grout{jt + 1}")
            if f"gap{jt + 1}" in loads:
                off.append(f"gap{jt + 1}")
            released.append(jt)
        # the ring released one stage earlier: lined, its grout load removed (taken by the lining)
        jl = jt - 1; new_lining = []
        if jl >= 0:
            new_lining = [f"lining{jl + 1}"]; lined.append(jl); off.append(f"grout{jl + 1}")
            active = active + new_lining
        if jl >= 0 and f"bk{jl + 1}" in loads:                         # the back-up reaches the ring just lined
            bi, bf = loads[f"bk{jl + 1}"]; load_idx.append(bi); load_F.append(bf); on.append(f"bk{jl + 1}")
        bk_on = [f"bk{j + 1}" for j in range(max(jl - N_BACKUP + 1, 0), max(jl, 0)) if f"bk{j + 1}" in loads]
        bk_off = [f"bk{j + 1}" for j in range(0, max(jl - N_BACKUP + 1, 0)) if f"bk{j + 1}" in loads]
        if load_idx:
            bc["loads"] = (np.concatenate(load_idx), np.concatenate(load_F))
        keep_gap = [f"gap{j + 1}" for j in range(max(jt + 1, 0), k) if f"gap{j + 1}" in loads]   # still under the shield
        bc["load_groups"] = on + keep_gap + bk_on
        bc["off_groups"] = off + [f"face{j + 1}" for j in range(k - 1)] + [f"gap{j + 1}" for j in range(max(jt, 0)) if f"gap{j + 1}" in loads] + bk_off
        if SUM_LOADS:
            act = ([f"face{k + 1}"] + [f"gap{j + 1}" for j in range(max(jt + 1, 0), k + 1)] + ([f"grout{jt + 1}"] if jt >= 0 else [])
                   + ([f"bk{j + 1}" for j in range(max(jl - N_BACKUP + 1, 0), jl + 1)] if jl >= 0 else []))
            T = np.zeros((len(xyz), 3))
            for g_ in act:
                if g_ in loads:
                    np.add.at(T, loads[g_][0], loads[g_][1])
            bc["loads"] = (all_idx, T[all_idx]); bc["loads_from"] = T_prev[all_idx].copy(); T_prev = T
            bc["load_groups"] = ["allload"]; bc["off_groups"] = []
        bc["free"] = free
        if SHIELD_MODE == "contact":
            bc["on_step"] = make_on_step(k, np.array(sorted(held), int))
        bc["rot_free"] = (np.unique(np.concatenate([rings[j]["wall"] for j in lined])) if lined else []) if LINING == "shell" else []
        P.stage(f"ring{k + 1}", mats, active=active, inactive=ring_parts, steps=steps, bc=bc)
        records.append(dict(stage=f"ring{k + 1}", face_y=(k + 1) * RING, released_ring=jt + 1 if jt >= 0 else None,
                            lined_ring=jl + 1 if jl >= 0 else None))
        held_at[f"ring{k + 1}"] = np.array(sorted(held), int)
        lined_at[f"ring{k + 1}"] = list(lined)
    results = {}

    def after(k, s, model):
        import KratosMultiphysics as KM
        mp = model["PorousDomain"]; name = P.stages[k]["name"]
        D = np.array([list(nd.GetSolutionStepValue(KM.DISPLACEMENT)) for nd in mp.Nodes]); results[name] = D
        if name.startswith("ring"):
            kk_ = int(name[4:]); cut = np.where(exc < kk_)[0]
            urad[cut] -= np.einsum("ij,ij->i", D[cut], nrm[cut])
            if k + 1 < len(P.stages) and P.stages[k + 1]["bc"].get("free"):
                contact.difference_update(int(i) for i in P.stages[k + 1]["bc"]["free"])
        if name.startswith("ring") and lined_at.get(name) and LINING == "solid":
            pi_ = mp.ProcessInfo; lf = {}; r_mid = R_EXC - T_LIN / 2
            for j in lined_at[name]:
                eids = P.eids[f"lining{j + 1}"]; cen = lin_cen[j]
                S = np.array([np.mean([np.array(m_) for m_ in mp.GetElement(int(eid)).CalculateOnIntegrationPoints(KM.CAUCHY_STRESS_TENSOR, pi_)], axis=0) for eid in eids])
                c = cen - np.array([0.0, 0.0, Z_AXIS]); c[:, 1] = 0.0; rr = np.linalg.norm(c, axis=1)
                et = np.c_[-c[:, 2], np.zeros(len(c)), c[:, 0]] / rr[:, None]; st = np.einsum("ei,eij,ej->e", et, S, et)
                th = np.degrees(np.arctan2(c[:, 0], c[:, 2])); sec = np.floor((th + 180.0) / 30.0).astype(int).clip(0, 11)   # 12 sectors, 0 = invert
                Ns, Ms, Nq, Mq = [], [], [], []
                for q in range(12):
                    m = sec == q
                    if m.sum() < 3:
                        Nq.append(None); Mq.append(None); continue
                    Ns.append(st[m].mean() * T_LIN)                    # hoop force per metre (compression < 0)
                    A = np.c_[np.ones(m.sum()), rr[m] - r_mid]; cf, *_ = np.linalg.lstsq(A, st[m], rcond=None)
                    Ms.append(cf[1] * T_LIN ** 3 / 12)                 # linear stress fit through the thickness (approximate)
                    Nq.append(round(float(Ns[-1]) / 1e3, 1)); Mq.append(round(float(Ms[-1]) / 1e3, 1))
                lf[f"ring{j + 1}"] = dict(N_hoop_kN_m=[round(float(min(Ns)) / 1e3, 1), round(float(np.mean(Ns)) / 1e3, 1), round(float(max(Ns)) / 1e3, 1)],
                                          M_hoop_kNm_m=[round(float(min(Ms)) / 1e3, 1), round(float(max(Ms)) / 1e3, 1)],
                                          # 12 sectors of 30 deg: centres -165, -135, ..., 165 deg from the crown (0), positive towards +x; +-165 = invert
                                          N_sector_kN_m=Nq, M_sector_kNm_m=Mq)
            lining_forces[name] = lf
        if name.startswith("ring") and lined_at.get(name) and LINING == "shell":
            import KratosMultiphysics.StructuralMechanicsApplication as SM
            pi_ = mp.ProcessInfo; lf = {}
            for j in lined_at[name]:
                N_, M_ = [], []
                for eid in P.eids[f"lining{j + 1}"]:
                    el = mp.GetElement(int(eid))
                    F_ = np.mean([np.array(m_) for m_ in el.CalculateOnIntegrationPoints(SM.SHELL_FORCE_GLOBAL, pi_)], axis=0)
                    Mm = np.mean([np.array(m_) for m_ in el.CalculateOnIntegrationPoints(SM.SHELL_MOMENT_GLOBAL, pi_)], axis=0)
                    N_.append(F_); M_.append(Mm)
                cen = lining_cen[j]; et = np.c_[-(cen[:, 2] - Z_AXIS), np.zeros(len(cen)), cen[:, 0]]; et /= np.linalg.norm(et, axis=1)[:, None]
                Nt = np.einsum("ei,eij,ej->e", et, np.array(N_), et); Mt = np.einsum("ei,eij,ej->e", et, np.array(M_), et)
                lf[f"ring{j + 1}"] = dict(N_hoop_kN_m=[round(float(Nt.min()) / 1e3, 1), round(float(Nt.mean()) / 1e3, 1), round(float(Nt.max()) / 1e3, 1)],
                                          M_hoop_kNm_m=[round(float(Mt.min()) / 1e3, 2), round(float(Mt.max()) / 1e3, 2)])
            lining_forces[name] = lf
        if name in held_at and len(held_at[name]):
            kk = int(name[4:])
            # ground on the shield: reactions of the held wall nodes, radial (outwards = ground pressing on the skin)
            h = held_at[name]; Rf = np.array([list(mp.GetNode(int(i) + 1).GetSolutionStepValue(KM.REACTION)) for i in h])
            c = xyz[h] - np.array([0.0, 0.0, Z_AXIS]); c[:, 1] = 0.0; n = c / np.linalg.norm(c, axis=1)[:, None]
            Nr = SIGN_R * np.einsum("ij,ij->i", Rf, n)                  # > 0: ground pressing on the shield, < 0: shield pulling
            L = float(np.ptp(xyz[h, 1])); th = np.degrees(np.arctan2(c[:, 0], c[:, 2]))   # 0 = crown, +-180 = invert
            sec = {"crown": np.abs(th) < 45, "springline": (np.abs(th) >= 45) & (np.abs(th) <= 135), "invert": np.abs(th) > 135}
            Rb = np.array([list(mp.GetNode(int(i) + 1).GetSolutionStepValue(KM.REACTION)) for i in base])
            shield[name] = dict(normal_MN=round(float(Nr.sum()) / 1e6, 3), axial_MN=round(float(-SIGN_R * Rf[:, 1].sum()) / 1e6, 3),
                                held_length_m=round(L, 2),
                                mean_pressure_kPa=round(float(Nr.sum()) / float(trib[h].sum()) / 1e3, 1),
                                **{f"p_{kk}_kPa": round(float(Nr[m].sum()) / float(trib[h][m].sum()) / 1e3, 1) for kk, m in sec.items() if m.any()},
                                base_reaction_z_MN=round(float(Rb[:, 2].sum()) / 1e6, 2),
                                contact_area_frac=round(float(trib[[i for i in h if int(i) in contact]].sum()) / float(trib[h].sum()), 3),
                                convergence_mm=dict(mean=round(float(urad[h].mean()) * 1e3, 1), max=round(float(urad[h].max()) * 1e3, 1)),
                                contact_steps=contact_log.get(kk - 1, []),
                                # per ring of the shield, from the face backwards (index 0 = the ring just cut)
                                p_ring_kPa=[round(float(Nr[m].sum()) / float(trib[h][m].sum()) / 1e3, 1) for m in
                                            [(xyz[h, 1] > (kk - j - 1) * RING + 1e-6) & (xyz[h, 1] < (kk - j) * RING - 1e-6)
                                             for j in range(kk)] if m.any()])
        print(f"stage {name}: {time.time() - t0:.0f} s, surface settlement {-D[surf, 2].min() * 1e3:.2f} mm", flush=True)
    if ENV("GEO_NSTAGES"):
        P.stages = P.stages[:int(ENV("GEO_NSTAGES"))]
    GK.run(P, after_stage=after)
    T = sum(results[n] for n in results if n.startswith("ring"))
    ysec = 0.5 * N_RINGS * RING; band = surf[np.abs(xyz[surf, 1] - ysec) < 1.2]; o = np.argsort(xyz[band, 0])
    V = float(getattr(np, "trapezoid", getattr(np, "trapz", None))(-T[band[o], 2], xyz[band[o], 0])) if len(band) > 2 else float("nan")
    axis = surf[np.abs(xyz[surf, 0]) < 1.5]; oa = np.argsort(xyz[axis, 1])
    out = dict(tag=TAG, nodes=len(xyz), tets=len(tets), rings=N_RINGS, shield_rings=N_SHIELD,
               assumed=dict(gap_fraction=GAP_F, taper_m=TAPER, shield_length_m=L_SHIELD, face_crown_bar=FACE_CROWN / 1e5,
                            face_gradient_kPa_m=FACE_GRAD / 1e3, grout_crown_bar=GROUT_CROWN / 1e5, lining_E_eff_GPa=E_LIN / 1e9,
                            ground_E_scale=ESCALE, fill_E_MPa=GROUND[GRND.default_layers[0]][0] / 1e6, rock_E_MPa=GROUND[GRND.default_layers[-1]][0] / 1e6, z_axis_m=Z_AXIS, z_rock_m=Z_ROCK, mesh=MESH,
                            layers=[[n, z, GROUND[n][0] / 1e6] for n, z in LAYERS],
                            shield_mode=SHIELD_MODE, gap_pressure_frac=GAP_P, steps=steps,
                            mohr_coulomb=MC, loads=("sum" if SUM_LOADS else "groups"), lining=LINING, shield_weight_kN=W_SHIELD / 1e3, backup_weight_kN=W_BACKUP / 1e3, backup_length_m=L_BACKUP, strength_c_kPa_phi_deg={u: [v[0] / 1e3, v[1]] for u, v in STRENGTH.items()} if MC else None),
               surface_settlement_max_mm=round(float(-T[surf, 2].min() * 1e3), 2),
               surface_volume_loss_pct_at_y=dict(y_m=ysec, pct=round(V / (math.pi * R_EXC ** 2) * 100, 3)),
               centreline=dict(y=[round(float(v), 2) for v in xyz[axis[oa], 1]], settlement_mm=[round(float(-v * 1e3), 2) for v in T[axis[oa], 2]]),
               shield_ground_force=shield, lining_forces=lining_forces, stages=records, seconds=round(time.time() - t0))
    np.savez_compressed(os.path.join(RES, f"{PREFIX}_{TAG}_fields.npz"), xyz=xyz, surf=surf,
                        **{f"U_{n}": d.astype(np.float32) for n, d in results.items()})
    json.dump(out, open(os.path.join(RES, f"{PREFIX}_{TAG}_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "centreline"}, indent=1))


if __name__ == "__main__":
    if MODE not in ("mesh", "run"):
        print(__doc__); sys.exit(0 if MODE == "help" else 2)
    {"mesh": mesh, "run": run}[MODE]()
