# -*- coding: utf-8 -*-
"""Staged ground models for the GeoMechanicsApplication (U-Pw solver, drained, no groundwater), from arrays.

    from geo_kratos import Project
    P = Project(folder, xyz)                       # node coordinates (m), tunnel frame
    P.part("soil_TprC", "UPwSmallStrainElement3D4N", tets)       # a body part (0-based node indices)
    P.part("shell_slab_0.15", "ShellThinElementCorotational3D3N", tris)
    P.nodes("bore1_face", idx)                     # node groups for boundary conditions
    P.write_mesh()                                 # mesh.mdpa
    P.stage(...)                                   # MaterialParameters_stage{i}.json + ProjectParameters_stage{i}.json
    P.run()                                        # every stage on one Kratos model, as run_multiple_stages.py does

The stage workflow is the GeoMechanicsApplication's own: each stage is a GeoMechanicsAnalysis on the same model,
materials are re-read per stage (the concrete of the substructure replaces the soil it was cast in), the
ApplyK0ProcedureProcess gives the initial stresses, the ApplyExcavationProcess removes and adds elements, and
DISPLACEMENT is the displacement of the stage (reset at its start), TOTAL_DISPLACEMENT the sum. Gravity, the fixed
(zero) pore pressure, the fixed rotations of the solid nodes, point loads and prescribed displacements are set per
node by the Stage class below, scaled by the stage's pseudo-time from 0 to 1.

Soil: GeoMohrCoulombWithTensionCutOff3D (drained), a law verified separately at the material point
(exact Mohr-Coulomb strength at every Lode angle and a return that survives the tensile apex).
"""
import json, os
import numpy as np

UPW4 = "UPwSmallStrainElement3D4N"
LINEAR_SOLVERS = {
    "amgcl": {"solver_type": "amgcl", "tolerance": 1e-9, "max_iteration": 1000, "scaling": False, "verbosity": 1,
              "preconditioner_type": "amg", "smoother_type": "ilu0", "krylov_type": "lgmres", "gmres_krylov_space_dimension": 100,
              "coarsening_type": "aggregation", "use_block_matrices_if_possible": False},
    "lu": {"solver_type": "sparse_lu"},
    "amg2": {"solver_type": "amgcl", "tolerance": 1e-6, "max_iteration": 2000, "scaling": True, "verbosity": 1,
             "preconditioner_type": "amg", "smoother_type": "ilu0", "krylov_type": "bicgstab",
             "coarsening_type": "smoothed_aggregation", "use_block_matrices_if_possible": False},
    "amg4": {"solver_type": "amgcl", "tolerance": 1e-6, "max_iteration": 3000, "scaling": True, "verbosity": 1,
             "preconditioner_type": "amg", "smoother_type": "spai0", "krylov_type": "lgmres", "gmres_krylov_space_dimension": 100,
             "coarsening_type": "ruge_stuben", "use_block_matrices_if_possible": False},
    "ilu": {"solver_type": "amgcl", "tolerance": 1e-6, "max_iteration": 3000, "scaling": True, "verbosity": 1,
            "preconditioner_type": "relaxation", "smoother_type": "ilu0", "krylov_type": "lgmres", "gmres_krylov_space_dimension": 100},
    "amg5": {"solver_type": "amgcl", "tolerance": 1e-6, "max_iteration": 3000, "scaling": True, "verbosity": 1,
             "preconditioner_type": "amg", "smoother_type": "ilu0", "krylov_type": "lgmres", "gmres_krylov_space_dimension": 150,
             "coarsening_type": "aggregation", "use_block_matrices_if_possible": True, "coarse_enough": 2000},
    "amg3": {"solver_type": "amgcl", "tolerance": 1e-6, "max_iteration": 2000, "scaling": True, "verbosity": 1,
             "preconditioner_type": "amg", "smoother_type": "ilu0", "krylov_type": "lgmres", "gmres_krylov_space_dimension": 200,
             "coarsening_type": "aggregation", "use_block_matrices_if_possible": True},
    "cg": {"solver_type": "amgcl", "tolerance": 1e-9, "max_iteration": 2000, "scaling": False, "verbosity": 1,
           "preconditioner_type": "amg", "smoother_type": "damped_jacobi", "krylov_type": "cg", "coarsening_type": "aggregation"}}
SHELL3 = os.environ.get("GEO_SHELL", "ShellThinElement3D3N")
FORCE1 = "UPwForceCondition3D1N"
G = 9.81


def soil_material(E, nu, gamma, c, phi, psi=0.0, t=None, k0=None, mc=True):
    """Drained soil. Unit weight gamma (kN/m3) from the solid density at porosity 0.3, fully saturated with water of
    zero pressure (the model has no groundwater: the weight is the total weight)."""
    n = 0.3; rho = gamma * 1e3 / G; rho_s = (rho - n * 1000.0) / (1 - n)
    V = {"YOUNG_MODULUS": E, "POISSON_RATIO": nu, "DENSITY_SOLID": rho_s, "DENSITY_WATER": 1000.0, "POROSITY": n,
         "BULK_MODULUS_SOLID": 1.0e12, "BULK_MODULUS_FLUID": 2.0e-30, "PERMEABILITY_XX": 1e-12, "PERMEABILITY_YY": 1e-12,
         "PERMEABILITY_ZZ": 1e-12, "PERMEABILITY_XY": 0.0, "PERMEABILITY_YZ": 0.0, "PERMEABILITY_ZX": 0.0,
         "DYNAMIC_VISCOSITY": 1e-3, "BIOT_COEFFICIENT": 1.0, "RETENTION_LAW": "SaturatedLaw", "SATURATED_SATURATION": 1.0,
         "RESIDUAL_SATURATION": 1e-10, "GEO_DRAINAGE_TYPE": "drained",
         "K0_MAIN_DIRECTION": 2, "K0_NC": k0 if k0 is not None else 1.0 - np.sin(np.radians(phi))}
    law = "GeoIncrementalLinearElastic3DLaw" if not mc else "GeoMohrCoulombWithTensionCutOff3D"
    if mc:
        V.update({"GEO_COHESION": c, "GEO_FRICTION_ANGLE": phi, "GEO_DILATANCY_ANGLE": psi,
                  "GEO_TENSILE_STRENGTH": t if t is not None else 0.25 * c})   # below the apex c cot(phi)
    return law, V


def concrete_material(E, nu=0.2, gamma=24.0):
    law, V = soil_material(E, nu, gamma, 0.0, 30.0, mc=False)
    return law, V


def shell_material(E, t, nu=0.2, gamma=24.0):
    return "LinearElasticPlaneStress2DLaw", {"YOUNG_MODULUS": E, "POISSON_RATIO": nu, "THICKNESS": t, "DENSITY": gamma * 1e3 / G}


class Project:
    def __init__(self, folder, xyz):
        self.dir = folder; os.makedirs(folder, exist_ok=True)
        self.xyz = np.asarray(xyz, float); self.parts = {}; self.groups = {}; self.stages = []

    def part(self, name, etype, conn):
        self.parts[name] = (etype, np.asarray(conn, np.int64))

    def point_loads(self, name, idx):
        """UPwForceCondition3D1N on these nodes (their POINT_LOAD is set per stage)."""
        self.conds = getattr(self, "conds", {}); self.conds[name] = np.unique(np.asarray(idx, np.int64))
        self.groups[name] = self.conds[name]

    def nodes(self, name, idx):
        self.groups[name] = np.unique(np.asarray(idx, np.int64))

    def write_mesh(self, name="mesh"):
        pid = {n: i + 1 for i, n in enumerate(self.parts)}; self.pid = pid
        with open(os.path.join(self.dir, name + ".mdpa"), "w") as f:
            f.write("Begin ModelPartData\nEnd ModelPartData\n\n")
            for n in self.parts:
                f.write(f"Begin Properties {pid[n]}\nEnd Properties\n")
            f.write("\nBegin Nodes\n")
            np.savetxt(f, np.c_[np.arange(1, len(self.xyz) + 1), self.xyz], fmt="%d %.6f %.6f %.6f")
            f.write("End Nodes\n\n")
            eid = 1; self.eids = {}
            for n, (et, C) in self.parts.items():
                f.write(f"Begin Elements {et}\n")
                ids = np.arange(eid, eid + len(C)); self.eids[n] = ids
                np.savetxt(f, np.c_[ids, np.full(len(C), pid[n]), C + 1], fmt="%d")
                f.write("End Elements\n\n"); eid += len(C)
            cid = 1; self.cids = {}
            for n, idx in getattr(self, "conds", {}).items():
                f.write(f"Begin Conditions {FORCE1}\n")
                ids = np.arange(cid, cid + len(idx)); self.cids[n] = ids
                np.savetxt(f, np.c_[ids, np.full(len(idx), 1), idx + 1], fmt="%d")
                f.write("End Conditions\n\n"); cid += len(idx)
            for n, (et, C) in self.parts.items():
                f.write(f"Begin SubModelPart {n}\n  Begin SubModelPartNodes\n")
                np.savetxt(f, np.unique(C) + 1, fmt="%d")
                f.write("  End SubModelPartNodes\n  Begin SubModelPartElements\n")
                np.savetxt(f, self.eids[n], fmt="%d")
                f.write("  End SubModelPartElements\nEnd SubModelPart\n\n")
            for n, idx in self.groups.items():
                f.write(f"Begin SubModelPart {n}\n  Begin SubModelPartNodes\n")
                np.savetxt(f, idx + 1, fmt="%d")
                f.write("  End SubModelPartNodes\n")
                if n in self.cids:
                    f.write("  Begin SubModelPartConditions\n"); np.savetxt(f, self.cids[n], fmt="%d"); f.write("  End SubModelPartConditions\n")
                f.write("End SubModelPart\n\n")
        self.mesh = name

    def stage(self, name, materials, active, inactive=(), k0=False, steps=1, reset=True, bc=None, max_iter=int(os.environ.get("GEO_MAXIT", 20)),
              rel_tol=float(os.environ.get("GEO_REL_TOL", 1e-3)), abs_tol=float(os.environ.get("GEO_ABS_TOL", 1.0)), extra_processes=()):
        """materials: {part: (law, variables)} for every part; active: the parts present in this stage (a part not
        present in the previous stage is activated); inactive: parts deactivated at the start of this stage (every
        part is in the model from stage 1: the computing model part only ever grows). k0: soil parts given the K0
        stresses. bc: a dict for Stage (see make_stage_class)."""
        i = len(self.stages) + 1
        mats = {"properties": [{"model_part_name": f"PorousDomain.{p}", "properties_id": self.pid[p],
                                "Material": {"constitutive_law": {"name": law}, "Variables": V}, "Tables": {}}
                               for p, (law, V) in materials.items()]}
        json.dump(mats, open(os.path.join(self.dir, f"MaterialParameters_stage{i}.json"), "w"), indent=1)
        procs = []
        for p in inactive:
            procs.append({"python_module": "apply_excavation_process", "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
                          "process_name": "ApplyExcavationProcess",
                          "Parameters": {"model_part_name": f"PorousDomain.{p}", "variable_name": "EXCAVATION", "deactivate_soil_part": True}})
        for p in active:
            if self.stages and p not in self.stages[-1]["active"]:
                procs.append({"python_module": "apply_excavation_process", "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
                              "process_name": "ApplyExcavationProcess",
                              "Parameters": {"model_part_name": f"PorousDomain.{p}", "variable_name": "EXCAVATION", "deactivate_soil_part": False}})
        for p in (k0 or []):
            procs.append({"python_module": "apply_k0_procedure_process", "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
                          "process_name": "ApplyK0ProcedureProcess",
                          "Parameters": {"model_part_name": f"PorousDomain.{p}", "use_standard_procedure": True}})
        procs += list(extra_processes)
        has_shell = any(self.parts[p][0] == SHELL3 for p in self.parts)
        pp = {"problem_data": {"problem_name": name, "start_time": 0.0, "end_time": 1.0, "echo_level": 1,
                               "parallel_type": "OpenMP", "number_of_threads": int(os.environ.get("OMP_NUM_THREADS", 2))},
              "solver_settings": {"solver_type": "U_Pw", "model_part_name": "PorousDomain", "domain_size": 3,
                                  "model_import_settings": {"input_type": "mdpa", "input_filename": self.mesh},
                                  "material_import_settings": {"materials_filename": f"MaterialParameters_stage{i}.json"},
                                  "time_stepping": {"time_step": 1.0 / steps, "max_delta_time_factor": 1.0},
                                  "buffer_size": 2, "echo_level": 1, "clear_storage": False, "compute_reactions": True,
                                  "move_mesh_flag": False, "reform_dofs_at_each_step": False,
                                  "block_builder": True, "solution_type": "quasi_static", "scheme_type": "Backward_Euler",
                                  "reset_displacements": reset, "strategy_type": "newton_raphson",
                                  "convergence_criterion": "residual_criterion",
                                  "residual_relative_tolerance": rel_tol, "residual_absolute_tolerance": abs_tol,
                                  "min_iterations": int(os.environ.get("GEO_MIN_IT", 6)), "max_iterations": max_iter, "number_cycles": 6,
                                  # speed-up (opt-in): GEO_INC = 2 lets a cut step grow again after a step that converges in
                                  # fewer than GEO_MIN_IT iterations; the default 1.0 keeps a cut step small for the rest of the stage
                                  "reduction_factor": 0.5, "increase_factor": float(os.environ.get("GEO_INC", 1.0)), "rotation_dofs": has_shell,
                                  "linear_solver_settings": LINEAR_SOLVERS[os.environ.get("GEO_LS", "ilu")],
                                  "problem_domain_sub_model_part_list": list(self.parts),
                                  "processes_sub_model_part_list": list(self.groups),
                                  "body_domain_sub_model_part_list": list(self.parts)},
              "output_processes": {}, "processes": {"constraints_process_list": [], "loads_process_list": [],
                                                    "auxiliary_process_list": procs}}
        json.dump(pp, open(os.path.join(self.dir, f"ProjectParameters_stage{i}.json"), "w"), indent=1)
        self.stages.append(dict(name=name, active=list(active), bc=bc or {}, file=f"ProjectParameters_stage{i}.json"))


def make_stage_class():
    import KratosMultiphysics as KM
    import KratosMultiphysics.GeoMechanicsApplication as KG
    import KratosMultiphysics.StructuralMechanicsApplication as SM
    from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import GeoMechanicsAnalysis

    class Stage(GeoMechanicsAnalysis):
        """A GeoMechanicsAnalysis stage with its boundary conditions set per node:
        bc = {"fix": {group: (fx, fy, fz)}        roller / fixed displacements (zero)
              "gravity": True,                      VOLUME_ACCELERATION on every node
              "loads": (node_idx, F[n, 3]),         POINT_LOAD, ramped with the pseudo-time
              "prescribed": (node_idx, U[n, 3]),    DISPLACEMENT of the stage, ramped
              "rot_free": node_idx,                 nodes whose rotations are free (the shells'); others fixed
              "on_step": callable(stage)}           called after every converged step (may end the stage)"""

        def __init__(self, model, parameters, bc, groups):
            super().__init__(model, parameters); self.bc = bc; self.groups = groups; self.history = []

        def ApplyBoundaryConditions(self):
            super().ApplyBoundaryConditions()
            mp = self._GetSolver().GetComputingModelPart(); t = mp.ProcessInfo[KM.TIME]
            nodes = self._nodes
            if not hasattr(self, "_fixed"):
                self._fixed = True
                for nd in mp.Nodes:
                    nd.Fix(KM.WATER_PRESSURE); nd.SetSolutionStepValue(KM.WATER_PRESSURE, 0.0)
                    if self.bc.get("gravity", True):
                        nd.SetSolutionStepValue(KM.VOLUME_ACCELERATION, [0.0, 0.0, -G])
                        nd.SetSolutionStepValue(KM.VOLUME_ACCELERATION, 1, [0.0, 0.0, -G])
                for g, fx in self.bc.get("fix", {}).items():
                    for i in self.groups[g]:
                        nd = nodes[i]
                        for on, var in zip(fx, (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z)):
                            if on:
                                nd.Fix(var); nd.SetSolutionStepValue(var, 0.0)
                if mp.HasNodalSolutionStepVariable(KM.ROTATION):
                    free = set(np.asarray(self.bc.get("rot_free", []), int).tolist())
                    for i, nd in nodes.items():
                        for var in (KM.ROTATION_X, KM.ROTATION_Y, KM.ROTATION_Z):
                            if i in free:
                                nd.Free(var)
                            else:
                                nd.Fix(var); nd.SetSolutionStepValue(var, 0.0)
                for i in self.bc.get("free", []):              # nodes held in earlier stages, released now
                    for var in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z):
                        nodes[i].Free(var)
                if "prescribed" in self.bc:
                    for i in self.bc["prescribed"][0]:
                        for var in (KM.DISPLACEMENT_X, KM.DISPLACEMENT_Y, KM.DISPLACEMENT_Z):
                            nodes[i].Fix(var)
            f = self.bc.get("scale_loads", lambda t: t)(t)
            if "loads" in self.bc:
                idx, F = self.bc["loads"]; F0 = self.bc.get("loads_from")      # loads_from: ramp from the previous stage's values
                if F0 is None:
                    for i, Fi in zip(idx, F):
                        nodes[i].SetSolutionStepValue(SM.POINT_LOAD, list(f * Fi))
                else:
                    for i, Fi, F0i in zip(idx, F, F0):
                        nodes[i].SetSolutionStepValue(SM.POINT_LOAD, list(F0i + f * (Fi - F0i)))
            f = self.bc.get("scale_prescribed", lambda t: t)(t)
            if "prescribed" in self.bc:
                idx, U = self.bc["prescribed"]
                for i, Ui in zip(idx, U):
                    nodes[i].SetSolutionStepValue(KM.DISPLACEMENT, list(f * Ui))

        def ModifyInitialGeometry(self):
            super().ModifyInitialGeometry()
            # the point-load conditions into the computing model part (CheckAndPrepareModelProcess leaves them out)
            main = self._GetSolver().main_model_part; cm = self._GetSolver().GetComputingModelPart()
            for g in self.bc.get("load_groups", []):
                cm.AddConditions([c.Id for c in main.GetSubModelPart(g).Conditions])
            for g in self.bc.get("off_groups", []):                   # point loads of earlier stages switched off
                for c in main.GetSubModelPart(g).Conditions:
                    c.Set(KM.ACTIVE, False)

        def Initialize(self):
            super().Initialize()
            mp = self._GetSolver().GetComputingModelPart()
            self._nodes = {nd.Id - 1: nd for nd in self._GetSolver().main_model_part.Nodes}

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()
            if "on_step" in self.bc:
                self.bc["on_step"](self)

        def stop(self):
            """End the stage after this step."""
            self.end_time = self._GetSolver().GetComputingModelPart().ProcessInfo[KM.TIME]

    return Stage


def run(project, only=None, after_stage=None):
    import KratosMultiphysics as KM
    import KratosMultiphysics.GeoMechanicsApplication  # noqa
    import KratosMultiphysics.StructuralMechanicsApplication  # noqa
    import KratosMultiphysics.LinearSolversApplication  # noqa
    Stage = make_stage_class()
    model = KM.Model(); cwd = os.getcwd(); os.chdir(project.dir); out = []
    try:
        for k, st in enumerate(project.stages):
            prm = KM.Parameters(open(st["file"]).read())
            s = Stage(model, prm, st["bc"], project.groups); s.Run(); out.append(s)
            if after_stage:
                after_stage(k, s, model)
    finally:
        os.chdir(cwd)
    return model, out
