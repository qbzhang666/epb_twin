# Third-party software

epb_twin is distributed under the BSD 3-Clause licence (LICENSE). It does not include or redistribute the packages
below; they are installed by the user (pip) or, for the container images, pulled at build time.

| Package | Used for | Licence | Required |
|---|---|---|---|
| numpy | arrays | BSD-3-Clause (and compatible) | always |
| pandas | record tables | BSD-3-Clause | always |
| PyYAML | project, adapter, machine and ground files | MIT | always |
| openpyxl | .xlsx records | MIT | `[excel]` |
| Kratos Multiphysics, GeoMechanics, StructuralMechanics, LinearSolvers applications | finite element advance model | BSD-4-Clause | `[fe]` |
| gmsh | meshes of the advance model | GPL-2.0-or-later | `[fe]` |
| anthropic | Claude agent back-end | MIT | `[anthropic]` |

gmsh is GPL-licensed. epb_twin calls it through its Python API only when meshes are made (`epb_twin.fe.advance
mesh`); the record tools, the load model, the agent layer and the benchmark do not import it. A container image that
includes gmsh is a redistribution of gmsh and carries its licence; the images built from `containers/` install it
from the Python Package Index at build time.
