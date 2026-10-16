"""epb_twin: tested tools for ring-by-ring analysis of earth-pressure-balance (EPB) shield drives.

Layer 1: a standard per-ring record (schema.py), project adapters that map a raw record onto it
(adapter.py, adapters/*.yaml) and a normaliser that absorbs deviations of form (normalise.py).
Layer 2: machine and ground files (spec.py) read by the advance and load models."""
__version__ = "0.1.0"
from .schema import FIELDS, DERIVED, validate                     # noqa: F401
from .adapter import Adapter, load_adapter                        # noqa: F401
try:                                                              # pandas is needed only by the normaliser:
    from .normalise import normalise, to_number, range_spread     # noqa: F401  the specs and the FE model
except ImportError:                                               # run without it (e.g. an HPC Kratos venv)
    pass
from .spec import Machine, Ground, Material, load_machine, load_ground  # noqa: F401
