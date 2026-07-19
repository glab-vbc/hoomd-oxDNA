"""oxDNA coarse-grained DNA model as a HOOMD-blue plugin.

Two layers share this package:
  * the compiled C++ force engine, imported here as ``_engine``;
  * the pure-Python model layer (``io``, ``forces``, ``model``) that supplies
    geometry, parameters and simulation wiring.

oxDNA1 is the first supported variant (see :mod:`oxdna.model.dna1`).
"""

import hoomd  # ensure the HOOMD shared libraries are loaded before the plugin

from . import _engine  # noqa: F401  (compiled extension, installed alongside)
from . import io  # noqa: F401
from . import forces  # noqa: F401
from . import model  # noqa: F401
from .version import version  # noqa: F401

__all__ = ["io", "forces", "model", "version"]
