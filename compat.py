"""
NumPy Compatibility Shim — Arogentis
=======================================
NumPy 2.0 removed many legacy aliases. MNE, Numba, and scipy still
reference them internally, causing AttributeError crashes.

This shim patches all known removals back before any MNE imports.
MUST be imported before any MNE / scipy / numba imports.
"""

import warnings
import numpy as np

# ─── Function aliases removed in NumPy 2.0 ────────────────────────────────────
_FUNC_PATCHES = {
    "trapz":       "trapezoid",
    "in1d":        "isin",
    "row_stack":   "vstack",
    "cumproduct":  "cumprod",
    "sometrue":    "any",
    "alltrue":     "all",
    "product":     "prod",
}

for _old, _new in _FUNC_PATCHES.items():
    if not hasattr(np, _old) and hasattr(np, _new):
        setattr(np, _old, getattr(np, _new))

# ─── Scalar type aliases removed in NumPy 2.0 ─────────────────────────────────
_TYPE_PATCHES = {
    "bool":    np.bool_,
    "int":     np.intp,
    "float":   np.float64,
    "complex": np.complex128,
    "object":  np.object_,
}

for _attr, _replacement in _TYPE_PATCHES.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (FutureWarning, DeprecationWarning))
        if not hasattr(np, _attr):
            setattr(np, _attr, _replacement)
