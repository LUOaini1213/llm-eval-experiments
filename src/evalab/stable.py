"""Round floats before writing result files, so reruns on different platforms give byte-identical JSON.

The last one or two digits of a float can differ between Windows and Linux builds of numpy/scipy (seen in CI on
2026-09-28). Ten significant digits keeps every reported number unchanged and makes the files reproducible.
"""
import math

import numpy as np


def stable_round(x, sig: int = 10):
    if isinstance(x, dict):
        return {k: stable_round(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [stable_round(v, sig) for v in x]
    if isinstance(x, np.ndarray):
        return stable_round(x.tolist(), sig)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        return None if math.isnan(v) else float(f"{v:.{sig}g}")
    return x
