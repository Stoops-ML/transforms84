import os
from typing import Union

import numpy as np

DTYPES_SUPPORTED = Union[np.float32, np.float64, np.int8, np.int16, np.int32, np.int64]
commit_hash = os.getenv("COMMIT_HASH", "")
if commit_hash:  # pragma: no cover
    commit_hash = f"+{commit_hash}"
__version__ = f"0.2.0{commit_hash}"
