import os

commit_hash = os.getenv("COMMIT_HASH", "")
if commit_hash:  # pragma: no cover
    commit_hash = f"+{commit_hash}"
__version__ = f"0.2.0{commit_hash}"
