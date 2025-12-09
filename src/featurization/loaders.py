import os, io, gzip
from typing import List, Tuple, Optional
import numpy as np
from .features import line_to_vector

def _open_any(path: str):
    """Open plain text or .gz transparently."""
    if path.lower().endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def read_vectors_from_log(
    path: str,
    max_lines: Optional[int] = None,
    bad_out: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    """
    Reads a log file, converts to feature vectors, skips unparsable lines.
    Returns: (X, skipped_count)
      - X: np.ndarray [n, d]
      - skipped_count: number of lines we ignored
    If bad_out is provided, writes unparsable raw lines to that file.
    """
    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    bad_f = open(bad_out, "w", encoding="utf-8") if bad_out else None
    vecs: List[np.ndarray] = []
    skipped = 0
    seen = 0

    with _open_any(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            seen += 1
            v, _ = line_to_vector(ln)
            if v is None:
                skipped += 1
                if bad_f: bad_f.write(ln.rstrip("\n") + "\n")
            else:
                vecs.append(v)
            if max_lines and seen >= max_lines:
                break

    if bad_f:
        bad_f.close()

    if not vecs:
        raise SystemExit(f"No usable lines found in {path} (skipped={skipped}).")

    X = np.stack(vecs, axis=0)
    return X, skipped
