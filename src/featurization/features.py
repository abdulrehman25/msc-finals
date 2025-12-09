def line_to_vector(line: str):
    """
    Parse a log line and return:
      - vector: np.ndarray (feature vector) or None if not parsable
      - meta: dict with lightweight fields (ip, ua, method, path, status, size)
    This version is NULL-SAFE: missing method/path/status/size won't crash.
    """
    import numpy as np
    from .parse import parse_line

    rec = parse_line(line)
    if not rec:
        return None, {}

    # ---- NULL-SAFE NORMALIZATION ----
    method = rec.get("method") or "UNK"
    try:
        method_up = method.upper()
    except Exception:
        method_up = "UNK"

    path = rec.get("path") or "/"
    status = rec.get("status")
    try:
        status_i = int(status) if status not in (None, "", "-") else 0
    except Exception:
        status_i = 0

    size = rec.get("size")
    try:
        size_i = int(size) if size not in (None, "", "-") else 0
    except Exception:
        size_i = 0

    ua = rec.get("ua") or "-"
    ip = rec.get("ip") or "-"

    # ---- SIMPLE FEATUREIZATION (keep whatever you already had) ----
    # Example mapping; keep your real METHOD_MAP etc.
    METHOD_MAP = {"GET":1, "POST":2, "PUT":3, "DELETE":4, "HEAD":5, "OPTIONS":6, "TRACE":7, "UNK":0}
    method_code = METHOD_MAP.get(method_up, 0)

    # Very simple path features; replace with your tokenizer if you already have one
    path_len = len(path)
    is_api = 1 if path.startswith("/api/") else 0
    has_query = 1 if "?" in path else 0
    dot_ext = 1 if "." in path.split("?")[0] else 0

    # Status buckets (one-hot-ish)
    s2xx = 1 if 200 <= status_i < 300 else 0
    s3xx = 1 if 300 <= status_i < 400 else 0
    s4xx = 1 if 400 <= status_i < 500 else 0
    s5xx = 1 if 500 <= status_i < 600 else 0

    # Size (clipped/logged)
    size_log = np.log1p(max(size_i, 0))

    # Build vector (align with your model input_dim)
    vec = np.array([
        method_code,
        path_len, is_api, has_query, dot_ext,
        status_i, s2xx, s3xx, s4xx, s5xx,
        size_log,
    ], dtype=np.float32)

    meta = {"ip": ip, "ua": ua, "method": method_up, "path": path, "status": status_i, "size": size_i}
    return vec, meta
