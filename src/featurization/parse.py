import re, json
from typing import Optional, Dict

# Apache Combined (IP ident authuser [date] "METHOD PATH HTTP/x" status size "ref" "ua")
RE_COMBINED = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<req>[^"]*)"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\S+)\s+'
    r'"(?P<ref>[^"]*)"\s+"(?P<ua>[^"]*)"\s*$'
)

# Common Log Format (no referrer/UA)
RE_COMMON = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+'
    r'"(?P<req>[^"]*)"\s+'
    r'(?P<status>\d{3})\s+(?P<size>\S+)\s*$'
)

# Nginx-like variant sometimes has "-" for size or HTTP/2, etc. (handled by patterns above)
# Basic JSON lines {"remote_addr": "...", "request": "GET /path HTTP/1.1", "status": 200, "body_bytes_sent": 123, "http_referer": "-", "http_user_agent": "..."}
JSON_KEYS = {
    "ip": ["remote_addr","host","client_ip","ip"],
    "ts": ["time_local","timestamp","time","@timestamp"],
    "req": ["request","req","http_request","method_path_proto"],
    "status": ["status","resp_status","sc_status"],
    "size": ["body_bytes_sent","bytes","size","resp_bytes","bytes_sent"],
    "ref": ["http_referer","referer","referrer","ref"],
    "ua":  ["http_user_agent","user_agent","ua"]
}

def _split_req(req: str):
    # req like: "GET /path HTTP/1.1" or "POST /x?y=1 HTTP/2"
    parts = req.split()
    method, path = None, None
    if len(parts) >= 2:
        method, path = parts[0], parts[1]
    return method, path

def _from_match(m: re.Match, has_refua: bool):
    gd = m.groupdict()
    method, path = _split_req(gd.get("req",""))
    size = gd.get("size","-")
    return {
        "ip": gd.get("ip"),
        "ts": gd.get("ts"),
        "method": method,
        "path": path,
        "status": gd.get("status"),
        "size": None if size in ("-","") else size,
        "ref": gd.get("ref") if has_refua else "-",
        "ua": gd.get("ua") if has_refua else "-"
    }

def _parse_json(line: str) -> Optional[Dict]:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    out = {}
    for k, aliases in JSON_KEYS.items():
        for a in aliases:
            if a in obj:
                out[k] = obj[a]
                break
    # request may be split across fields (method + uri)
    if "req" not in out:
        method = obj.get("method") or obj.get("http_method")
        uri = obj.get("uri") or obj.get("path") or obj.get("request_uri")
        proto = obj.get("protocol") or obj.get("http_version")
        if method and uri:
            out["req"] = f"{method} {uri} {proto or ''}".strip()
    method, path = _split_req(out.get("req",""))
    if not method and not path:
        return None
    size = out.get("size")
    if size in ("","-"): size = None
    status = out.get("status")
    return {
        "ip": out.get("ip"),
        "ts": out.get("ts"),
        "method": method,
        "path": path,
        "status": str(status) if status is not None else None,
        "size": size,
        "ref": out.get("ref","-"),
        "ua": out.get("ua","-"),
    }

def parse_line(line: str) -> Optional[Dict]:
    s = line.strip()
    if not s:
        return None
    # Try Combined
    m = RE_COMBINED.match(s)
    if m:
        return _from_match(m, has_refua=True)
    # Try Common
    m = RE_COMMON.match(s)
    if m:
        return _from_match(m, has_refua=False)
    # Try JSON
    j = _parse_json(s)
    if j:
        return j
    return None
