"""The same image through Tripo's paid API, for a side-by-side with the open path.

Needs TRIPO_API_KEY (platform.tripo3d.ai). The endpoint shapes follow Tripo's
v2 OpenAPI docs; the TASK FIELDS are the part to check against their current
docs if a request is refused -- they add options between model versions, and
`--set key=value` passes any field straight through without a code change.
"""
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

API = "https://api.tripo3d.ai/v2/openapi"


def _hdr():
    key = os.environ.get("TRIPO_API_KEY")
    if not key:
        raise SystemExit("set TRIPO_API_KEY first")
    return {"Authorization": f"Bearer {key}"}


def _ok(r):
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise SystemExit(f"Tripo refused: {body}")
    return body["data"]


def run(image, out_dir, faces=5000, quad=True, extra=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(image, "rb") as f:
        tok = _ok(requests.post(f"{API}/upload", headers=_hdr(), files={"file": f}))["image_token"]

    ext = Path(image).suffix.lstrip(".").lower().replace("jpeg", "jpg")
    task = {"type": "image_to_model", "file": {"type": ext, "file_token": tok},
            "texture": True, "pbr": True, "face_limit": faces, "quad": quad,
            "smart_low_poly": True}
    for k, v in (extra or {}).items():
        task[k] = {"true": True, "false": False}.get(v.lower(), v)
    print("[tripo] task", task)
    tid = _ok(requests.post(f"{API}/task", headers=_hdr(), json=task))["task_id"]

    while True:
        d = _ok(requests.get(f"{API}/task/{tid}", headers=_hdr()))
        print(f"[tripo] {d['status']} {d.get('progress', '')}%", flush=True)
        if d["status"] == "success":
            break
        if d["status"] in ("failed", "cancelled", "banned", "expired", "unknown"):
            raise SystemExit(f"Tripo task {d['status']}: {d}")
        time.sleep(4)

    o = d["output"]
    url = o.get("pbr_model") or o.get("model") or o.get("base_model")
    if not url:
        raise SystemExit(f"no model URL in {o}")
    path = out / ("tripo" + (Path(urlparse(url).path).suffix or ".glb"))
    path.write_bytes(requests.get(url, timeout=300).content)
    print("[tripo] ->", path)
    return path
