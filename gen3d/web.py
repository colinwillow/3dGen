"""The 3dGen web app:  python -m gen3d.web   ->   http://localhost:7860

Each job runs the ordinary command line (`python -m gen3d make|retopo`) as a
child process, one at a time. A separate process per job means every model
starts with the GPU's full 24 GB and gives it all back when it exits, and the
web page can never get out of step with what the command line does.
"""
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
JOBS = Path(os.environ.get("GEN3D_JOBS", ROOT / "jobs"))
JOBS.mkdir(parents=True, exist_ok=True)
IMG = {".png", ".jpg", ".jpeg", ".webp"}
MESH = {".glb", ".gltf", ".fbx", ".obj"}

# Log line -> the stage it starts. Matched in order, last match wins.
STAGES = [
    (r"\[shape\] tencent", "shape"),
    (r"\[shape\] \d+ faces", "texture"),
    (r"\[retopo\] high:", "quads"),
    (r"\[retopo\] clean:", "export"),
    (r"\[retopo\] baking on", "bake"),
    (r"\[retopo\] wrote .*\.fbx", "export"),
    (r"\[gen3d\] done", "done"),
]

app = FastAPI()
# three.js is vendored (r180) so the app works with no internet connection.
app.mount("/vendor", StaticFiles(directory=Path(__file__).parent / "web" / "vendor"), name="vendor")
work: "queue.Queue[str]" = queue.Queue()
lock = threading.Lock()


def meta_path(jid):
    return JOBS / jid / "job.json"


def load(jid):
    try:
        return json.loads(meta_path(jid).read_text())
    except (OSError, ValueError):
        return None


def save(m):
    with lock:
        meta_path(m["id"]).write_text(json.dumps(m, indent=1))


def worker():
    while True:
        jid = work.get()
        m = load(jid)
        if not m:
            continue
        m.update(status="running", stage="start", started=time.time())
        save(m)
        log = open(JOBS / jid / "log.txt", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1",
                   PYTHONPATH=f"{ROOT}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
        p = subprocess.Popen(m["cmd"], cwd=ROOT, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, env=env)
        for line in p.stdout:
            log.write(line)
            log.flush()
            for pat, stage in STAGES:
                if re.search(pat, line) and m["stage"] != stage:
                    m["stage"] = stage
                    save(m)
        p.wait()
        log.close()
        fbx = JOBS / jid / f"{jid}.fbx"
        m.update(status="done" if p.returncode == 0 and fbx.exists() else "failed",
                 finished=time.time())
        save(m)


def files(jid):
    d = JOBS / jid
    out = {}
    for name, key in ((f"{jid}.fbx", "fbx"), (f"{jid}.glb", "glb"), (f"{jid}_highres.glb", "highres"),
                      ("preview.glb", "preview"), ("wire.bin", "wire")):
        if (d / name).exists():
            out[key] = f"/files/{jid}/{name}"
    out["textures"] = [f"/files/{jid}/{p.name}" for p in sorted(d.glob("*.png"))]
    ins = sorted((d / "input").glob("*"))
    pics = [p for p in ins if p.suffix.lower() in IMG]
    if pics:
        out["thumb"] = f"/files/{jid}/input/{pics[0].name}"
    return out


def view(m):
    d = JOBS / m["id"]
    r = {k: m.get(k) for k in ("id", "name", "mode", "status", "stage", "params", "created",
                               "started", "finished")}
    r["files"] = files(m["id"])
    try:
        r["stats"] = json.loads((d / "stats.json").read_text())
    except (OSError, ValueError):
        r["stats"] = None
    try:
        r["log"] = (d / "log.txt").read_text()[-4000:]
    except OSError:
        r["log"] = ""
    r["queue"] = work.qsize()
    return r


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "web" / "index.html").read_text()


@app.get("/api/jobs")
def list_jobs():
    ms = [load(p.parent.name) for p in JOBS.glob("*/job.json")]
    ms = sorted((m for m in ms if m), key=lambda m: m["created"], reverse=True)
    return [view(m) for m in ms[:60]]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    m = load(jid) if re.fullmatch(r"[\w-]+", jid) else None
    if not m:
        raise HTTPException(404)
    return view(m)


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    m = load(jid) if re.fullmatch(r"[\w-]+", jid) else None
    if not m:
        raise HTTPException(404)
    if m["status"] in ("queued", "running"):
        raise HTTPException(409, "still running")
    shutil.rmtree(JOBS / jid)
    return {"ok": True}


@app.post("/api/jobs")
async def create_job(front: UploadFile = File(None), left: UploadFile = File(None),
                     back: UploadFile = File(None), right: UploadFile = File(None),
                     mesh: UploadFile = File(None),
                     faces: int = Form(5000), tris: bool = Form(False),
                     symmetry: bool = Form(False), tex: int = Form(2048),
                     turbo: bool = Form(False), texture: bool = Form(True),
                     retopo: bool = Form(False)):
    src = mesh if mesh and mesh.filename else front
    if not src or not src.filename:
        raise HTTPException(400, "upload a photo (or a mesh)")
    stem = re.sub(r"[^\w-]+", "_", Path(src.filename).stem)[:40] or "model"
    jid = time.strftime("%Y%m%d-%H%M%S-") + stem
    d = JOBS / jid / "input"
    d.mkdir(parents=True)

    async def keep(u, as_name, allowed):
        if not u or not u.filename:
            return None
        ext = Path(u.filename).suffix.lower()
        if ext not in allowed:
            raise HTTPException(400, f"{u.filename}: expected {', '.join(sorted(allowed))}")
        p = d / f"{as_name}{ext}"
        p.write_bytes(await u.read())
        return p

    faces = max(200, min(200000, faces))
    tex = tex if tex in (512, 1024, 2048, 4096) else 2048
    opts = ["--faces", str(faces), "--tex", str(tex), "--name", jid, "--out", str(JOBS)]
    opts += ["--tris"] * tris + ["--symmetry"] * symmetry
    if mesh and mesh.filename:
        p = await keep(mesh, "mesh", MESH)
        mode, cmd = "retopo", [sys.executable, "-m", "gen3d", "retopo", str(p)] + opts
    else:
        p = await keep(front, "front", IMG)
        cmd = [sys.executable, "-m", "gen3d", "make", str(p)] + opts
        for k, u in (("left", left), ("back", back), ("right", right)):
            q = await keep(u, k, IMG)
            if q:
                cmd += [f"--{k}", str(q)]
        cmd += ["--turbo"] * turbo + ["--no-texture"] * (not texture) + ["--retopo"] * retopo
        mode = "make" if retopo else "high"
    m = dict(id=jid, name=stem, mode=mode, status="queued", stage="queued", cmd=cmd,
             created=time.time(),
             params=dict(faces=faces, tris=tris, symmetry=symmetry, tex=tex, turbo=turbo,
                         retopo=retopo or mode == "retopo"))
    save(m)
    work.put(jid)
    return view(m)


@app.get("/files/{jid}/{path:path}")
def get_file(jid: str, path: str):
    base = (JOBS / jid).resolve()
    f = (base / path).resolve()
    if not re.fullmatch(r"[\w-]+", jid) or base not in f.parents or not f.is_file():
        raise HTTPException(404)
    name = f.name
    if f.suffix in (".fbx", ".glb") and f.stem.startswith(jid):
        # robot.fbx / robot_highres.glb, not 20251003-...fbx
        name = (load(jid) or {}).get("name", jid) + f.stem[len(jid):] + f.suffix
    return FileResponse(f, filename=name)


def resume():
    # Anything that was queued or mid-run when the app was closed: run it again.
    for p in sorted(JOBS.glob("*/job.json")):
        m = load(p.parent.name)
        if m and m["status"] in ("queued", "running"):
            m.update(status="queued", stage="queued")
            save(m)
            work.put(m["id"])


def main():
    import uvicorn
    resume()
    threading.Thread(target=worker, daemon=True).start()
    port = int(os.environ.get("GEN3D_PORT", 7860))
    print(f"\n  3dGen is running:  http://localhost:{port}\n")
    uvicorn.run(app, host=os.environ.get("GEN3D_HOST", "127.0.0.1"), port=port, log_level="warning")


if __name__ == "__main__":
    main()
