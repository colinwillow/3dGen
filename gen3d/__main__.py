"""3dGen command line.

    python -m gen3d make  photo.png --faces 5000            # image -> quad FBX
    python -m gen3d make  front.png --left l.png --back b.png
    python -m gen3d retopo some_highpoly.glb --faces 5000   # any mesh -> quad FBX
    python -m gen3d tripo photo.png --faces 5000            # same image through Tripo's API

Everything lands in out/<name>/.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def blender():
    exe = os.environ.get("BLENDER") or shutil.which("blender")
    if not exe:
        sys.exit("Blender not found: set BLENDER=/path/to/blender (setup/install_wsl.sh does)")
    return exe


def retopo(high, out, name, a):
    cmd = [blender(), "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(HERE / "retopo.py"), "--",
           "--high", str(high), "--out", str(out), "--name", name, "--faces", str(a.faces),
           "--tex", str(a.tex), "--voxel", str(a.voxel)]
    if a.tris:
        cmd.append("--tris")
    if a.symmetry:
        cmd.append("--symmetry")
    if a.glb:
        cmd.append("--glb")
    if getattr(a, "clean", False):
        cmd += ["--mode", "clean"]
    print("[gen3d]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return Path(out) / f"{name}.fbx"


def mesh_opts(ap):
    ap.add_argument("--faces", type=int, default=5000, help="target face count (default 5000)")
    ap.add_argument("--tris", action="store_true", help="triangles instead of quads")
    ap.add_argument("--tex", type=int, default=2048, help="baked texture size")
    ap.add_argument("--voxel", type=float, default=0.006,
                    help="pre-remesh detail; smaller keeps thin parts, bigger is more robust")
    ap.add_argument("--symmetry", action="store_true", help="mirror the quad flow across X")
    ap.add_argument("--glb", action="store_true", help="also write a triangulated GLB")
    ap.add_argument("--name", help="output name (default: the input file's name)")
    ap.add_argument("--out", default="out", help="output root folder")


def main():
    ap = argparse.ArgumentParser(prog="gen3d")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="image -> textured quad FBX (Hunyuan3D-2 + Blender)")
    m.add_argument("image", help="front view")
    m.add_argument("--left"), m.add_argument("--back"), m.add_argument("--right")
    m.add_argument("--no-texture", action="store_true")
    m.add_argument("--turbo", action="store_true", help="faster, slightly worse shape")
    m.add_argument("--seed", type=int, default=1234)
    m.add_argument("--octree", type=int, default=384, help="shape resolution (256-512)")
    mesh_opts(m)

    r = sub.add_parser("retopo", help="any high-poly mesh -> quad FBX with baked textures")
    r.add_argument("mesh")
    mesh_opts(r)

    t = sub.add_parser("tripo", help="the same image through Tripo's API (needs TRIPO_API_KEY)")
    t.add_argument("image")
    t.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="extra Tripo task field, e.g. --set model_version=v3.0-20250812")
    mesh_opts(t)

    a = ap.parse_args()
    src = Path(a.image if a.cmd in ("make", "tripo") else a.mesh)
    name = a.name or src.stem
    out = Path(a.out) / name
    out.mkdir(parents=True, exist_ok=True)

    if a.cmd == "make":
        from gen3d.shape import generate
        high = generate(a.image, out / "high", left=a.left, back=a.back, right=a.right,
                        texture=not a.no_texture, seed=a.seed, octree=a.octree, turbo=a.turbo)
        fbx = retopo(high, out, name, a)
    elif a.cmd == "retopo":
        fbx = retopo(src, out, name, a)
    else:
        from gen3d.tripo import run
        extra = dict(kv.split("=", 1) for kv in a.set)
        got = run(a.image, out / "tripo", faces=a.faces, quad=not a.tris, extra=extra)
        a.clean = True  # Tripo already did the topology; just ground it and re-export
        fbx = retopo(got, out, name + "_tripo", a)
    print(f"\n[gen3d] done -> {fbx}")


if __name__ == "__main__":
    main()
