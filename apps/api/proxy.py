"""Cheap point-cloud proxies for the viewer's distant-view layer.

A full Gaussian-splat PLY is ~64 MB / ~1.17 M vertices. The viewer only needs
a few thousand of those points (centres + colour) to draw a lightweight
stand-in for every scene it isn't currently standing inside. Downloading the
whole PLY just to throw 99 % of it away is what made the viewer take ~45 s to
show anything, so we pre-decimate each PLY into a tiny binary blob once and
cache it next to the PLY.

Blob format (little-endian), read directly by a DataView in the browser:

    magic   : 4 bytes  b"MPX1"
    count   : uint32
    for each point:
        x, y, z : float32   (12 bytes)
        r, g, b : uint8      (3 bytes)

So a 12 k-point proxy is 8 + 12000*15 ≈ 180 KB instead of 64 MB.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

MAGIC = b"MPX1"
SH_C0 = 0.28209479177387814
_TYPE_NP = {
    "float": "<f4", "double": "<f8",
    "int": "<i4", "uint": "<u4",
    "short": "<i2", "ushort": "<u2",
    "char": "<i1", "uchar": "<u1",
    "int8": "<i1", "uint8": "<u1",
    "int16": "<i2", "uint16": "<u2",
    "int32": "<i4", "uint32": "<u4",
    "float32": "<f4", "float64": "<f8",
}
_TYPE_BYTES = {k: int(v[-1]) for k, v in _TYPE_NP.items()}


def _parse_header(f) -> tuple[int, int, dict]:
    """Return (vertex_count, stride, {name: (offset, np_dtype)}) for the vertex
    element. Reads only the ASCII header, then leaves f positioned at the start
    of the binary body."""
    raw = f.read(8192)
    end = raw.find(b"end_header")
    if end == -1:
        raise ValueError("no end_header in first 8 KB")
    body_start = end + len(b"end_header")
    while body_start < len(raw) and raw[body_start] in (13, 10):  # \r \n
        body_start += 1
    f.seek(body_start)

    header = raw[:end].decode("ascii", "replace")
    count = 0
    stride = 0
    fields: dict[str, tuple[int, str]] = {}
    in_vertex = False
    for line in header.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                count = int(parts[2])
            continue
        if in_vertex and parts[0] == "property" and len(parts) == 3:
            _, ptype, name = parts
            fields[name] = (stride, _TYPE_NP.get(ptype, "<f4"))
            stride += _TYPE_BYTES.get(ptype, 4)
    return count, stride, fields


def build_proxy_bytes(ply_path: Path, max_points: int = 12_000) -> bytes:
    """Decimate a splat PLY to at most max_points and return the proxy blob."""
    with open(ply_path, "rb") as f:
        total, stride, fields = _parse_header(f)
        if total == 0 or not all(k in fields for k in ("x", "y", "z")):
            raise ValueError("PLY missing vertex positions")
        body = np.fromfile(f, dtype=np.uint8, count=total * stride)

    body = body.reshape(total, stride)
    step = max(1, -(-total // max_points))  # ceil
    sel = body[::step]
    n = sel.shape[0]

    def column(name: str) -> np.ndarray:
        off, dt = fields[name]
        width = int(dt[-1])
        return sel[:, off:off + width].copy().view(dt).reshape(n)

    xyz = np.empty((n, 3), dtype="<f4")
    xyz[:, 0] = column("x")
    xyz[:, 1] = column("y")
    xyz[:, 2] = column("z")

    rgb = np.empty((n, 3), dtype=np.uint8)
    if all(f"f_dc_{i}" in fields for i in range(3)):
        for i in range(3):
            # Standard 3DGS spherical-harmonics DC term -> base RGB:
            #   colour = 0.5 + SH_C0 * f_dc
            # The previous `f_dc / SH_C0 * 0.5 + 0.5` scaled the coefficient by
            # ~1.77 instead of ~0.28 (≈6.3x too hot), so nearly every channel
            # clipped to 0 or 1 — that's what made the proxy points look
            # blown-out and over-saturated next to the (correct) splats.
            c = column(f"f_dc_{i}").astype("<f4")
            c = np.clip(SH_C0 * c + 0.5, 0.0, 1.0) * 255.0
            rgb[:, i] = c.astype(np.uint8)
    elif all(k in fields for k in ("red", "green", "blue")):
        rgb[:, 0] = column("red")
        rgb[:, 1] = column("green")
        rgb[:, 2] = column("blue")
    else:
        rgb[:] = (77, 153, 255)

    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", n)
    out += xyz.tobytes()
    out += rgb.tobytes()
    return bytes(out)


# ── Decimated full-res splat ────────────────────────────────────────────────
# The full-quality PLY is ~1.17 M gaussians, and the viewer library spends
# ~2.6 s building an octree over it every time it's swapped in — the wall
# behind the "10 s to turn a point cloud into a splat" complaint. Keeping only
# a fraction of the gaussians shrinks the download AND the octree build roughly
# proportionally, so the on-approach swap lands far sooner. We keep every
# step-th vertex (uniform decimation) and leave any trailing camera-metadata
# elements untouched, producing a valid, smaller gaussian PLY.


def build_decimated_ply(ply_path: Path, keep: float) -> bytes:
    keep = max(0.02, min(1.0, keep))
    with open(ply_path, "rb") as f:
        raw = f.read(8192)
        end = raw.find(b"end_header")
        if end == -1:
            raise ValueError("no end_header in first 8 KB")
        body_start = end + len(b"end_header")
        while body_start < len(raw) and raw[body_start] in (13, 10):
            body_start += 1
        header_text = raw[:body_start].decode("ascii", "replace")

        f.seek(0)
        count, stride, fields = _parse_header(f)  # re-reads header, seeks to body_start
        f.seek(body_start)
        vertex_block = np.frombuffer(f.read(count * stride), dtype=np.uint8)
        trailing = f.read()  # other elements (extrinsics/intrinsics/…), if any

    step = max(1, round(1.0 / keep))
    if step == 1:
        return ply_path.read_bytes()

    rows = vertex_block.reshape(count, stride)[::step]
    new_count = rows.shape[0]
    new_header = re.sub(r"element vertex \d+", f"element vertex {new_count}", header_text, count=1)
    return new_header.encode("ascii") + rows.tobytes() + trailing


def lite_ply_path_for(ply_path: Path, keep: float) -> Path:
    return ply_path.with_suffix(f".k{int(round(max(0.02, min(1.0, keep)) * 100)):02d}.ply")


def get_or_build_lite_ply(ply_path: Path, keep: float = 0.4) -> Path:
    """Return a path to a cached decimated PLY (keep ≈ fraction of gaussians),
    building it if missing/stale. keep >= 1 just returns the original PLY."""
    if keep >= 0.999:
        return ply_path
    out = lite_ply_path_for(ply_path, keep)
    if out.exists() and out.stat().st_mtime >= ply_path.stat().st_mtime:
        return out
    data = build_decimated_ply(ply_path, keep)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    return out
