"""
VSLAM-LAB capability: flat-port refraction removal with Refrax (entry point of github.com/VSLAM-LAB/refrax, a fork of
github.com/cllim118/Refrax).

Removes flat-port refraction from the rgb_0 frames of the given VSLAM-LAB sequence folders: every frame is re-rendered
as the image an in-air pinhole camera would have taken of a fronto-parallel scene plane at depth --z0, using the
physical housing model (port normal, glass/water refractive indices, port distance and glass thickness) and the
camera's own intrinsics. The ray tracing and the forward map are Refrax's (core/optics.py, core/undistort.py); this
file adds what VSLAM-LAB needs on top:

  1. The correction zoom (isotropic scale about the principal point). --zoom auto (default) is the unconstrained
     optimum over --zoom-bounds (scipy bounded minimisation of the RMSE between the zoomed forward map and the
     identity, i.e. the zoom that keeps the corrected image at the original magnification); --zoom in-bounds is
     Refrax's own choice (core/scale.py: the best sampled zoom whose map keeps every valid pixel inside the source
     W x H); --zoom <float> skips the search. The sampled zoom -> RMSE curve is written to zoom_sweep.csv.
  2. Canvas fitting (fit_canvas, default on): the output canvas is the bounding box of the valid corrected region at
     z0 instead of the source W x H, and the principal point moves by the canvas origin, so nothing the zoomed map
     pushes outside the source frame is lost. --no-fit-canvas keeps the source W x H (Refrax's behaviour; pair it
     with --zoom in-bounds). With cropping (crop_valid_bbox / --crop) every output is then cropped to the largest
     all-valid rectangle and mask.png is all ones; with --no-crop the full canvas is kept and mask.png (1 = valid
     pixel, 0 = outside, the mask2former convention) records the invalid border. mask.png is shared by every frame.

Camera parameters come from the sequence's calibration.yaml (rgb_0 entry: focal_length, principal_point,
image_dimension and, for radtan models, k1/k2/p1/p2 - lens undistortion is applied on top of the refraction map);
--calibration-yaml points at another yaml relative to the sequence folder (e.g. anycalib/calibration.yaml). Refrax
needs the camera's IN-AIR intrinsics, which is what VSLAM-LAB calibration.yaml files hold (--intrinsics in-air,
the default); for a camera calibrated underwater pass --intrinsics in-water (focal lengths divided by mu_w first).
Housing parameters, z0, fit_canvas and cropping default to configs/vslamlab.yaml (housing: and correction: blocks),
another config can be given with --housing-yaml, and every value can be overridden per flag.

Output is a per-sequence artifact: <sequence>/refrax_0/<frame stem>.png (3-channel PNG, one per rgb_0 frame),
mask.png, zoom_sweep.csv, calibration.yaml and the .refrax_complete marker. calibration.yaml is the sequence's
calibration.yaml with the rgb_0 entry replaced by the corrected camera (cam_model pinhole, focal_length scaled by
the zoom, principal_point shifted by the canvas origin and the crop offset, image_dimension of the written PNGs,
distortion_type radtan4 with all-zero distortion_coefficients) - every other field and entry kept verbatim; the
marker records every parameter of the run as 'key: value' lines. Frames whose PNG exists are skipped (an
interrupted run resumes); a sequence whose marker exists is skipped unless --overwrite. Neither rgb.csv nor
calibration.yaml of the sequence is modified. Frames are taken from rgb_raw.csv when it exists (the full list before
a VSLAM-LAB sampling step). Only rgb_0 is corrected; fisheye/equidistant models are not supported, and radtan5's k3
is dropped with a warning.

VSLAM-LAB (Capabilities/refrax.py) resolves dataset/sequence targets into --sequence-path folders and runs this
script through the `inference` pixi task; Run/run_functions.py swaps the artifact in when an experiment sets
'refraction: refrax'.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))  # Refrax's core/ and remove_refraction.py

from core.optics import apply_radtan_distortion, matrix_K  # noqa: E402
from core.scale import find_in_bounds_scale, sweep_zoom  # noqa: E402
from core.undistort import compute_housing_geometry, forward_map  # noqa: E402
from remove_refraction import compute_valid_mask, largest_valid_rectangle  # noqa: E402

FOLDER_BASE = "refrax"
COMPLETE_MARKER = ".refrax_complete"
MASK_FILE = "mask.png"
CALIBRATION_FILE = "calibration.yaml"
ZOOM_SWEEP_FILE = "zoom_sweep.csv"

# Housing/correction defaults come from configs/vslamlab.yaml; the constants below are only the fallback when that
# file is missing.
DEFAULT_HOUSING_YAML = REPO_ROOT / "configs" / "vslamlab.yaml"
DEFAULT_HOUSING = {"n_port": [0.0, 0.0, 1.0], "mu_a": 1.0, "mu_g": 1.47, "mu_w": 1.34, "rflat": 0.02, "tglass": 0.002}
DEFAULT_Z0 = 1.0
DEFAULT_ZOOM_BOUNDS = (1.0, 2.5)  # wider than core/scale.py's (1.3, 1.55): a wide-FOV camera optimum can sit above 1.55
ZOOM_SAMPLES = 80
INTRINSICS_MODES = ("in-water", "in-air")
SUPPORTED_DISTORTION = {None, "none", "radtan4", "radtan5"}

SCRIPT_LABEL = "\033[95m[vslamlab_refrax.py]\033[0m "


def print_info(msg: str) -> None:
    print(f"{SCRIPT_LABEL}{msg}", flush=True)


def print_warning(msg: str) -> None:
    print(f"{SCRIPT_LABEL}\033[93m[warning]\033[0m {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Correction: zoom search and a canvas-fitting corrector on Refrax's forward map
# ---------------------------------------------------------------------------------------------------------------------

def find_zoom(camera: dict, housing: dict, z0: float, bounds: tuple[float, float]) -> dict:
    """The correction zoom at depth z0: unconstrained optimum ('zoom'), Refrax's in-bounds choice over the sampled
    curve ('zoom_in_bounds', None if no sample qualifies) and the curve itself (s_values, rmse_values)."""
    W, H, fx, fy, cx, cy = (camera[k] for k in ("W", "H", "fx", "fy", "cx", "cy"))
    _, K_inv = matrix_K(fx, fy, cx, cy)
    P2, ray_water = compute_housing_geometry(H, W, K_inv, **housing_kwargs(housing))
    map_x, map_y = forward_map(P2, ray_water, z0, fx, fy, cx, cy)
    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dx0, dy0 = map_x - cx, map_y - cy

    def cost(s):
        per_pixel = np.sqrt((dx0 * s + cx - u_grid) ** 2 + (dy0 * s + cy - v_grid) ** 2)
        return np.sqrt(np.nanmean(per_pixel ** 2))

    result = minimize_scalar(cost, bounds=bounds, method="bounded")
    s_values = np.linspace(bounds[0], bounds[1], ZOOM_SAMPLES)
    rmse_values = sweep_zoom(map_x, map_y, cx, cy, s_values)
    zoom_in_bounds, rmse_in_bounds = find_in_bounds_scale(map_x, map_y, s_values, rmse_values, W, H, cx, cy)
    return {"zoom": float(result.x), "rmse": float(result.fun),
            "zoom_in_bounds": None if zoom_in_bounds is None else float(zoom_in_bounds),
            "rmse_in_bounds": float(rmse_in_bounds), "s_values": s_values, "rmse_values": rmse_values}


def housing_kwargs(housing: dict) -> dict:
    return dict(n_port=np.asarray(housing["n_port"], dtype=float), rflat=housing["rflat"], tglass=housing["tglass"],
                mu_a=housing["mu_a"], mu_g=housing["mu_g"], mu_w=housing["mu_w"])


def zoom_forward_map(map_x, map_y, cx, cy, zoom):
    """Isotropic scale of a forward map about the principal point, in float64 whatever zoom's type (a NumPy float64
    zoom would otherwise promote the float32 map while a Python float keeps it float32)."""
    return ((map_x.astype(np.float64) - cx) * float(zoom) + cx,
            (map_y.astype(np.float64) - cy) * float(zoom) + cy)


def forward_map_canvas(map_x, map_y):
    """Smallest integer canvas holding every finite point of a forward map: (W, H, origin), origin = (x0, y0) being
    the corrected-plane coordinate of the canvas' pixel (0, 0)."""
    finite = np.isfinite(map_x) & np.isfinite(map_y)
    if not np.any(finite):
        raise ValueError("forward map has no finite entries - cannot size a canvas")
    x_min, x_max = np.floor(map_x[finite].min()), np.ceil(map_x[finite].max())
    y_min, y_max = np.floor(map_y[finite].min()), np.ceil(map_y[finite].max())
    return int(x_max - x_min) + 1, int(y_max - y_min) + 1, (float(x_min), float(y_min))


def invert_map(map_x, map_y, H, W, step=4, origin=(0.0, 0.0)):
    """Invert a forward map (source pixel -> corrected-plane coordinate) onto an H x W output canvas whose pixel
    (0, 0) sits at corrected-plane coordinate `origin`. Returns, for every canvas pixel, the source pixel to sample
    (NaN outside the forward map's hull). Unlike core/undistort.invert_map, the last source row/column are always
    interpolation sites (so the hull reaches the source border) and non-finite forward-map points are dropped."""
    src_h, src_w = map_x.shape
    u_s = np.unique(np.append(np.arange(0, src_w, step), src_w - 1))
    v_s = np.unique(np.append(np.arange(0, src_h, step), src_h - 1))
    ug_s, vg_s = np.meshgrid(u_s, v_s)

    src = np.stack([map_x[np.ix_(v_s, u_s)].ravel(), map_y[np.ix_(v_s, u_s)].ravel()], axis=1)
    finite = np.all(np.isfinite(src), axis=1)
    src, ug_flat, vg_flat = src[finite], ug_s.ravel()[finite], vg_s.ravel()[finite]

    x0, y0 = (float(v) for v in origin)
    ug_full, vg_full = np.meshgrid(np.arange(W, dtype=np.float32) + x0, np.arange(H, dtype=np.float32) + y0)
    undist_x = griddata(src, ug_flat.astype(np.float32), (ug_full, vg_full), method="linear")
    undist_y = griddata(src, vg_flat.astype(np.float32), (ug_full, vg_full), method="linear")
    return undist_x.astype(np.float32), undist_y.astype(np.float32)


class RefractionCorrector:
    """Fixed-depth refraction removal for one camera/housing, reused across frames. camera: W, H, fx, fy, cx, cy
    (in-air pinhole) and optional k1, k2, p1, p2; housing: n_port, mu_a, mu_g, mu_w, rflat, tglass. With fit_canvas
    the canvas (out_W x out_H, pixel (0, 0) at `origin` in the source pixel frame) holds the whole corrected image;
    otherwise it is the source W x H. crop_box (y0, y1, x0, x1) is the largest all-valid rectangle when cropping."""

    def __init__(self, camera, housing, z0, zoom, crop_valid_bbox=False, fit_canvas=True):
        self.W, self.H = int(camera["W"]), int(camera["H"])
        self.fx, self.fy = float(camera["fx"]), float(camera["fy"])
        self.cx, self.cy = float(camera["cx"]), float(camera["cy"])
        self.distortion = tuple(float(camera.get(k, 0.0) or 0.0) for k in ("k1", "k2", "p1", "p2"))
        self.z0, self.zoom = float(z0), float(zoom)
        self.fit_canvas = bool(fit_canvas)

        _, K_inv = matrix_K(self.fx, self.fy, self.cx, self.cy)
        self.P2, self.ray_water = compute_housing_geometry(self.H, self.W, K_inv, **housing_kwargs(housing))

        self.out_W, self.out_H, self.origin = self.W, self.H, (0.0, 0.0)
        if self.fit_canvas:
            self.out_W, self.out_H, self.origin = self._estimate_canvas()

        self.undist_x, self.undist_y = self._build_map()
        self.mask = compute_valid_mask(self.undist_x, self.undist_y, self.W, self.H)
        if self.fit_canvas:
            self._tighten_canvas()
        self.crop_box = tuple(int(v) for v in largest_valid_rectangle(self.mask)) if crop_valid_bbox else None

    def _estimate_canvas(self):
        """Canvas holding every finite point of the zoomed forward map at z0, plus one pixel of slack on every side
        (the interpolation hull can reach fractionally beyond the sampled points; _tighten_canvas trims it)."""
        map_x, map_y = forward_map(self.P2, self.ray_water, self.z0, self.fx, self.fy, self.cx, self.cy)
        out_w, out_h, origin = forward_map_canvas(*zoom_forward_map(map_x, map_y, self.cx, self.cy, self.zoom))
        out_w, out_h, origin = out_w + 2, out_h + 2, (origin[0] - 1.0, origin[1] - 1.0)
        if out_w > 8 * self.W or out_h > 8 * self.H:
            raise ValueError(f"fit_canvas: corrected image would be {out_w}x{out_h} for a {self.W}x{self.H} source "
                             f"- check the housing parameters / zoom ({self.zoom})")
        return out_w, out_h, origin

    def _tighten_canvas(self):
        """Shrink the canvas to the bounding box of the valid pixels of the fixed-depth map."""
        rows, cols = np.flatnonzero(self.mask.any(axis=1)), np.flatnonzero(self.mask.any(axis=0))
        if len(rows) == 0 or len(cols) == 0:
            raise ValueError("fit_canvas: the fixed-depth map has no valid pixel")
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        if (y0, y1, x0, x1) == (0, self.out_H, 0, self.out_W):
            return
        self.undist_x, self.undist_y = self.undist_x[y0:y1, x0:x1], self.undist_y[y0:y1, x0:x1]
        self.mask = self.mask[y0:y1, x0:x1]
        self.origin = (self.origin[0] + float(x0), self.origin[1] + float(y0))
        self.out_W, self.out_H = int(x1 - x0), int(y1 - y0)

    def _build_map(self):
        """Undistortion map (canvas pixel -> source pixel) at z0, lens distortion applied, NaN -> -1 (invalid)."""
        map_x, map_y = forward_map(self.P2, self.ray_water, self.z0, self.fx, self.fy, self.cx, self.cy)
        map_x, map_y = zoom_forward_map(map_x, map_y, self.cx, self.cy, self.zoom)
        ux, uy = invert_map(map_x, map_y, self.out_H, self.out_W, origin=self.origin)
        if any(d != 0.0 for d in self.distortion):
            x_dist, y_dist = apply_radtan_distortion((ux - self.cx) / self.fx, (uy - self.cy) / self.fy, *self.distortion)
            ux, uy = x_dist * self.fx + self.cx, y_dist * self.fy + self.cy
        return np.nan_to_num(ux, nan=-1).astype(np.float32), np.nan_to_num(uy, nan=-1).astype(np.float32)

    def correct(self, img):
        """The corrected (and, with cropping, cropped) image."""
        return self.crop(cv2.remap(img, self.undist_x, self.undist_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0))

    def crop(self, array):
        if self.crop_box is None:
            return array
        y0, y1, x0, x1 = self.crop_box
        return array[y0:y1, x0:x1]

    def corrected_camera(self):
        """Pinhole intrinsics + size of the output images: the source principal point moved by the canvas origin,
        then by the crop offset; focal lengths scaled by the zoom."""
        cx, cy = self.cx - self.origin[0], self.cy - self.origin[1]
        W, H = self.out_W, self.out_H
        if self.crop_box is not None:
            y0, y1, x0, x1 = self.crop_box
            cx, cy, W, H = cx - x0, cy - y0, x1 - x0, y1 - y0
        return {"fx": self.fx * self.zoom, "fy": self.fy * self.zoom, "cx": cx, "cy": cy, "W": W, "H": H}


# ---------------------------------------------------------------------------------------------------------------------
# VSLAM-LAB sequence handling
# ---------------------------------------------------------------------------------------------------------------------

def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return (rows[0], rows[1:]) if rows else ([], [])


def load_camera(label: str, calib_yaml: Path, intrinsics: str, mu_w: float):
    """(camera dict, calibration file lines, source description) from the rgb_0 entry of calib_yaml, or None (with a
    warning) if the entry is unusable."""
    if not calib_yaml.exists():
        print_warning(f"Skipping {label} - missing {calib_yaml}")
        return None
    lines = calib_yaml.read_text().splitlines()
    cameras = {cam.get("cam_name"): cam for cam in (yaml.safe_load("\n".join(lines)) or {}).get("cameras", [])}
    cam = cameras.get("rgb_0")
    if cam is None:
        print_warning(f"Skipping {label} - {calib_yaml.name} has no rgb_0 camera entry")
        return None

    fx, fy = (float(v) for v in cam.get("focal_length", [0.0, 0.0]))
    cx, cy = (float(v) for v in cam.get("principal_point", [0.0, 0.0]))
    if cam.get("cam_model") != "pinhole" or fx <= 0 or fy <= 0:
        print_warning(f"Skipping {label} - rgb_0 in {calib_yaml.name} is not a calibrated pinhole camera "
                      f"(cam_model={cam.get('cam_model')}, focal_length={cam.get('focal_length')}); calibrate it first "
                      f"(e.g. VSLAM-LAB's 'pixi run calib-inference' and --calibration-yaml anycalib/calibration.yaml)")
        return None
    dist_type = cam.get("distortion_type")
    if dist_type not in SUPPORTED_DISTORTION:
        print_warning(f"Skipping {label} - distortion_type '{dist_type}' not supported (only radtan/none)")
        return None
    W, H = (int(v) for v in cam["image_dimension"])

    camera = {"W": W, "H": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    coeffs = [float(v) for v in cam.get("distortion_coefficients", [])] if dist_type in ("radtan4", "radtan5") else []
    if coeffs:
        camera.update({"k1": coeffs[0], "k2": coeffs[1], "p1": coeffs[2], "p2": coeffs[3]})
        if len(coeffs) > 4 and coeffs[4] != 0.0:
            print_warning(f"{label} - radtan5 k3={coeffs[4]} ignored (Refrax models k1, k2, p1, p2)")
    if intrinsics == "in-water":
        camera["fx"], camera["fy"] = fx / mu_w, fy / mu_w
    source = (f"{calib_yaml.name} rgb_0 ({intrinsics}): f=({fx:.2f}, {fy:.2f}) c=({cx:.2f}, {cy:.2f})"
              + (f" radtan={coeffs[:4]}" if coeffs else ""))
    return camera, lines, source


def choose_zoom(label: str, camera: dict, housing: dict, z0: float, zoom_spec: str,
                bounds: tuple[float, float], sweep_csv: Path) -> tuple[float, str, float | None]:
    """(zoom, mode, rmse_px) per --zoom: a number is used as is; 'auto'/'in-bounds' run the zoom search and write its
    zoom -> RMSE curve to sweep_csv."""
    try:
        return float(zoom_spec), "fixed", None
    except ValueError:
        pass
    result = find_zoom(camera, housing, z0, bounds)
    with open(sweep_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["zoom", "rmse_px"])
        writer.writerows(zip((f"{s:.6f}" for s in result["s_values"]), (f"{r:.6f}" for r in result["rmse_values"])))

    if zoom_spec == "in-bounds":
        if result["zoom_in_bounds"] is None:
            print_warning(f"{label} - no zoom in {bounds} keeps the corrected image inside the frame; using the "
                          f"unconstrained optimum instead")
            zoom, rmse = result["zoom"], result["rmse"]
        else:
            zoom, rmse = result["zoom_in_bounds"], result["rmse_in_bounds"]
    else:
        zoom, rmse = result["zoom"], result["rmse"]
    span = bounds[1] - bounds[0]
    if min(zoom - bounds[0], bounds[1] - zoom) < 0.02 * span:
        print_warning(f"{label} - zoom {zoom:.4f} sits at the edge of --zoom-bounds {bounds}; widen the bounds to check "
                      f"it is a true optimum")
    return zoom, zoom_spec, rmse


def fmt_list(values) -> str:
    return "[" + ", ".join(f"{float(v):.6f}".rstrip("0").rstrip(".") or "0" for v in values) + "]"


def patch_calibration_lines(lines: list[str], corrected: dict, comment: str) -> bool:
    """Replace the rgb_0 entry's camera model with the corrected pinhole camera: cam_model, focal_length,
    principal_point and image_dimension rewritten (annotated), distortion_type forced to radtan4 with all-zero
    distortion_coefficients (explicit, for consumers that branch on those fields being present). Line-based, so the
    hand-formatted flow style and every other entry survive. Returns False if there is no rgb_0 entry."""
    try:
        start = next(i for i, line in enumerate(lines) if "cam_name: rgb_0," in line or "cam_name: rgb_0}" in line)
        end = next(i for i in range(start, len(lines)) if lines[i].strip() == "}")
    except StopIteration:
        return False

    def find(key: str) -> int | None:
        return next((i for i in range(start, end + 1) if lines[i].lstrip().startswith(f"{key}:")), None)

    def prefix(idx: int | None) -> str:
        return lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())] if idx is not None else "     "

    replacements = {
        "cam_model": "pinhole",
        "focal_length": fmt_list([corrected["fx"], corrected["fy"]]),
        "principal_point": fmt_list([corrected["cx"], corrected["cy"]]),
        "image_dimension": f"[{corrected['W']}, {corrected['H']}]",
        "distortion_type": "radtan4",
        "distortion_coefficients": fmt_list([0.0, 0.0, 0.0, 0.0]),
    }
    anchors = {
        "cam_model": "cam_type",
        "focal_length": "cam_model",
        "principal_point": "focal_length",
        "image_dimension": "principal_point",
        "distortion_type": "image_dimension",
        "distortion_coefficients": "distortion_type",
    }
    for key, value in replacements.items():
        idx = find(key)
        comment_text = comment if key == "focal_length" else ("zero (corrected)" if key.startswith("distortion") else "corrected")
        new_line = f"{prefix(idx)}{key}: {value}, # refrax: {comment_text}"
        if idx is not None:
            lines[idx] = new_line
        else:
            anchor = find(anchors[key])
            lines.insert((anchor if anchor is not None else start) + 1, new_line)
            end += 1
    return True


def correct_sequence(seq_path: Path, *, calibration_rel: str, intrinsics: str, housing: dict, z0: float,
                     zoom_spec: str, zoom_bounds: tuple[float, float], crop: bool, fit_canvas: bool, folder_base: str,
                     overwrite: bool, housing_source: str) -> None:
    label = f"{seq_path.parent.name}/{seq_path.name}"
    rgb_csv, rgb_raw = seq_path / "rgb.csv", seq_path / "rgb_raw.csv"
    # The pre-sampling backup holds the full frame list, so the artifact covers every frame after a sampling step.
    source_csv = rgb_raw if rgb_raw.exists() else rgb_csv
    if not source_csv.exists():
        print_warning(f"Skipping {label} - missing rgb.csv")
        return

    out_dir = seq_path / f"{folder_base}_0"
    marker = out_dir / COMPLETE_MARKER
    if marker.exists() and not overwrite:
        print_info(f"Skipping {label} - {out_dir.name} already complete (use --overwrite to recompute)")
        return
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    header, rows = read_csv_rows(source_csv)
    if "path_rgb_0" not in header or not rows:
        print_warning(f"Skipping {label} - {source_csv.name} has no 'path_rgb_0' column or no frames")
        return
    if "path_rgb_1" in header:
        print_warning(f"{label} - only rgb_0 is corrected; the rgb_1 stream is left untouched")

    loaded = load_camera(label, seq_path / calibration_rel, intrinsics, housing["mu_w"])
    if loaded is None:
        return
    camera, calib_lines, source = loaded

    path_idx = header.index("path_rgb_0")
    missing = [row[path_idx] for row in rows if not (seq_path / row[path_idx]).exists()]
    if missing:
        print_warning(f"Skipping {label} - {len(missing)}/{len(rows)} frames listed in {source_csv.name} are missing on "
                      f"disk (broken symlinks / unmounted drive?), e.g. {missing[0]}")
        return
    first = cv2.imread(str(seq_path / rows[0][path_idx]))
    if first is None:
        print_warning(f"Skipping {label} - cannot read {rows[0][path_idx]}")
        return
    if first.shape[:2] != (camera["H"], camera["W"]):
        print_warning(f"Skipping {label} - image_dimension {[camera['W'], camera['H']]} in calibration does not match "
                      f"the frames ({first.shape[1]}x{first.shape[0]})")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    zoom, zoom_mode, zoom_rmse = choose_zoom(label, camera, housing, z0, zoom_spec, zoom_bounds, out_dir / ZOOM_SWEEP_FILE)
    if fit_canvas and zoom_mode == "in-bounds":
        print_info(f"{label} - --zoom in-bounds with the canvas fitted to the corrected image: the in-bounds constraint "
                   f"only matters with --no-fit-canvas; 'auto' is the natural choice here")
    corrector = RefractionCorrector(camera, housing, z0=z0, zoom=zoom, crop_valid_bbox=crop, fit_canvas=fit_canvas)
    corrected = corrector.corrected_camera()
    canvas = (f"canvas {corrector.out_W}x{corrector.out_H} at ({corrector.origin[0]:g}, {corrector.origin[1]:g}) of the "
              f"source frame" if fit_canvas else f"canvas {camera['W']}x{camera['H']} (source size)")
    print_info(f"{label} - {source}; in-air f=({camera['fx']:.2f}, {camera['fy']:.2f}); zoom={zoom:.4f} ({zoom_mode}"
               f"{'' if zoom_rmse is None else f', rmse={zoom_rmse:.2f} px'}), z0={z0:g} m; {canvas}; corrected "
               f"f=({corrected['fx']:.2f}, {corrected['fy']:.2f}) c=({corrected['cx']:.2f}, {corrected['cy']:.2f}) "
               f"{corrected['W']}x{corrected['H']}" + (f" (crop {list(corrector.crop_box)})" if crop else ""))

    # mask.png in the mask2former convention (1 = usable pixel), at the output geometry.
    cv2.imwrite(str(out_dir / MASK_FILE), corrector.crop((corrector.mask > 0).astype(np.uint8)))

    written = skipped = 0
    for k, row in enumerate(rows):
        src = seq_path / row[path_idx]
        out_png = out_dir / f"{src.stem}.png"
        if out_png.exists():  # resume: never recompute an existing frame
            skipped += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(f"Could not read {src}")
        cv2.imwrite(str(out_png), corrector.correct(img))
        written += 1
        if (k + 1) % 100 == 0 or k + 1 == len(rows):
            print_info(f"{label} - {k + 1}/{len(rows)} frames")

    stamp = datetime.date.today().isoformat()
    comment = f"zoom {zoom:.4f} x in-air focal ({camera['fx']:.2f}, {camera['fy']:.2f}), z0={z0:g} m"
    if not patch_calibration_lines(calib_lines, corrected, comment):
        print_warning(f"{label} - could not patch the rgb_0 entry; no calibration.yaml written")
        return
    header_idx = next((i for i, line in enumerate(calib_lines) if line.strip() == "---"), -1) + 1
    calib_lines[header_idx:header_idx] = [
        f"# rgb_0: refrax-corrected ({stamp}) - frames in {out_dir.name}/ are re-rendered as an in-air pinhole camera",
        f"# (flat-port refraction removed, z0={z0:g} m, zoom={zoom:.4f}"
        f"{', canvas fitted to the whole corrected image' if fit_canvas else ', clipped to the source frame'}"
        f"{', cropped to the valid region' if crop else ''});",
        f"# cam_model/focal_length/principal_point/image_dimension replaced accordingly, distortion_type forced to radtan4 "
        f"with distortion_coefficients all zero. Source: {source}.",
        "# Every other field is the sequence's original calibration.",
    ]
    (out_dir / CALIBRATION_FILE).write_text("\n".join(calib_lines).rstrip("\n") + "\n")

    marker.write_text("".join(f"{key}: {value}\n" for key, value in [
        ("zoom", f"{zoom:.6f}"), ("zoom_mode", zoom_mode),
        ("zoom_rmse_px", "" if zoom_rmse is None else f"{zoom_rmse:.4f}"),
        ("zoom_bounds", fmt_list(zoom_bounds)), ("z0", f"{z0:g}"),
        ("fit_canvas", "true" if fit_canvas else "false"),
        ("canvas", f"[{corrector.out_W}, {corrector.out_H}]"), ("canvas_origin", fmt_list(corrector.origin)),
        ("crop", "true" if crop else "false"), ("crop_box", list(corrector.crop_box) if corrector.crop_box else "null"),
        ("image_dimension", f"[{corrected['W']}, {corrected['H']}]"),
        ("focal_length", fmt_list([corrected["fx"], corrected["fy"]])),
        ("principal_point", fmt_list([corrected["cx"], corrected["cy"]])),
        ("intrinsics", intrinsics), ("calibration_source", calibration_rel), ("housing_source", housing_source),
        ("source_focal_length_in_air", fmt_list([camera["fx"], camera["fy"]])),
        ("housing_n_port", fmt_list(housing["n_port"])), ("housing_mu_a", housing["mu_a"]), ("housing_mu_g", housing["mu_g"]),
        ("housing_mu_w", housing["mu_w"]), ("housing_rflat", housing["rflat"]), ("housing_tglass", housing["tglass"]),
    ]))
    print_info(f"{label} - wrote {written} corrected frames to {out_dir}"
               + (f" ({skipped} already existed)" if skipped else "") + f", {MASK_FILE}, {CALIBRATION_FILE}")


def resolve_housing(args: argparse.Namespace) -> dict:
    """Housing/correction settings: built-in fallbacks < the housing yaml (--housing-yaml, default configs/vslamlab.yaml:
    housing:, correction: z0_fixed / fit_canvas / crop_valid_bbox) < explicit flags."""
    housing = dict(DEFAULT_HOUSING)
    settings = {"z0": DEFAULT_Z0, "crop": False, "fit_canvas": True, "source": "built-in defaults"}
    housing_yaml = Path(args.housing_yaml) if args.housing_yaml else DEFAULT_HOUSING_YAML
    if housing_yaml.exists():
        cfg = yaml.safe_load(housing_yaml.read_text()) or {}
        housing.update({k: v for k, v in (cfg.get("housing") or {}).items() if k in housing})
        corr = cfg.get("correction") or {}
        settings["z0"] = corr.get("z0_fixed", settings["z0"])
        settings["crop"] = bool(corr.get("crop_valid_bbox", settings["crop"]))
        settings["fit_canvas"] = bool(corr.get("fit_canvas", settings["fit_canvas"]))
        settings["source"] = str(housing_yaml)
    elif args.housing_yaml:
        print_warning(f"--housing-yaml {housing_yaml} not found")
        sys.exit(1)
    for key in ("mu_a", "mu_g", "mu_w", "rflat", "tglass"):
        if getattr(args, key) is not None:
            housing[key] = getattr(args, key)
    if args.n_port is not None:
        housing["n_port"] = list(args.n_port)
    if args.z0 is not None:
        settings["z0"] = args.z0
    if args.crop is not None:
        settings["crop"] = args.crop
    if args.fit_canvas is not None:
        settings["fit_canvas"] = args.fit_canvas
    housing["n_port"] = [float(v) for v in housing["n_port"]]
    settings["z0"] = float(settings["z0"])
    settings["housing"] = housing
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove flat-port refraction from the rgb_0 frames of VSLAM-LAB sequences.")
    parser.add_argument("--sequence-path", nargs="+", type=Path, dest="sequence_paths", help="VSLAM-LAB sequence folders")
    parser.add_argument("--calibration-yaml", default=CALIBRATION_FILE, dest="calibration_yaml",
                        help="Calibration yaml to read rgb_0 from, relative to the sequence folder (default: "
                             "calibration.yaml; e.g. anycalib/calibration.yaml)")
    parser.add_argument("--intrinsics", choices=INTRINSICS_MODES, default="in-air",
                        help="What the calibration's focal lengths are: the camera's in-air calibration (default), or an "
                             "underwater (flat-port magnified) calibration, divided by mu_w to get the in-air focal")
    parser.add_argument("--housing-yaml", default=None, dest="housing_yaml",
                        help="Config yaml whose housing: and correction: (z0_fixed, fit_canvas, crop_valid_bbox) blocks "
                             "set the defaults (default: configs/vslamlab.yaml)")
    parser.add_argument("--n-port", type=float, nargs=3, default=None, dest="n_port", metavar=("NX", "NY", "NZ"),
                        help=f"Flat-port normal in the camera frame (default: {DEFAULT_HOUSING['n_port']})")
    parser.add_argument("--mu-a", type=float, default=None, dest="mu_a", help="Refractive index of air")
    parser.add_argument("--mu-g", type=float, default=None, dest="mu_g", help="Refractive index of the port glass")
    parser.add_argument("--mu-w", type=float, default=None, dest="mu_w", help="Refractive index of water")
    parser.add_argument("--rflat", type=float, default=None, help="Camera centre to port distance (m)")
    parser.add_argument("--tglass", type=float, default=None, help="Port glass thickness (m)")
    parser.add_argument("--z0", type=float, default=None, help="Scene depth (m) the correction map is built for")
    parser.add_argument("--zoom", default="auto",
                        help="Correction zoom: 'auto' (unconstrained optimum, default), 'in-bounds' (best zoom keeping "
                             "the whole valid image inside the source frame) or a number")
    parser.add_argument("--zoom-bounds", type=float, nargs=2, default=DEFAULT_ZOOM_BOUNDS, dest="zoom_bounds",
                        metavar=("MIN", "MAX"), help=f"Zoom search interval (default: {list(DEFAULT_ZOOM_BOUNDS)})")
    parser.add_argument("--fit-canvas", action="store_true", default=None, dest="fit_canvas",
                        help="Size the output canvas to the whole corrected image, shifting the principal point")
    parser.add_argument("--no-fit-canvas", action="store_false", dest="fit_canvas",
                        help="Keep the source W x H and clip what the zoomed map pushes outside it (pair with --zoom in-bounds)")
    parser.add_argument("--crop", action="store_true", default=None, dest="crop",
                        help="Crop every output to the largest all-valid rectangle")
    parser.add_argument("--no-crop", action="store_false", dest="crop",
                        help="Keep the full canvas (black outside the valid region, recorded in mask.png)")
    parser.add_argument("--folder-base", default=FOLDER_BASE, dest="folder_base",
                        help="Output folder prefix; frames are written to <base>_0 (default: refrax)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if the artifact already exists")
    parser.add_argument("--prefetch", action="store_true", help="Check the Refrax modules import and exit (no weights)")
    args = parser.parse_args()

    if args.prefetch:
        print_info(f"Refrax importable from {REPO_ROOT}")
        return
    if not args.sequence_paths:
        parser.error("--sequence-path is required")
    if args.zoom not in ("auto", "in-bounds"):
        try:
            float(args.zoom)
        except ValueError:
            parser.error(f"--zoom must be 'auto', 'in-bounds' or a number (got '{args.zoom}')")

    settings = resolve_housing(args)
    print_info(f"housing/correction defaults from {settings['source']}: mu_w={settings['housing']['mu_w']}, "
               f"rflat={settings['housing']['rflat']}, tglass={settings['housing']['tglass']}, z0={settings['z0']:g} m, "
               f"fit_canvas={settings['fit_canvas']}, crop={settings['crop']}")
    for seq_path in args.sequence_paths:
        correct_sequence(
            seq_path.resolve(), calibration_rel=args.calibration_yaml, intrinsics=args.intrinsics,
            housing=settings["housing"], z0=settings["z0"], zoom_spec=args.zoom, zoom_bounds=tuple(args.zoom_bounds),
            crop=settings["crop"], fit_canvas=settings["fit_canvas"], folder_base=args.folder_base,
            overwrite=args.overwrite, housing_source=settings["source"],
        )


if __name__ == "__main__":
    main()
