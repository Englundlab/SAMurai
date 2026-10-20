"""Periodic smooth curves for editing one instance, including holes/islands."""
import numpy as np
from scipy.interpolate import CubicSpline
from skimage.measure import find_contours
from skimage.draw import polygon
from shapely.geometry import LinearRing


def boundary_points(mask, spacing=10):
    rings = []
    for contour in find_contours(np.pad(mask.astype(float), 1), .5):
        contour = contour - 1
        if not np.allclose(contour[0], contour[-1]):
            contour = np.vstack([contour, contour[0]])
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(contour, axis=0), axis=1))]
        count = max(4, int(np.ceil(distance[-1] / spacing)))
        t = np.linspace(0, distance[-1], count, endpoint=False)
        rings.append(np.column_stack([np.interp(t, distance, contour[:, d]) for d in (0, 1)]))
    if not rings:
        raise ValueError("Select an existing object before editing its boundary.")
    return rings


def smooth_rings(points, sizes):
    if len(points) != sum(sizes) or not np.isfinite(points).all():
        raise ValueError("Move the existing points; use point spacing to change their number.")
    curves = []
    start = 0
    for size in sizes:
        ring = np.asarray(points[start:start+size], dtype=float)
        start += size
        ring = np.vstack([ring, ring[0]])
        lengths = np.linalg.norm(np.diff(ring, axis=0), axis=1)
        if np.any(lengths < .01):
            raise ValueError("Boundary points must not coincide.")
        t = np.r_[0, np.cumsum(lengths)]
        curve = CubicSpline(t, ring, bc_type="periodic")(np.linspace(0, t[-1], max(32, int(t[-1]*2)+1)))
        if not LinearRing(curve).is_simple:
            raise ValueError("Boundary crosses itself. Move the points apart before applying.")
        curves.append(curve)
    for i, a in enumerate(curves):
        for b in curves[i+1:]:
            if LinearRing(a).intersects(LinearRing(b)):
                raise ValueError("Separate boundary loops must not cross or touch.")
    return curves


def rasterize_boundary(curves, shape):
    mask = np.zeros(shape, dtype=bool)
    for curve in curves:
        rr, cc = polygon(curve[:, 0], curve[:, 1], shape)
        mask[rr, cc] ^= True
    return mask
