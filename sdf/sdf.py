import subprocess, sys, os

# Auto-install missing dependencies
def install(pkg):
    print(f"Installing {pkg} …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for pkg in ("numpy", "scipy", "matplotlib"):
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =============================================================================
# 1. RESOLVE INPUT CSV PATH
#    Raw boundary files are named <name>_raw.csv.
#    The ordered polygon will be saved as <name>.csv in the same directory.
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Shorthand aliases for input files
SHORTCUTS = {
    "iter": os.path.join(BASE_DIR, "ITER", "iter_raw.csv"),
    "iter.csv": os.path.join(BASE_DIR, "ITER", "iter_raw.csv"),
    "st40": os.path.join(BASE_DIR, "ST40", "st40_raw.csv"),
    "st40.csv": os.path.join(BASE_DIR, "ST40", "st40_raw.csv"),
    "mast": os.path.join(BASE_DIR, "MAST", "mast_raw.csv"),
    "mast.csv": os.path.join(BASE_DIR, "MAST", "mast_raw.csv"),
}

if len(sys.argv) > 1:
    arg = sys.argv[1]
    csv_path = SHORTCUTS.get(arg, os.path.abspath(arg))
else:
    options = [
        ("MAST/mast_raw.csv", SHORTCUTS["mast"]),
        ("ITER/iter_raw.csv", SHORTCUTS["iter"]),
        ("ST40/st40_raw.csv", SHORTCUTS["st40"]),
        ("Other path", None),
    ]
    print("Choose CSV to run:")
    for i, (name, path) in enumerate(options, 1):
        label = path if path else "(enter path)"
        print(f"  {i}) {name:25s}  {label}")
    try:
        choice = int(input("Select option [1-4]: ").strip() or "1")
    except Exception:
        choice = 1
    if 1 <= choice <= 4:
        csv_path = options[choice - 1][1]
    else:
        csv_path = os.path.abspath(input("Enter CSV path: ").strip())

if not os.path.isfile(csv_path):
    print(f"\nERROR: CSV not found:\n {csv_path}")
    sys.exit(1)

# Derive output paths
out_dir = os.path.dirname(csv_path)
raw_base = os.path.splitext(os.path.basename(csv_path))[0]
if raw_base.endswith("_raw"):
    out_name = raw_base[:-4]
else:
    out_name = raw_base        # fallback for filename

out_csv = os.path.join(out_dir, f"{out_name}.csv")
out_npz = os.path.join(out_dir, f"{out_name}.npz")
out_dat = os.path.join(out_dir, f"{out_name}.dat")
out_png = os.path.join(out_dir, f"{out_name}.png")

print(f"\nInput (raw)     : {csv_path}")
print(f"Ordered polygon : {out_csv}")
print(f"Output .npz     : {out_npz}")
print(f"Output .dat     : {out_dat}")
print(f"Output .png     : {out_png}")

# =============================================================================
# 2. LOAD BOUNDARY POINTS
#    CSV defines the vessel wall geometry as a point-to-point boundary profile.
#    Each row is an (x, y) coordinate; no header is assumed.
# =============================================================================

pts = np.loadtxt(csv_path, delimiter=",")
if pts.ndim != 2 or pts.shape[1] != 2:
    print(f"ERROR: expected N×2 CSV, got shape {pts.shape}")
    sys.exit(1)
print(f"\nLoaded {len(pts)} boundary points")
print(f"  x range: [{pts[:,0].min():.4f}, {pts[:,0].max():.4f}]")
print(f"  y range: [{pts[:,1].min():.4f}, {pts[:,1].max():.4f}]")

# =============================================================================
# 3. RECONSTRUCT POLYGON TRAVERSAL ORDER
#    CSV points are typically unordered or sampled column-by-column.
#    A greedy KDTree nearest-neighbour chain rebuilds a valid polygon walk
#    so that segment-by-segment distance and winding-number calculations
#    are geometrically correct.
# =============================================================================

print("\nReconstructing polygon order …")
remaining = set(range(len(pts)))
ordered = [0]
remaining.discard(0)
tree = KDTree(pts)
current = 0
k = min(50, len(pts))

for _ in range(len(pts) - 1):
    _, idxs = tree.query(pts[current], k = k)
    max_jump = 3.0 * np.median(np.linalg.norm(np.diff(pts, axis = 0), axis = 1))
    for idx in idxs:
        if idx in remaining:
            if np.linalg.norm(pts[idx] - pts[current]) < max_jump:
                ordered.append(idx)
                remaining.discard(idx)
                current = idx
                break

poly = pts[ordered]   # (N, 2) ordered polygon vertices
print(f"Polygon reconstructed: {len(poly)} vertices")

# Save the ordered polygon
np.savetxt(out_csv, poly, delimiter=",", fmt="%.8e")
print(f"Ordered polygon saved: {out_csv}")

# =============================================================================
# 4. BUILD CARTESIAN GRID
#    1600 cells along the longest axis for fine resolution.
#    10% padding ensures ghost cells near the domain boundary have valid SDF
#    values.  No domain restriction is applied — the full padded bounding box
#    is used so that no exterior region is masked out.
# =============================================================================

RESOLUTION = 1600
PADDING = 0.10

xmin, xmax = pts[:,0].min(), pts[:,0].max()
ymin, ymax = pts[:,1].min(), pts[:,1].max()
dx_data = xmax - xmin
dy_data = ymax - ymin
xmin -= PADDING * dx_data
xmax += PADDING * dx_data
ymin -= PADDING * dy_data
ymax += PADDING * dy_data

aspect = (xmax - xmin) / (ymax - ymin)
nx = int(RESOLUTION * max(1.0, aspect))
ny = int(RESOLUTION * max(1.0, 1.0 / aspect))
print(f"\nGrid: {ny} × {nx}  ({nx*ny:,} points)")

gx = np.linspace(xmin, xmax, nx)
gy = np.linspace(ymin, ymax, ny)
px_s = (xmax - xmin) / (nx - 1)
py_s = (ymax - ymin) / (ny - 1)

GX, GY = np.meshgrid(gx, gy)

# =============================================================================
# 5. ANALYTIC UNSIGNED DISTANCE
#
#    For each segment (x1, y1) → (x2, y2):
#      n = (x2 - x1, y2 - y1)                               edge vector
#      nSquared = n·n
#      w = ((x - x1) * n[0] + (y - y1) * n[1]) / nSquared   projection parameter
#
#    Three cases:
#      w in [0, 1]  → perpendicular foot is on the segment;
#                     offset q = (x - x1) - w * n, dist = |q|
#      w < 0        → closest point is the start vertex (x1, y1)
#      w > 1        → closest point is the end vertex   (x2, y2)
#
#    Running minimum over all segments gives the unsigned distance.
# =============================================================================

print("\nComputing analytic SDF…")
print("  Building edge distances (this may take ~30 s for large grids) …")

n_seg = len(poly)
dist  = np.full(GX.shape, np.inf)

qx_flat = GX.ravel()
qy_flat = GY.ravel()

for i in range(n_seg):
    x1, y1 = poly[i]
    x2, y2 = poly[(i + 1) % n_seg]

    nx_e = x2 - x1
    ny_e = y2 - y1
    nSquared = nx_e * nx_e + ny_e * ny_e

    if nSquared == 0.0:
        # Degenerate segment (point); distance to that point
        d = np.sqrt((qx_flat - x1)**2 + (qy_flat - y1)**2)
    else:
        w = ((qx_flat - x1) * nx_e + (qy_flat - y1) * ny_e) / nSquared

        # Case 1: perpendicular foot lies on segment
        on_seg  = (w >= 0.0) & (w <= 1.0)
        # offset from query point to foot
        foot_qx = (qx_flat - x1) - w * nx_e
        foot_qy = (qy_flat - y1) - w * ny_e
        d_perp  = np.sqrt(foot_qx**2 + foot_qy**2)

        # Case 2: w < 0  → distance to start vertex
        d_start = np.sqrt((qx_flat - x1)**2 + (qy_flat - y1)**2)

        # Case 3: w > 1  → distance to end vertex
        d_end = np.sqrt((qx_flat - x2)**2 + (qy_flat - y2)**2)

        d = np.where(on_seg, d_perp, np.where(w < 0.0, d_start, d_end))

    dist = np.minimum(dist, d.reshape(ny, nx))

    if i % max(1, n_seg // 10) == 0:
        print(f"    {i}/{n_seg} segments done …")

print(f"  Unsigned distance range: [{dist.min():.4f}, {dist.max():.4f}]")

# =============================================================================
# 6. RAY-CASTING SIGN
#
#    Ray travels leftward (to −∞) and counts crossings where xIntercept < x.  
#    Odd count → inside.
#
#    Returns +minDist inside, −minDist outside.
# =============================================================================

print("  Computing sign (ray-casting) …")

inside = np.zeros((ny, nx), dtype=bool)

for i in range(n_seg):
    x1, y1 = poly[i]
    x2, y2 = poly[(i + 1) % n_seg]

    if y1 == y2:
        continue   # horizontal edge

    # Straddle condition: (y1 < y && y <= y2) || (y2 < y && y <= y1)
    straddle = ((y1 < GY) & (GY <= y2)) | ((y2 < GY) & (GY <= y1))

    # x-coordinate of the edge at height GY
    with np.errstate(divide = "ignore", invalid = "ignore"):
        l          = (GY - y1) / (y2 - y1)
        xIntercept = x1 + l * (x2 - x1)

    # Ray travels left: count crossing only if xIntercept < x
    crosses = straddle & (xIntercept < GX)
    inside ^= crosses   # XOR toggles inside/outside for each crossing

# Inverted sign convention: negative interior, positive exterior
sign = np.where(inside, -1.0, 1.0)
sdf = sign * dist
print(f"  Signed distance range: [{sdf.min():.4f}, {sdf.max():.4f}]")

# =============================================================================
# 7. NARROW-BAND GAUSSIAN SMOOTHING
#    At sharp re-entrant corners the SDF gradient is discontinuous, which
#    causes the C++ central-difference normal computation to produce incorrect
#    wall normals. A Gaussian blur is applied only within a narrow band around
#    the zero level-set; outside this band raw analytic distances are kept.
#    The sign mask is re-applied after blending to preserve the zero crossing.
# =============================================================================

print("\nApplying narrow-band smoothing …")

SMOOTH_SIGMA = 4.0
SMOOTH_BAND_CELLS = 10
band_width = SMOOTH_BAND_CELLS * max(px_s, py_s)
sdf_smooth = gaussian_filter(sdf, sigma = SMOOTH_SIGMA)
in_band = np.abs(sdf) < band_width
sdf = np.where(in_band, sdf_smooth, sdf)
sdf = sign * np.abs(sdf)

SMOOTH_SIGMA_2 = 2.0
SMOOTH_BAND_CELLS_2 = 3
band_width_2 = SMOOTH_BAND_CELLS_2 * max(px_s, py_s)
sdf_smooth2 = gaussian_filter(sdf, sigma = SMOOTH_SIGMA_2)
in_band2 = np.abs(sdf) < band_width_2
sdf = np.where(in_band2, sdf_smooth2, sdf)
sdf = sign * np.abs(sdf)

print(f"  SDF range after smoothing: [{sdf.min():.4f}, {sdf.max():.4f}]")
print(f"  Band width: {band_width:.4f} (physical units)")

# =============================================================================
# 8. SAVE OUTPUTS
# =============================================================================

np.savez_compressed(out_npz, sdf=sdf, x=gx, y=gy, pts=pts)
print(f"\nSaved: {out_npz}")
print("  Keys: sdf (ny×nx), x (nx,), y (ny,), pts (N×2)")

print("\nWriting SDF .dat file …")
with open(out_dat, "w") as f:
    f.write("# x y sdf\n")
    for j in range(ny):
        for i in range(nx):
            f.write(f"{gx[i]:.8e} {gy[j]:.8e} {sdf[j, i]:.8e}\n")
        f.write("\n")
print(f"Saved: {out_dat}")

# =============================================================================
# 9. DIAGNOSTIC PLOT
# =============================================================================

print("\nGenerating plot …")
fig, axes = plt.subplots(1, 2, figsize = (12, 7), dpi = 150)

ax = axes[0]
vmax = np.percentile(np.abs(sdf), 98)
im = ax.imshow(
    sdf, origin = "lower",
    extent = [xmin, xmax, ymin, ymax],
    cmap = "viridis", vmin = -vmax, vmax = vmax,
    interpolation = "bilinear",
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size = "5%", pad = 0.05)
plt.colorbar(im, cax = cax, label = "φ (signed distance)")
ax.contour(GX, GY, sdf, levels = [0], colors = "red", linewidths = 2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")

gx_grad, gy_grad = np.gradient(sdf, px_s, py_s)
grad_mag = np.sqrt(gx_grad**2 + gy_grad**2)
ax2 = axes[1]
im2 = ax2.imshow(
    grad_mag, origin = "lower",
    extent = [xmin, xmax, ymin, ymax],
    cmap = "hot_r", vmin = 0, vmax = 2,
    interpolation = "bilinear",
)
divider2 = make_axes_locatable(ax2)
cax2     = divider2.append_axes("right", size = "5%", pad = 0.05)
plt.colorbar(im2, cax = cax2, label = "|∇φ| (gradient magnitude)")
ax2.contour(GX, GY, sdf, levels = [0], colors = "cyan", linewidths = 0.8)
ax2.set_xlabel("x");
ax2.set_ylabel("y")
ax2.set_aspect("equal")

plt.tight_layout()
plt.savefig(out_png, bbox_inches = "tight")
print(f"Saved: {out_png}")
print("\nDone.")