import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


st.set_page_config(page_title="2D GMM PDF Visualizer", layout="wide")


@dataclass
class Component:
    weight: float
    mean_x: float
    mean_y: float
    sigma_x: float
    sigma_y: float
    rho: float  # correlation in [-0.999, 0.999]

    def cov(self) -> np.ndarray:
        sx, sy, r = self.sigma_x, self.sigma_y, self.rho
        return np.array([[sx * sx, r * sx * sy], [r * sx * sy, sy * sy]], dtype=float)

    def mean(self) -> np.ndarray:
        return np.array([self.mean_x, self.mean_y], dtype=float)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    s = e / np.sum(e)
    return s


def gaussian_pdf_grid(mean: np.ndarray, cov: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    pos = np.dstack((X, Y))
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    diff = pos - mean
    expo = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
    return norm * np.exp(-0.5 * expo)


def mixture_pdf_grid(components: List[Component], X: np.ndarray, Y: np.ndarray, normalized_weights: bool) -> Tuple[np.ndarray, np.ndarray]:
    w = np.array([c.weight for c in components], dtype=float)
    if not normalized_weights:
        w = softmax(w)
    else:
        s = np.sum(w)
        w = w / s if s > 0 else softmax(w)
    Z = np.zeros_like(X, dtype=float)
    Z_parts = []
    for i, c in enumerate(components):
        Zi = gaussian_pdf_grid(c.mean(), c.cov(), X, Y)
        Z += w[i] * Zi
        Z_parts.append(Zi)
    return Z, np.array(Z_parts)


def is_pos_def_2x2(sx: float, sy: float, rho: float) -> bool:
    return sx > 0 and sy > 0 and abs(rho) < 1.0


def default_components(k: int) -> List[Component]:
    base = [
        Component(0.6, -1.5, -0.5, 1.0, 0.6, 0.2),
        Component(0.4, 2.0, 1.0, 0.8, 1.2, -0.3),
        Component(0.3, 0.0, -2.0, 0.7, 0.7, 0.0),
        Component(0.2, 3.0, -1.5, 1.4, 0.5, 0.4),
        Component(0.1, -3.0, 2.5, 0.9, 1.1, -0.5),
    ]
    return base[:k]


st.title("2D Gaussian Mixture Model — 3D PDF Surface Visualizer")

with st.sidebar:
    st.header("Mixture Settings")
    k = st.slider("# Components", min_value=1, max_value=5, value=2)
    x_min, x_max = st.slider("X range", value=(-6.0, 6.0), min_value=-20.0, max_value=20.0, step=0.5)
    y_min, y_max = st.slider("Y range", value=(-6.0, 6.0), min_value=-20.0, max_value=20.0, step=0.5)
    res = st.select_slider("Grid resolution", options=[50, 75, 100, 150, 200], value=100)

    weight_mode = st.radio(
        "Weight handling",
        options=["Auto-normalize (softmax)", "Use as-is (normalize sum)"]
    )
    normalized_weights = (weight_mode == "Use as-is (normalize sum)")

    load_cfg = st.file_uploader("Load config (JSON)", type=["json"], accept_multiple_files=False)

    st.markdown("**View Settings**")
    elev = st.slider("Elevation", min_value=0, max_value=90, value=30)
    azim = st.slider("Azimuth", min_value=-180, max_value=180, value=-60)
    log_density = st.checkbox("Log-density (ln)", value=False)
    draw_wire = st.checkbox("Wireframe overlay", value=False)
    stride = st.slider("Wireframe stride", min_value=1, max_value=12, value=6)
    draw_projections = st.checkbox("Show contour projections on walls", value=True)
    cmap_name = st.selectbox("Colormap", ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"], index=0)
    shade_surface = st.checkbox("Shaded surface", value=True)

if "components" not in st.session_state or st.session_state.get("components_k", 0) != k:
    st.session_state.components = default_components(k)
    st.session_state.components_k = k

if load_cfg is not None:
    try:
        cfg = json.loads(load_cfg.read().decode("utf-8"))
        comps = []
        for c in cfg.get("components", [])[:k]:
            comps.append(Component(
                c.get("weight", 0.2), c.get("mean_x", 0.0), c.get("mean_y", 0.0),
                max(1e-3, float(c.get("sigma_x", 1.0))), max(1e-3, float(c.get("sigma_y", 1.0))),
                float(np.clip(c.get("rho", 0.0), -0.999, 0.999)),
            ))
        if len(comps) < k:
            comps += default_components(k)[len(comps):]
        st.session_state.components = comps
        st.success("Configuration loaded.")
    except Exception as e:
        st.error(f"Failed to load config: {e}")

components: List[Component] = st.session_state.components

st.sidebar.divider()
st.sidebar.header("Components")

for i in range(k):
    with st.sidebar.expander(f"Component {i+1}", expanded=(i < 2)):
        c = components[i]
        w = st.number_input(f"w{i+1}", value=float(c.weight), step=0.05, format="%.4f", key=f"w_{i}")
        mx = st.number_input(f"μx{i+1}", value=float(c.mean_x), step=0.1, format="%.3f", key=f"mx_{i}")
        my = st.number_input(f"μy{i+1}", value=float(c.mean_y), step=0.1, format="%.3f", key=f"my_{i}")
        sx = st.number_input(f"σx{i+1}", value=float(c.sigma_x), min_value=1e-3, step=0.05, format="%.4f", key=f"sx_{i}")
        sy = st.number_input(f"σy{i+1}", value=float(c.sigma_y), min_value=1e-3, step=0.05, format="%.4f", key=f"sy_{i}")
        r = st.slider(f"ρ{i+1}", min_value=-0.999, max_value=0.999, value=float(c.rho), step=0.001, key=f"r_{i}")
        components[i] = Component(w, mx, my, sx, sy, r)
        if not is_pos_def_2x2(sx, sy, r):
            st.warning("σx, σy must be > 0 and |ρ| < 1 for a valid covariance.")

cfg_dict = {"components": [asdict(c) for c in components]}
st.download_button(
    label="Download current config (JSON)",
    file_name="gmm_config.json",
    mime="application/json",
    data=json.dumps(cfg_dict, indent=2),
)

@st.cache_data(show_spinner=False)
def compute_grid_and_pdf(x_min, x_max, y_min, y_max, res, comps: List[Component], normalized_weights: bool):
    xs = np.linspace(x_min, x_max, res)
    ys = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(xs, ys)
    Z, Z_parts = mixture_pdf_grid(comps, X, Y, normalized_weights)
    return X, Y, Z, Z_parts

X, Y, Z, Z_parts = compute_grid_and_pdf(x_min, x_max, y_min, y_max, res, components, normalized_weights)

st.subheader("Mixture PDF — 3D Surface")
fig = plt.figure(figsize=(8, 6), dpi=150)
ax = fig.add_subplot(111, projection="3d")

# Optionally log-transform density for better dynamic range
Z_plot = np.log(Z + 1e-12) if log_density else Z
zlabel = "log density" if log_density else "Density"

# Surface
surf = ax.plot_surface(
    X, Y, Z_plot,
    cmap=cmap_name,
    linewidth=0,
    antialiased=True,
    alpha=0.95,
    shade=shade_surface,
)

# Optional wireframe overlay
if draw_wire:
    ax.plot_wireframe(X, Y, Z_plot, rstride=stride, cstride=stride, linewidth=0.3, alpha=0.6)

# Optional contour projections onto the three orthogonal walls
if draw_projections:
    zmin, zmax = float(np.nanmin(Z_plot)), float(np.nanmax(Z_plot))
    ax.contour(X, Y, Z_plot, zdir='z', offset=zmin, levels=10, linewidths=0.6)
    ax.contour(X, Y, Z_plot, zdir='x', offset=x_min, levels=10, linewidths=0.6)
    ax.contour(X, Y, Z_plot, zdir='y', offset=y_min, levels=10, linewidths=0.6)
    ax.set_zlim(zmin, zmax)

ax.view_init(elev=elev, azim=azim)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(zlabel)
ax.set_title("GMM PDF Surface")
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, label=zlabel)
st.pyplot(fig, clear_figure=True)

st.caption(
    """
    Tips:
    • Use the sidebar to tune weights, means, covariances (σx, σy, ρ).\
    • "Auto-normalize (softmax)" interprets weights as unconstrained.\
    • "Use as-is (normalize sum)" divides by the sum so weights act like probabilities directly.\
    • Download / load JSON configs to share setups.
    """
)
