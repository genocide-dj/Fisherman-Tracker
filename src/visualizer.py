"""
visualizer.py
All visualization utilities:
  - 2D UMAP / t-SNE embedding scatter plots (class separation validation)
  - Identity drift heatmaps
  - Per-person working hour bar charts
  - Gantt-style presence timeline
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# Optional imports — fall back gracefully
try:
    import umap.umap_ as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ─── Color palette ────────────────────────────────────────────────────────────
PALETTE = [
    "#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
    "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"
]


# ─── Embedding 2D Plots ───────────────────────────────────────────────────────

def plot_embeddings_umap(gallery: dict, save_path: str = config.EMBED_PLOT_PATH) -> str:
    """
    UMAP 2D scatter of all gallery embeddings coloured by identity.
    Returns path to saved PNG.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ids, vectors = _flatten_gallery(gallery)
    if len(vectors) < 4:
        return _placeholder_plot("Not enough embeddings for UMAP\n(need ≥ 4)", save_path)

    if UMAP_AVAILABLE:
        reducer = umap_lib.UMAP(
            n_neighbors=config.UMAP_N_NEIGHBORS,
            min_dist=config.UMAP_MIN_DIST,
            n_components=2,
            random_state=42
        )
        reduced = reducer.fit_transform(np.array(vectors))
        method = "UMAP"
    elif TSNE_AVAILABLE:
        perp = min(config.TSNE_PERPLEXITY, len(vectors) - 1)
        reduced = TSNE(n_components=2, perplexity=perp,
                       random_state=42).fit_transform(np.array(vectors))
        method = "t-SNE"
    else:
        return _placeholder_plot("umap-learn / sklearn not installed", save_path)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    unique_ids = sorted(set(ids))
    color_map  = {pid: PALETTE[i % len(PALETTE)] for i, pid in enumerate(unique_ids)}

    for pid in unique_ids:
        mask = [i == pid for i in ids]
        pts  = reduced[mask]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=color_map[pid], label=pid, s=18, alpha=0.75, edgecolors="none")
        # Centroid label
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax.text(cx, cy, pid, fontsize=7, ha="center", va="center",
                color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=color_map[pid], alpha=0.8, lw=0))

    ax.set_title(f"{method} — Identity Embedding Clusters", fontsize=13, pad=12)
    ax.set_xlabel(f"{method} dim 1"); ax.set_ylabel(f"{method} dim 2")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Embedding plot saved → {save_path}")
    return save_path


def plot_embeddings_plotly(gallery: dict) -> "go.Figure | None":
    """Interactive Plotly version of the embedding scatter (used in dashboard)."""
    if not PLOTLY_AVAILABLE:
        return None
    ids, vectors = _flatten_gallery(gallery)
    if len(vectors) < 4:
        return None
    if UMAP_AVAILABLE:
        reducer = umap_lib.UMAP(n_neighbors=min(config.UMAP_N_NEIGHBORS, len(vectors)-1),
                                min_dist=config.UMAP_MIN_DIST, n_components=2, random_state=42)
        reduced = reducer.fit_transform(np.array(vectors))
        method  = "UMAP"
    elif TSNE_AVAILABLE:
        perp    = min(config.TSNE_PERPLEXITY, len(vectors)-1)
        reduced = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(np.array(vectors))
        method  = "t-SNE"
    else:
        return None

    df = pd.DataFrame({"x": reduced[:,0], "y": reduced[:,1], "Person": ids})
    fig = px.scatter(df, x="x", y="y", color="Person",
                     title=f"{method} — Identity Embedding Clusters",
                     labels={"x": f"{method} 1", "y": f"{method} 2"},
                     template="plotly_white", height=480)
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    return fig


# ─── Working Hour Bar Chart ───────────────────────────────────────────────────

def plot_working_hours(report_df: pd.DataFrame,
                       save_path: str = "outputs/working_hours.png") -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if report_df.empty:
        return _placeholder_plot("No data", save_path)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    persons = report_df["Person ID"].tolist()
    hours   = report_df["Total Hours"].tolist()
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(persons))]
    bars    = ax.bar(persons, hours, color=colors, width=0.55, edgecolor="white")

    for bar, h in zip(bars, hours):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{h:.2f}h", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Hours on screen"); ax.set_title("Working Hours per Person")
    ax.set_ylim(0, max(hours) * 1.2 if hours else 1)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_working_hours_plotly(report_df: pd.DataFrame) -> "go.Figure | None":
    if not PLOTLY_AVAILABLE or report_df.empty:
        return None
    fig = px.bar(report_df, x="Person ID", y="Total Hours",
                 color="Person ID", text="Total Hours",
                 title="Working Hours per Person",
                 template="plotly_white", height=400)
    fig.update_traces(texttemplate="%{text:.2f}h", textposition="outside")
    return fig


# ─── Presence Timeline (Gantt) ────────────────────────────────────────────────

def plot_presence_timeline(sessions_df: pd.DataFrame,
                           save_path: str = "outputs/presence_timeline.png") -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if sessions_df.empty:
        return _placeholder_plot("No sessions", save_path)

    persons = sorted(sessions_df["Person ID"].unique())
    color_map = {pid: PALETTE[i % len(PALETTE)] for i, pid in enumerate(persons)}

    fig, ax = plt.subplots(figsize=(14, max(3, len(persons)*0.8 + 1)), dpi=120)

    for i, pid in enumerate(persons):
        pdata = sessions_df[sessions_df["Person ID"] == pid]
        for _, row in pdata.iterrows():
            ax.barh(i, row["Duration (s)"], left=row["Start (s)"],
                    height=0.5, color=color_map[pid], alpha=0.85)

    ax.set_yticks(range(len(persons))); ax.set_yticklabels(persons)
    ax.set_xlabel("Time (seconds from video start)")
    ax.set_title("Presence Timeline per Person")
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_presence_timeline_plotly(sessions_df: pd.DataFrame) -> "go.Figure | None":
    if not PLOTLY_AVAILABLE or sessions_df.empty:
        return None
    fig = px.timeline(
        sessions_df.assign(
            Start_dt=pd.to_datetime(sessions_df["Start (s)"], unit="s"),
            End_dt  =pd.to_datetime(sessions_df["End (s)"],   unit="s")
        ),
        x_start="Start_dt", x_end="End_dt", y="Person ID",
        color="Person ID", title="Presence Timeline",
        template="plotly_white", height=400
    )
    return fig


# ─── Hourly Heatmap ───────────────────────────────────────────────────────────

def plot_hourly_heatmap(hourly_df: pd.DataFrame,
                        save_path: str = "outputs/hourly_heatmap.png") -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if hourly_df.empty:
        return _placeholder_plot("No data", save_path)

    fig, ax = plt.subplots(figsize=(14, max(3, len(hourly_df)*0.6 + 1)), dpi=120)
    im = ax.imshow(hourly_df.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(24)); ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], fontsize=7)
    ax.set_yticks(range(len(hourly_df))); ax.set_yticklabels(hourly_df.index, fontsize=9)
    ax.set_title("Hours Active per Time Slot"); ax.set_xlabel("Hour of Day")
    plt.colorbar(im, ax=ax, label="Seconds active")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_hourly_heatmap_plotly(hourly_df: pd.DataFrame) -> "go.Figure | None":
    if not PLOTLY_AVAILABLE or hourly_df.empty:
        return None
    fig = px.imshow(hourly_df,
                    labels=dict(x="Hour", y="Person", color="Seconds"),
                    title="Hourly Activity Heatmap",
                    color_continuous_scale="YlOrRd",
                    template="plotly_white", height=400)
    return fig


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _flatten_gallery(gallery: dict):
    ids, vectors = [], []
    for pid, embeds in gallery.items():
        for e in embeds:
            if e is not None:
                ids.append(pid)
                vectors.append(e)
    return ids, vectors


def _placeholder_plot(message: str, save_path: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="gray")
    ax.axis("off")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
