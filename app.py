"""
dashboard/app.py
Streamlit dashboard for the Fisherman Facial Recognition & Working Hour Estimation system.

Run with:
  streamlit run dashboard/app.py
"""

import streamlit as st
import cv2, os, sys, time, json, pickle
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# ── path fix so src imports work ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
import config
from src.pipeline   import Pipeline
from src.visualizer import (
    plot_embeddings_plotly,
    plot_working_hours_plotly,
    plot_presence_timeline_plotly,
    plot_hourly_heatmap_plotly,
)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fisherman Tracker",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 16px 20px; margin: 4px 0;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #1e40af; }
.metric-label { font-size: 0.85rem; color: #64748b; margin-top: 2px; }
.status-ok { color: #16a34a; font-weight: 600; }
.status-warn { color: #d97706; font-weight: 600; }
section[data-testid="stSidebar"] { background: #f1f5f9 !important; }
section[data-testid="stSidebar"] * { color: #1e293b !important; }
section[data-testid="stSidebar"] .stRadio label { color: #1e293b !important; }
section[data-testid="stSidebar"] .stCaption { color: #475569 !important; }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 { color: #0f172a !important; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.image("https://img.icons8.com/fluency/96/fish.png", width=64)
    st.sidebar.title("🎣 Fisherman Tracker")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["🎬 Run Pipeline", "📊 Results Dashboard", "🔬 Embedding Analysis",
         "⏱ Working Hours", "⚙️ Settings"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline Config**")
    st.sidebar.caption(f"YOLO conf: {config.YOLO_CONF_THRESH}")
    st.sidebar.caption(f"Re-ID thresh: {config.REID_SIMILARITY_THRESH}")
    st.sidebar.caption(f"Embed model: {config.EMBEDDING_MODEL}")
    st.sidebar.caption(f"Frame skip: {config.FRAME_SKIP}")

    # Gallery status
    gallery_path = config.IDENTITY_DB_PATH
    if os.path.exists(gallery_path):
        with open(gallery_path, "rb") as f:
            g = pickle.load(f)
        n_ids = len(g.get("gallery", {}))
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Identity Gallery**: {n_ids} known persons")
        if st.sidebar.button("🗑 Clear Gallery"):
            os.remove(gallery_path)
            st.sidebar.success("Gallery cleared!")

    return page


# ─── Pages ────────────────────────────────────────────────────────────────────

def page_run_pipeline():
    st.title("🎬 Run Pipeline")
    st.markdown("Upload a surveillance video to run the full end-to-end tracking pipeline.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload video file", type=["mp4", "avi", "mov", "mkv"],
            help="24-hour surveillance footage or any test clip"
        )

    with col2:
        st.markdown("**Options**")
        save_video   = st.checkbox("Save annotated video", value=True)
        load_gallery = st.checkbox("Load existing gallery", value=False,
                                   help="Resume from a previous session")
        show_prev    = st.checkbox("Show live preview", value=False,
                                   help="Disable on headless servers")

    if uploaded is None:
        st.info("👆 Upload a video to get started. You can use any MP4/AVI clip.")

        # Demo mode with sample data
        st.markdown("---")
        st.markdown("### 📦 Or Load Existing Results")
        if os.path.exists(config.REPORT_CSV):
            if st.button("📂 Load Last Run Results"):
                st.session_state["results_loaded"] = True
                st.success("Previous results loaded — go to Results Dashboard →")
        return

    # Show video info
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read()); tfile.flush()
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    st.success(f"✅ Video loaded: **{uploaded.name}** — {w}×{h} @ {fps:.0f}fps — {frames} frames "
               f"({frames/fps/3600:.2f} hours)")

    if st.button("🚀 Run Tracking Pipeline", type="primary"):
        progress_bar = st.progress(0, text="Initialising pipeline...")
        status_box   = st.empty()
        log_box      = st.empty()

        try:
            with st.spinner("Running pipeline — this may take a while for long videos..."):
                pipeline = Pipeline(load_gallery=load_gallery)
                t0 = time.time()
                results  = pipeline.run(
                    video_path,
                    save_video=save_video,
                    save_report=True,
                    show_preview=show_prev
                )
                elapsed = time.time() - t0

            st.session_state["pipeline_results"] = results
            progress_bar.progress(100, text="Complete!")
            status_box.success(
                f"✅ Pipeline complete in {elapsed:.0f}s  |  "
                f"Found **{len(results['gallery'])} identities**  |  "
                f"Re-IDs: {results['stats']['reidentifications']}"
            )

            # Quick summary metrics
            report = results["report_df"]
            if not report.empty:
                cols = st.columns(len(report))
                for i, (_, row) in enumerate(report.iterrows()):
                    with cols[i]:
                        st.metric(row["Person ID"], row["Total (HH:MM:SS)"],
                                  f"{row['Num Sessions']} sessions")

            st.markdown("→ Go to **Results Dashboard** for full analysis.")

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback; st.code(traceback.format_exc())

        finally:
            os.unlink(video_path)


def page_results():
    st.title("📊 Results Dashboard")

    results = st.session_state.get("pipeline_results")

    # Try loading from disk if no in-memory results
    if results is None and os.path.exists(config.REPORT_CSV):
        report_df   = pd.read_csv(config.REPORT_CSV)
        sessions_df = pd.read_csv("outputs/sessions.csv") if os.path.exists("outputs/sessions.csv") else pd.DataFrame()
        hourly_df   = pd.read_csv("outputs/hourly_activity.csv", index_col=0) if os.path.exists("outputs/hourly_activity.csv") else pd.DataFrame()
        gallery     = {}
        stats       = {}

        if os.path.exists(config.IDENTITY_DB_PATH):
            with open(config.IDENTITY_DB_PATH, "rb") as f:
                g = pickle.load(f)
            gallery = g.get("gallery", {})
            stats   = g.get("stats", {})

        results = {
            "report_df": report_df, "sessions_df": sessions_df,
            "hourly_df": hourly_df, "gallery": gallery, "stats": stats
        }

    if results is None:
        st.warning("No results found. Run the pipeline first.")
        return

    report_df = results["report_df"]
    stats     = results.get("stats", {})

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 People Tracked",     len(report_df))
    col2.metric("🔁 Re-identifications", stats.get("reidentifications", "—"))
    col3.metric("🛡 Switches Prevented",  stats.get("identity_switches_prevented", "—"))
    col4.metric("📋 Work Sessions",       results.get("sessions_df", pd.DataFrame()).shape[0])

    st.markdown("---")

    # ── Working Hours Chart ────────────────────────────────────────────────────
    st.subheader("⏱ Working Hours per Person")
    fig = plot_working_hours_plotly(report_df)
    if fig: st.plotly_chart(fig, use_container_width=True)

    st.dataframe(report_df, use_container_width=True)

    # ── Presence Timeline ─────────────────────────────────────────────────────
    sessions_df = results.get("sessions_df", pd.DataFrame())
    if not sessions_df.empty:
        st.subheader("📅 Presence Timeline")
        fig = plot_presence_timeline_plotly(sessions_df)
        if fig: st.plotly_chart(fig, use_container_width=True)

    # ── Hourly Heatmap ────────────────────────────────────────────────────────
    hourly_df = results.get("hourly_df", pd.DataFrame())
    if not hourly_df.empty:
        st.subheader("🌡 Hourly Activity Heatmap")
        fig = plot_hourly_heatmap_plotly(hourly_df)
        if fig: st.plotly_chart(fig, use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Download Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("⬇ Working Hours CSV", report_df.to_csv(index=False),
                           "working_hours.csv", "text/csv")
    with col2:
        if not sessions_df.empty:
            st.download_button("⬇ Sessions CSV", sessions_df.to_csv(index=False),
                               "sessions.csv", "text/csv")
    with col3:
        if os.path.exists(config.ANNOTATED_VIDEO):
            with open(config.ANNOTATED_VIDEO, "rb") as f:
                st.download_button("⬇ Annotated Video", f, "annotated.mp4", "video/mp4")


def page_embeddings():
    st.title("🔬 Embedding Analysis")
    st.markdown(
        "2D visualisation of identity embedding clusters. "
        "Well-separated clusters indicate good class separation and low identity drift."
    )

    results = _load_results()
    if results is None:
        st.warning("No results found. Run the pipeline first."); return

    gallery = results.get("gallery", {})

    if not gallery:
        st.warning("No gallery data found."); return

    col1, col2 = st.columns([2, 1])
    with col2:
        method = st.selectbox("Reduction method", ["UMAP", "t-SNE"])
        if st.button("♻ Recompute"):
            st.cache_data.clear()

    with col1:
        with st.spinner("Computing 2D projections..."):
            fig = plot_embeddings_plotly(gallery)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                if os.path.exists(config.EMBED_PLOT_PATH):
                    st.image(config.EMBED_PLOT_PATH)
                else:
                    st.info("Not enough embeddings to plot (need ≥ 4 per identity).")

    # Gallery stats table
    st.subheader("Gallery Summary")
    summary = pd.DataFrame([
        {"Person": k, "Embeddings stored": len(v)}
        for k, v in gallery.items()
    ])
    st.dataframe(summary, use_container_width=True)

    # Similarity matrix
    if len(gallery) >= 2:
        st.subheader("Inter-Identity Similarity Matrix")
        from src.embedder import FaceEmbedder
        import plotly.express as px
        persons = sorted(gallery.keys())
        means   = {}
        for pid in persons:
            embs = [e for e in gallery[pid] if e is not None]
            if embs:
                means[pid] = FaceEmbedder.mean_embedding(embs)
        if len(means) >= 2:
            sim_matrix = np.zeros((len(persons), len(persons)))
            for i, pi in enumerate(persons):
                for j, pj in enumerate(persons):
                    if pi in means and pj in means:
                        sim_matrix[i,j] = round(FaceEmbedder.cosine_similarity(means[pi], means[pj]), 3)
            sim_df = pd.DataFrame(sim_matrix, index=persons, columns=persons)
            fig = px.imshow(sim_df, text_auto=True, color_continuous_scale="Blues",
                            title="Pairwise cosine similarity (diagonal=1.0, lower off-diag = better separation)",
                            zmin=0, zmax=1)
            st.plotly_chart(fig, use_container_width=True)


def page_hours():
    st.title("⏱ Working Hours Detail")
    results = _load_results()
    if results is None:
        st.warning("No results found."); return

    report_df   = results["report_df"]
    sessions_df = results.get("sessions_df", pd.DataFrame())

    if report_df.empty:
        st.warning("No working hour data available."); return

    # Person selector
    persons = report_df["Person ID"].tolist()
    sel = st.selectbox("Select person", persons)

    row = report_df[report_df["Person ID"] == sel].iloc[0]
    st.markdown(f"### {sel}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Time",    row["Total (HH:MM:SS)"])
    col2.metric("Total Hours",   row["Total Hours"])
    col3.metric("Sessions",      row["Num Sessions"])
    col4.metric("Longest (s)",   row["Longest Session (s)"])

    if not sessions_df.empty:
        st.subheader("Sessions")
        person_sessions = sessions_df[sessions_df["Person ID"] == sel]
        st.dataframe(person_sessions, use_container_width=True)

    # Hourly breakdown
    hourly_df = results.get("hourly_df", pd.DataFrame())
    if not hourly_df.empty and sel in hourly_df.index:
        import plotly.graph_objects as go
        row_data = hourly_df.loc[sel]
        fig = go.Figure(go.Bar(
            x=[f"{h:02d}:00" for h in range(24)],
            y=row_data.values,
            marker_color="#4C72B0"
        ))
        fig.update_layout(title=f"Hourly Activity — {sel}",
                          xaxis_title="Hour", yaxis_title="Seconds active",
                          template="plotly_white", height=350)
        st.plotly_chart(fig, use_container_width=True)


def page_settings():
    st.title("⚙️ Settings")
    st.markdown("Edit `config.py` to change these values. They take effect on the next pipeline run.")

    st.subheader("Current Configuration")
    cfg_pairs = {
        "YOLO Confidence":      config.YOLO_CONF_THRESH,
        "Re-ID Threshold":      config.REID_SIMILARITY_THRESH,
        "Embedding Model":      config.EMBEDDING_MODEL,
        "Frame Skip":           config.FRAME_SKIP,
        "YOLO Min Face Size":   config.YOLO_MIN_FACE_SIZE,
        "DeepSORT Max Age":     config.DEEPSORT_MAX_AGE,
        "Max Gap (seconds)":    config.MAX_GAP_SECONDS,
        "Gallery Max Per ID":   config.REID_GALLERY_MAX_PER_ID,
    }
    st.table(pd.DataFrame(cfg_pairs.items(), columns=["Parameter", "Value"]))

    st.markdown("---")
    st.subheader("Output Files")
    files = [
        (config.REPORT_CSV,         "Working Hours CSV"),
        (config.EMBED_PLOT_PATH,    "Embedding Cluster Plot"),
        (config.ANNOTATED_VIDEO,    "Annotated Video"),
        (config.IDENTITY_DB_PATH,   "Identity Gallery"),
        ("outputs/sessions.csv",    "Sessions CSV"),
        ("outputs/run_metadata.json","Run Metadata"),
    ]
    for path, label in files:
        exists = os.path.exists(path)
        col1, col2 = st.columns([3, 1])
        col1.markdown(f"{'✅' if exists else '❌'} **{label}** — `{path}`")
        if exists and col2.button(f"Delete", key=path):
            os.remove(path); st.rerun()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_results():
    """Load results from session state or disk."""
    r = st.session_state.get("pipeline_results")
    if r: return r
    if not os.path.exists(config.REPORT_CSV):
        return None

    report_df = pd.read_csv(config.REPORT_CSV)
    sessions_df = pd.read_csv("outputs/sessions.csv") if os.path.exists("outputs/sessions.csv") else pd.DataFrame()
    hourly_df   = pd.read_csv("outputs/hourly_activity.csv", index_col=0) if os.path.exists("outputs/hourly_activity.csv") else pd.DataFrame()
    gallery, stats = {}, {}
    if os.path.exists(config.IDENTITY_DB_PATH):
        with open(config.IDENTITY_DB_PATH, "rb") as f:
            g = pickle.load(f)
        gallery = g.get("gallery", {})
        stats   = g.get("stats", {})
    return {"report_df": report_df, "sessions_df": sessions_df,
            "hourly_df": hourly_df, "gallery": gallery, "stats": stats}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()
    if   page == "🎬 Run Pipeline":        page_run_pipeline()
    elif page == "📊 Results Dashboard":   page_results()
    elif page == "🔬 Embedding Analysis":  page_embeddings()
    elif page == "⏱ Working Hours":        page_hours()
    elif page == "⚙️ Settings":            page_settings()


if __name__ == "__main__":
    main()
