# =========================
# Climate Risk Heatmaps Explorer — v4.3
# =========================
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io, zipfile
import plotly.express as px
from urllib.parse import urlencode

# --- Portable project bootstrap (find repo root, import in_out) ---
import sys
from pathlib import Path as _Path  # avoid name clash with your Path import
cur = _Path.cwd().resolve()
while cur != cur.parent and not ((cur / "lib").exists() and (cur / "config.yaml").exists()):
    cur = cur.parent
sys.path.insert(0, str(cur)); sys.path.insert(0, str(cur / "lib"))
from lib.paths import in_out, ensure_out_dirs
ensure_out_dirs()


# =========================
# CONFIG
# =========================
# outputs/ root (portable, matches your notebooks’ standardized exports)
ROOT = Path(in_out("."))   # e.g., <project>/outputs


RISK_TYPES = ["Transition", "Physical"]
HORIZONS = ["2020-2030", "2020-2050"]

# Transition now includes PM, BO, AS + Eq9 variants
TRANSITION_METRICS = [
    "Eq9 (regional BAU)",
    "Eq9 (BAU=$20)",
    "Path misalignment (PM)",
    "Budget overshoot (BO)",
    "Abatement share (AS)",
]

# Physical: scenarios, layouts, metrics
PHYS_SCENARIOS = ["DAPS", "DIRE"]
PHYS_LAYOUTS   = ["Regions x Sectors", "10 Categories"]
PHYS_METRICS   = ["HDW", "SF", "Average (HDW & SF)"]  # "Average" can be R×S or 10-cats

PALETTES = ["RdYlGn_r", "Viridis", "Cividis"]  # default matches exports (first)

# =========================
# HELPERS (files, csv/xlsx, plotting, UI)
# =========================
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_long_from_matrix(mat_df: pd.DataFrame, index_name="Region", col_name="Sector", value_name="Value"):
    idx = mat_df.index.name or index_name
    return (mat_df.reset_index()
            .rename(columns={idx: index_name})
            .melt(id_vars=index_name, var_name=col_name, value_name=value_name))

def write_heatmap_tables(base_outdir: Path, tag: str, mat_df: pd.DataFrame):
    """Write wide + long CSV next to PNG/PDF. tag is a short name used in filenames."""
    data_dir = _ensure_dir(base_outdir / "data")
    wide_path = data_dir / f"{tag}_heatmap_data.csv"
    long_path = data_dir / f"{tag}_heatmap_data_long.csv"
    mat_df.to_csv(wide_path)
    _to_long_from_matrix(mat_df).to_csv(long_path, index=False)
    return wide_path, long_path

def to_excel_bytes(df_dict: dict[str, pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, df in df_dict.items():
            if df is None:
                continue
            df.to_excel(writer, sheet_name=name[:31])
    bio.seek(0)
    return bio.read()

# Streamlit < 1.30 compatibility: prefer use_container_width, fall back to use_column_width
def show_image(img_bytes):
    if img_bytes is None:
        return
    try:
        st.image(img_bytes, use_container_width=True)
    except TypeError:
        st.image(img_bytes, use_column_width=True)

def file_mtime(path: Path):
    return path.stat().st_mtime if path.exists() else None

def fmt_mtime(path: Path):
    ts = file_mtime(path)
    if not ts: return "n/a"
    import datetime as _dt
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

# ------- mtime-aware caches (cache key includes mtime) -------
@st.cache_data(show_spinner=False)
def cached_read_png_bytes(path: Path, mtime: float | None):
    if not path or not isinstance(path, Path) or not path.exists():
        return None
    with open(path, "rb") as f:
        return f.read()

@st.cache_data(show_spinner=False)
def cached_read_csv(path: Path, mtime: float | None):
    if not path or not isinstance(path, Path) or not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return None

def zip_bundle(img_bytes: bytes | None, wide_df: pd.DataFrame | None, long_df: pd.DataFrame | None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if img_bytes:
            zf.writestr("heatmap.png", img_bytes)
        if wide_df is not None:
            zf.writestr("heatmap_data.csv", wide_df.to_csv().encode("utf-8"))
        if long_df is not None:
            zf.writestr("heatmap_data_long.csv", long_df.to_csv().encode("utf-8"))
    buf.seek(0)
    return buf.getvalue()

def p95_from_df(df: pd.DataFrame):
    vals = df.to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 95)) if vals.size else 1.0

def plot_interactive_heatmap(df: pd.DataFrame, palette: str, cap_mode: str, title=""):
    zmax = p95_from_df(df) if cap_mode == "p95" else np.nanmax(df.to_numpy(dtype=float))
    fig = px.imshow(df,
                    color_continuous_scale=palette,
                    zmin=0, zmax=zmax,
                    aspect="auto",
                    labels=dict(x="Sector", y="Region", color="Value"),
                    title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                      coloraxis_colorbar=dict(title="Value"))
    fig.update_xaxes(side="bottom", tickangle=45)
    return fig

def share_url(**state):
    base = dict(st.query_params)  # Streamlit stable API
    base.update({k: [v] for k, v in state.items() if v is not None})
    return f"?{urlencode({k: v[0] for k, v in base.items()})}"

def long_from_wide(df: pd.DataFrame, index_label_fallback="Region"):
    idx_name = df.index.name or index_label_fallback
    return (df.reset_index()
              .rename(columns={idx_name: "Region"})
              .melt(id_vars="Region", var_name="Sector", value_name="Value")
              .dropna())

# =========================
# TRANSITION PATH HELPERS
# =========================
def _metric_key(metric: str) -> str:
    """Map human label -> folder key."""
    ml = metric.lower()
    if "regional" in ml: return "eq9_regional"
    if "bau=$20" in ml or "bau=20" in ml: return "eq9_bau20"
    if "path misalignment" in ml or "(pm)" in ml: return "pm"
    if "budget overshoot" in ml or "(bo)" in ml: return "bo"
    if "abatement share" in ml or "(as)" in ml: return "as"
    return "pm"  # safe default

def is_eq9(metric: str) -> bool:
    mk = _metric_key(metric)
    return mk in {"eq9_regional", "eq9_bau20"}

def std_transition_paths(horizon: str, metric: str, rtag: str | None):
    """
    rtag: 'r2' or 'r0' for Eq9; None/'' for PM/BO/AS.
    """
    mk = _metric_key(metric)
    base = ROOT / "transition" / horizon / mk
    if mk in {"eq9_regional", "eq9_bau20"}:
        if rtag not in {"r2","r0"}:
            rtag = "r2"  # default
        base = base / rtag
    return {
        "data": base / "data" / "heatmap_data.csv",
        "data_long": base / "data" / "heatmap_data_long.csv",
        "png": base / "png" / "heatmap.png",
        "root": base
    }

# =========================
# PHYSICAL PATH HELPERS (match your outputs; robust to two naming styles)
# =========================
def _scenario_titlecase(s: str) -> str:
    # Your 10-cat PNGs use "DAPS__..." and "DiRe__..." — keep that exact casing.
    return "DiRe" if s.upper() == "DIRE" else "DAPS"

def physical_paths_regions(scenario: str, storyline: str):
    base = ROOT / f"physical_shortterm_{scenario.upper()}"
    dire_tag = "__DIRE" if scenario.upper() == "DIRE" else ""
    return {
        "png":  base / "png" / f"total_loss__{storyline}__regionsxsectors__peak{dire_tag}.png",
        "pdf":  base / "pdf" / f"total_loss__{storyline}__regionsxsectors__peak{dire_tag}.pdf",
        "data": base / "data" / f"{storyline}_regionsxsectors_peak{dire_tag}_heatmap_data.csv",
        "data_long": base / "data" / f"{storyline}_regionsxsectors_peak{dire_tag}_heatmap_data_long.csv",
        "root": base
    }

def physical_paths_10cats(scenario: str, storyline: str):
    base = ROOT / f"physical_shortterm_{scenario.upper()}"
    dire_tag = "__DIRE" if scenario.upper() == "DIRE" else ""
    # Two possible PNG/PDF stems:
    #  1) Legacy/no-model-prefix: HDW_10cats_peak[__DIRE].png
    #  2) Current code with model prefix: DAPS__HDW__10cats__peak.png or DiRe__HDW__10cats__peak.png
    fn_legacy = f"{storyline}_10cats_peak{dire_tag}"
    fn_modelp = f"{_scenario_titlecase(scenario)}__{storyline}__10cats__peak"
    return {
        "png":     base / "png_10cats" / f"{fn_legacy}.png",
        "png_alt": base / "png_10cats" / f"{fn_modelp}.png",
        "pdf":     base / "pdf_10cats" / f"{fn_legacy}.pdf",
        "pdf_alt": base / "pdf_10cats" / f"{fn_modelp}.pdf",
        "data":    base / "data_10cats" / f"{fn_legacy}_heatmap_data.csv",         # script writes legacy names for CSVs
        "data_long": base / "data_10cats" / f"{fn_legacy}_heatmap_data_long.csv",
        "root": base
    }

def physical_paths_average(scenario: str):
    base = ROOT / f"physical_shortterm_{scenario.upper()}"
    tag = f"{scenario.upper()}_Average_HDW_SF"
    return {
        "png":  base / "png" / f"{tag}.png",
        "pdf":  base / "pdf" / f"{tag}.pdf",
        "data": base / "data" / f"{tag}_heatmap_data.csv",
        "data_long": base / "data" / f"{tag}_heatmap_data_long.csv",
        "root": base
    }

# NEW: 10-categories "Average (HDW & SF)" paths
def physical_paths_10cats_average(scenario: str):
    base = ROOT / f"physical_shortterm_{scenario.upper()}"
    tag_data = "AVG_HDW_SF_10cats"  # your CSV base names
    png_alt  = f"{('DiRe' if scenario.upper()=='DIRE' else 'DAPS')}__AVG_HDW_SF__10cats.png"
    return {
        # primary (legacy style; may not exist)
        "png":     base / "png_10cats" / f"{tag_data}.png",
        "pdf":     base / "pdf_10cats" / f"{tag_data}.pdf",
        # current code with model prefix (likely exists)
        "png_alt": base / "png_10cats" / png_alt,
        "pdf_alt": base / "pdf_10cats" / png_alt.replace(".png", ".pdf"),
        "data":    base / "data_10cats" / f"{tag_data}_heatmap_data.csv",
        "data_long": base / "data_10cats" / f"{tag_data}_heatmap_data_long.csv",
        "root": base
    }

def _pick_png(paths: dict):
    """Return (img_bytes, chosen_path), mtime-aware."""
    p = paths.get("png")
    img = cached_read_png_bytes(p, file_mtime(p)) if p else None
    if img:
        return img, p
    alt = paths.get("png_alt")
    img2 = cached_read_png_bytes(alt, file_mtime(alt)) if alt else None
    if img2:
        return img2, alt
    return None, p

def physical_paths_10cats_pick_png(paths: dict):
    """Back-compat name kept; delegates to _pick_png."""
    return _pick_png(paths)

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Climate Risk Heatmaps", layout="wide")
st.title("Climate Risk Heatmaps Explorer")

with st.sidebar:
    st.header("Controls")
    # Dev helper: clear cache
    if st.button("🔄 Refresh data cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")

risk = st.sidebar.radio("Risk type", RISK_TYPES)

with st.expander("What am I looking at? (Methodology, metrics, downloads)"):
    st.markdown(
        """
**Heatmaps** show region × sector values exported by your analysis.

**Transition metrics**  
- **Path misalignment (PM):** how far NZ emissions deviate from BAU, by sector/region.  
- **Budget overshoot (BO):** when cumulative NZ emissions exceed the carbon budget.  
- **Abatement share (AS):** share of abatement relative to BAU (capped visually in exports).  
- **Eq9 (normalized cost):** relative cost of moving from BAU to NZ; two denominator choices:
  - *Eq9 (regional BAU):* region's own BAU carbon price.
  - *Eq9 (BAU=$20):* fixed $20/tCO₂ everywhere.
  - Discount rate: **r2 = 2%**, **r0 = 0%** (weight on near-term).

**Physical (short-term, peak 2023–2030)**  
- Storylines: **HDW** (Heatwave + Drought + Wildfire), **SF** (Storm + Flood).  
- Scenarios: **DAPS**, **DIRE**.  
- Layouts: **Regions × Sectors** and **10 Categories**.  
- Also: **Average (HDW & SF)** pages per scenario.  
- Higher = worse.

**Downloads**: PNG, CSV (wide), optional CSV (long), Excel, or a ZIP bundle.
        """
    )

# =========================
# TRANSITION VIEW
# =========================
if risk == "Transition":
    horizon = st.sidebar.selectbox("Time horizon", HORIZONS)
    metric = st.sidebar.selectbox("Metric", TRANSITION_METRICS)

    # Show discount-rate only for Eq9
    rtag = None
    if is_eq9(metric):
        rtag = st.sidebar.radio("Discount rate (Eq9 only)", ["r2", "r0"], index=0)

    # View options (transition)
    st.sidebar.markdown("**View options**")
    interactive = st.sidebar.toggle("Interactive heatmap", value=False)
    palette = st.sidebar.selectbox("Color palette", PALETTES, index=0)
    cap_mode = st.sidebar.radio("Color cap", ["p95", "max"], index=0)

    paths = std_transition_paths(horizon, metric, rtag)

    title_bits = [metric, horizon]
    if is_eq9(metric): title_bits.append(rtag)
    st.subheader(" — ".join(title_bits))

    img_bytes = cached_read_png_bytes(paths["png"], file_mtime(paths["png"]))
    wide_df   = cached_read_csv(paths["data"], file_mtime(paths["data"]))
    long_df   = cached_read_csv(paths["data_long"], file_mtime(paths["data_long"]))

    # meta line
    col_info = st.columns([1, 1, 3])
    with col_info[0]:
        st.caption(f"PNG updated: {fmt_mtime(paths['png'])}")
    with col_info[1]:
        st.caption(f"CSV updated: {fmt_mtime(paths['data'])}")
    with col_info[2]:
        st.caption(f"Path: {paths['root']}")

    # --- Heatmap display ---
    title_txt = " — ".join(title_bits)
    if interactive and (wide_df is not None):
        fig = plot_interactive_heatmap(wide_df, palette=palette, cap_mode=cap_mode, title=title_txt)
        st.plotly_chart(fig, use_container_width=True)
    else:
        if img_bytes:
            show_image(img_bytes)
        else:
            st.warning("Heatmap image not found.")

    # Downloads
    if img_bytes:
        fname = f"transition_{_metric_key(metric)}_{horizon}{'_' + rtag if is_eq9(metric) else ''}.png"
        st.download_button("Download heatmap (PNG)", img_bytes, file_name=fname, mime="image/png")

    if wide_df is not None:
        st.dataframe(wide_df, use_container_width=True, height=500)
        st.download_button("Download data (CSV, wide)",
                           wide_df.to_csv().encode("utf-8"),
                           file_name=f"transition_{_metric_key(metric)}_{horizon}{'_' + rtag if is_eq9(metric) else ''}.csv",
                           mime="text/csv")
        if long_df is not None:
            st.download_button("Download data (CSV, long)",
                               long_df.to_csv().encode("utf-8"),
                               file_name=f"transition_{_metric_key(metric)}_{horizon}{'_' + rtag if is_eq9(metric) else ''}_long.csv",
                               mime="text/csv")
        # Excel
        xls = to_excel_bytes({"wide": wide_df, "long": long_df})
        st.download_button("Download data (Excel)", xls,
                           file_name=f"transition_{_metric_key(metric)}_{horizon}{'_' + rtag if is_eq9(metric) else ''}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        # ZIP bundle
        st.download_button("Download bundle (ZIP)",
                           zip_bundle(img_bytes, wide_df, long_df),
                           file_name=f"transition_{_metric_key(metric)}_{horizon}{'_' + rtag if is_eq9(metric) else ''}.zip",
                           mime="application/zip")
    else:
        st.info("No data table found for this selection.")

    # ---------- Hot spots ----------
    st.markdown("### Hot spots")
    if wide_df is not None:
        max_val = float(np.nanmax(wide_df.to_numpy(dtype=float))) if not wide_df.empty else 0.0
        step = max(0.01, round(max_val/200, 3))
        thresh = st.slider("Highlight threshold", min_value=0.0, max_value=max(0.1, max_val),
                           value=0.0, step=step)
        long_sorted = long_from_wide(wide_df, index_label_fallback="Region").sort_values("Value", ascending=False)
        if thresh > 0:
            long_sorted = long_sorted[long_sorted["Value"] > thresh]
        st.caption(f"Cells above threshold: **{int((wide_df > thresh).sum().sum())}**")
        N = st.slider("Show Top-N", 5, 30, 10)
        st.dataframe(long_sorted.head(N), use_container_width=True, height=240)

    st.markdown("---")
    st.header("Comparison")
    st.caption("Compare across horizons and metrics (up to 4 tiles).")

    sel_horiz = st.multiselect("Horizons", HORIZONS, default=[horizon])
    sel_metrics = st.multiselect("Metrics", TRANSITION_METRICS, default=[metric])
    any_eq9 = any(is_eq9(m) for m in sel_metrics)
    sel_rt = [""]  # placeholder for non-Eq9
    if any_eq9:
        sel_rt = st.multiselect("Rates (for Eq9 tiles)", ["r2", "r0"], default=["r2"])

    # Build combos, respecting Eq9 vs non-Eq9
    combos = []
    for h in sel_horiz:
        for m in sel_metrics:
            if is_eq9(m):
                for r in sel_rt[:2]:  # cap to 2 tiles for layout
                    combos.append((h, m, r))
            else:
                combos.append((h, m, None))
    combos = combos[:4]

    cols = st.columns(2)
    for i, (h, m, r) in enumerate(combos):
        p = std_transition_paths(h, m, r)
        with cols[i % 2]:
            title = f"{m}, {h}" + (f", {r}" if is_eq9(m) else "")
            st.subheader(title)
            img = cached_read_png_bytes(p["png"], file_mtime(p["png"]))
            if img:
                show_image(img)
                st.download_button("PNG", img,
                                   file_name=f"transition_{_metric_key(m)}_{h}{'_' + r if is_eq9(m) else ''}.png",
                                   mime="image/png", key=f"tpng_{i}")
            else:
                st.warning("PNG missing")
            d = cached_read_csv(p["data"], file_mtime(p["data"]))
            if d is not None:
                st.download_button("CSV (wide)",
                                   d.to_csv().encode("utf-8"),
                                   file_name=f"transition_{_metric_key(m)}_{h}{'_' + r if is_eq9(m) else ''}.csv",
                                   mime="text/csv", key=f"tcsv_{i}")

    # ---------- Delta view (A − B) ----------
    st.markdown("### Delta view (A − B)")
    colA, colB = st.columns(2)
    with colA:
        A_h = st.selectbox("A: Horizon", HORIZONS, key="A_h")
        A_m = st.selectbox("A: Metric", TRANSITION_METRICS, key="A_m")
        A_r = st.selectbox("A: Rate (Eq9 only)", ["r2", "r0"], key="A_r") if is_eq9(A_m) else None
    with colB:
        B_h = st.selectbox("B: Horizon", HORIZONS, key="B_h")
        B_m = st.selectbox("B: Metric", TRANSITION_METRICS, key="B_m")
        B_r = st.selectbox("B: Rate (Eq9 only)", ["r2", "r0"], key="B_r") if is_eq9(B_m) else None

    A_paths = std_transition_paths(A_h, A_m, A_r)
    B_paths = std_transition_paths(B_h, B_m, B_r)
    A_df = cached_read_csv(A_paths["data"], file_mtime(A_paths["data"]))
    B_df = cached_read_csv(B_paths["data"], file_mtime(B_paths["data"]))

    if (A_df is not None) and (B_df is not None):
        common_rows = A_df.index.intersection(B_df.index)
        common_cols = A_df.columns.intersection(B_df.columns)
        delta = A_df.loc[common_rows, common_cols] - B_df.loc[common_rows, common_cols]
        figD = px.imshow(
            delta,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            aspect="auto",
            labels=dict(x="Sector", y="Region", color="Δ A−B"),
            title=(
                f"Δ (A − B): {A_m}/{A_h}{'/' + A_r if is_eq9(A_m) else ''}  minus  "
                f"{B_m}/{B_h}{'/' + B_r if is_eq9(B_m) else ''}"
            ),
        )
        figD.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        figD.update_xaxes(side="bottom", tickangle=45)
        st.plotly_chart(figD, use_container_width=True)
        st.download_button(
            "Download Δ (CSV)",
            delta.to_csv().encode("utf-8"),
            file_name="transition_delta_A_minus_B.csv",
            mime="text/csv",
        )
    else:
        st.info("Both selections must have data to compute Δ.")

    st.caption("Share this view:")
    st.code(share_url(risk=risk, horizon=horizon, metric=metric, rtag=(rtag if is_eq9(metric) else "")))

# =========================
# PHYSICAL VIEW (Regions×Sectors, 10 Categories, Average)
# =========================
elif risk == "Physical":
    scenario   = st.sidebar.selectbox("Scenario", PHYS_SCENARIOS)         # DAPS / DIRE
    layout     = st.sidebar.selectbox("Layout", PHYS_LAYOUTS)             # Regions x Sectors / 10 Categories
    metric_sel = st.sidebar.selectbox("Metric", PHYS_METRICS)             # HDW / SF / Average

    # View options (physical)
    st.sidebar.markdown("**View options**")
    interactive = st.sidebar.toggle("Interactive heatmap", value=False)
    palette     = st.sidebar.selectbox("Color palette", PALETTES, index=0)
    cap_mode    = st.sidebar.radio("Color cap", ["p95", "max"], index=0)

    # Resolve paths (now respects Layout for "Average")
    if metric_sel == "Average (HDW & SF)":
        if layout == "10 Categories":
            paths = physical_paths_10cats_average(scenario)
            title_txt = f"{scenario} — Average (HDW & SF) — 10 Categories (peak)"
            img_bytes, chosen_png_path = _pick_png(paths)
        else:
            paths = physical_paths_average(scenario)
            title_txt = f"{scenario} — Average (HDW & SF) — Regions × Sectors (peak)"
            chosen_png_path = paths["png"]
            img_bytes = cached_read_png_bytes(chosen_png_path, file_mtime(chosen_png_path))
    else:
        storyline = "HDW" if metric_sel == "HDW" else "SF"
        if layout == "Regions x Sectors":
            paths = physical_paths_regions(scenario, storyline)
            title_txt = f"{scenario} — {storyline} — Regions × Sectors (peak)"
            chosen_png_path = paths["png"]
            img_bytes = cached_read_png_bytes(chosen_png_path, file_mtime(chosen_png_path))
        else:
            paths = physical_paths_10cats(scenario, storyline)
            title_txt = f"{scenario} — {storyline} — 10 Categories (peak)"
            img_bytes, chosen_png_path = physical_paths_10cats_pick_png(paths)

    st.subheader(title_txt)

    wide_df = cached_read_csv(paths["data"], file_mtime(paths["data"]))
    long_df = cached_read_csv(paths["data_long"], file_mtime(paths["data_long"]))

    col_info = st.columns([1, 1, 3])
    with col_info[0]:
        st.caption(f"PNG updated: {fmt_mtime(chosen_png_path)}")
    with col_info[1]:
        st.caption(f"CSV updated: {fmt_mtime(paths['data'])}")
    with col_info[2]:
        st.caption(f"Path: {paths['root']}")

    # --- Heatmap display ---
    if interactive and (wide_df is not None):
        fig = plot_interactive_heatmap(wide_df, palette=palette, cap_mode=cap_mode, title=title_txt)
        st.plotly_chart(fig, use_container_width=True)
    else:
        if img_bytes:
            show_image(img_bytes)
        else:
            st.warning("Heatmap image not found.")

    # Downloads (filename reflects chosen layout)
    base_fn = (
        f"physical_{scenario}_"
        f"{metric_sel.replace(' ','_')}_"
        f"{'10cats' if (layout=='10 Categories') else 'regionsxsectors'}"
    )
    if img_bytes:
        st.download_button("Download heatmap (PNG)", img_bytes,
                           file_name=f"{base_fn}.png", mime="image/png")

    if wide_df is not None:
        st.dataframe(wide_df, use_container_width=True, height=500)
        # CSVs
        st.download_button("Download data (CSV, wide)",
                           wide_df.to_csv().encode("utf-8"),
                           file_name=f"{base_fn}.csv",
                           mime="text/csv")
        if long_df is not None:
            st.download_button("Download data (CSV, long)",
                               long_df.to_csv().encode("utf-8"),
                               file_name=f"{base_fn}_long.csv",
                               mime="text/csv")
        # Excel
        xls = to_excel_bytes({"wide": wide_df, "long": long_df})
        st.download_button("Download data (Excel)", xls,
                           file_name=f"{base_fn}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        # ZIP bundle
        st.download_button("Download bundle (ZIP)",
                           zip_bundle(img_bytes, wide_df, long_df),
                           file_name=f"{base_fn}.zip",
                           mime="application/zip")
    else:
        st.info("No data table found for this selection.")

    # ---------- Hot spots ----------
    st.markdown("### Hot spots")
    if wide_df is not None:
        max_val = float(np.nanmax(wide_df.to_numpy(dtype=float))) if not wide_df.empty else 0.0
        step = max(0.1, round(max_val/200, 2))
        thresh = st.slider("Highlight threshold", min_value=0.0, max_value=max(0.1, max_val),
                           value=0.0, step=step, key="phys_thresh")
        idx_name = wide_df.index.name or "macro_region"
        long_sorted = (wide_df.reset_index()
                                .rename(columns={idx_name: "Region"})
                                .melt(id_vars="Region", var_name="Sector", value_name="Value")
                                .dropna()
                                .sort_values("Value", ascending=False))
        if thresh > 0:
            long_sorted = long_sorted[long_sorted["Value"] > thresh]
        st.caption(f"Cells above threshold: **{int((wide_df > thresh).sum().sum())}**")
        N = st.slider("Show Top-N", 5, 30, 10, key="phys_topn")
        st.dataframe(long_sorted.head(N), use_container_width=True, height=240)

    st.markdown("---")
    st.header("Comparison")
    st.caption("Compare across scenarios (up to 4 tiles) for the selected layout/metric.")

    sel_scen = st.multiselect("Scenarios", PHYS_SCENARIOS, default=[scenario])

    # Build up to 4 tiles for the chosen layout/metric
    combos = sel_scen[:4]
    cols = st.columns(2)
    for i, s in enumerate(combos):
        # Resolve tile paths (now respects Layout for "Average")
        if metric_sel == "Average (HDW & SF)":
            if layout == "10 Categories":
                p = physical_paths_10cats_average(s)
                tile_title = f"{s} — Average (HDW & SF) — 10 Categories"
                img, _chosen = _pick_png(p)
            else:
                p = physical_paths_average(s)
                tile_title = f"{s} — Average (HDW & SF) — Regions × Sectors"
                img = cached_read_png_bytes(p["png"], file_mtime(p["png"]))
        else:
            storyline = "HDW" if metric_sel == "HDW" else "SF"
            if layout == "Regions x Sectors":
                p = physical_paths_regions(s, storyline)
                tile_title = f"{s} — {storyline} — Regions × Sectors"
                img = cached_read_png_bytes(p["png"], file_mtime(p["png"]))
            else:
                p = physical_paths_10cats(s, storyline)
                tile_title = f"{s} — {storyline} — 10 Categories"
                img, _chosen = physical_paths_10cats_pick_png(p)

        with cols[i % 2]:
            st.subheader(tile_title)
            if img:
                show_image(img)
                st.download_button("PNG", img,
                                   file_name=p.get("png_alt", p["png"]).name,
                                   mime="image/png", key=f"ppng_{i}")
            else:
                st.warning("PNG missing")
            d = cached_read_csv(p["data"], file_mtime(p["data"]))
            if d is not None:
                st.download_button("CSV (wide)",
                                   d.to_csv().encode("utf-8"),
                                   file_name=p["data"].name,
                                   mime="text/csv", key=f"pcsv_{i}")

    # ---------- Delta view (A − B) ----------
    st.markdown("### Delta view (A − B)")
    colA, colB = st.columns(2)
    with colA:
        A_s = st.selectbox("A: Scenario", PHYS_SCENARIOS, key="PA_s")
    with colB:
        B_s = st.selectbox("B: Scenario", PHYS_SCENARIOS, key="PB_s")

    if metric_sel == "Average (HDW & SF)":
        if layout == "10 Categories":
            A_paths = physical_paths_10cats_average(A_s)
            B_paths = physical_paths_10cats_average(B_s)
            delta_title = f"Δ (A − B): {A_s}/Average/10-cat minus {B_s}/Average/10-cat"
        else:
            A_paths = physical_paths_average(A_s)
            B_paths = physical_paths_average(B_s)
            delta_title = f"Δ (A − B): {A_s}/Average/R×S minus {B_s}/Average/R×S"
    else:
        storyline = "HDW" if metric_sel == "HDW" else "SF"
        if layout == "Regions x Sectors":
            A_paths = physical_paths_regions(A_s, storyline)
            B_paths = physical_paths_regions(B_s, storyline)
            delta_title = f"Δ (A − B): {A_s}/{storyline}/R×S minus {B_s}/{storyline}/R×S"
        else:
            A_paths = physical_paths_10cats(A_s, storyline)
            B_paths = physical_paths_10cats(B_s, storyline)
            delta_title = f"Δ (A − B): {A_s}/{storyline}/10-cat minus {B_s}/{storyline}/10-cat"

    A_df = cached_read_csv(A_paths["data"], file_mtime(A_paths["data"]))
    B_df = cached_read_csv(B_paths["data"], file_mtime(B_paths["data"]))

    if (A_df is not None) and (B_df is not None):
        common_rows = A_df.index.intersection(B_df.index)
        common_cols = A_df.columns.intersection(B_df.columns)
        delta = A_df.loc[common_rows, common_cols] - B_df.loc[common_rows, common_cols]
        figD = px.imshow(
            delta,
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            aspect="auto",
            labels=dict(x="Sector", y="Region", color="Δ A−B"),
            title=delta_title,
        )
        figD.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        figD.update_xaxes(side="bottom", tickangle=45)
        st.plotly_chart(figD, use_container_width=True)
        st.download_button(
            "Download Δ (CSV)",
            delta.to_csv().encode("utf-8"),
            file_name="physical_delta_A_minus_B.csv",
            mime="text/csv",
        )
    else:
        st.info("Both selections must have data to compute Δ.")

    st.caption("Share this view:")
    # add layout+metric to shareable URL
    st.code(share_url(risk=risk,
                      scenario=scenario,
                      layout=layout,
                      metric=metric_sel))
