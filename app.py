import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import io

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="SQC Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f8f9fb; }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] { font-size:0.78rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }
    [data-testid="stMetricValue"] { font-size:1.6rem; font-weight:700; color:#1e293b; }
    .section-header {
        font-size:1.05rem; font-weight:700; color:#1e293b;
        border-bottom:2px solid #3b82f6; display:inline-block;
        padding-bottom:4px; margin-bottom:8px;
    }
    .kpi-good { background:#dcfce7; color:#166534; border-radius:6px; padding:3px 10px; font-weight:700; }
    .kpi-warn { background:#fef9c3; color:#854d0e; border-radius:6px; padding:3px 10px; font-weight:700; }
    .kpi-bad  { background:#fee2e2; color:#991b1b; border-radius:6px; padding:3px 10px; font-weight:700; }
    .upload-box {
        border: 2px dashed #3b82f6; border-radius: 12px;
        padding: 32px; text-align: center; background: #eff6ff;
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────
P = {
    "blue":   "#3b82f6",
    "indigo": "#6366f1",
    "red":    "#ef4444",
    "amber":  "#f59e0b",
    "purple": "#a855f7",
    "slate":  "#64748b",
}

XR_CONST = {
    2:  {"A2":1.880,"D3":0.000,"D4":3.267,"d2":1.128},
    3:  {"A2":1.023,"D3":0.000,"D4":2.574,"d2":1.693},
    4:  {"A2":0.729,"D3":0.000,"D4":2.282,"d2":2.059},
    5:  {"A2":0.577,"D3":0.000,"D4":2.114,"d2":2.326},
    6:  {"A2":0.483,"D3":0.000,"D4":2.004,"d2":2.534},
    7:  {"A2":0.419,"D3":0.076,"D4":1.924,"d2":2.704},
    8:  {"A2":0.373,"D3":0.136,"D4":1.864,"d2":2.847},
    9:  {"A2":0.337,"D3":0.184,"D4":1.816,"d2":2.970},
    10: {"A2":0.308,"D3":0.223,"D4":1.777,"d2":3.078},
}

def c4(n): return 4*(n-1)/(4*n-3)

# ── Helper functions ──────────────────────────────────────────────
def get_ooc(values, ucl, lcl):
    return [i+1 for i, v in enumerate(values) if v > ucl or v < lcl]

def make_fig(nrows=2):
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 5*nrows), tight_layout=True)
    fig.patch.set_facecolor("#ffffff")
    for ax in (axes if nrows > 1 else [axes]):
        ax.set_facecolor("#f8f9fb")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#cbd5e1")
        ax.tick_params(colors="#475569", labelsize=9)
        for lbl in [ax.yaxis.label, ax.xaxis.label, ax.title]:
            lbl.set_color("#1e293b")
    return fig, axes

def draw_chart(ax, x, y, ucl, cl, lcl, ooc, color, series_label, ylabel):
    ax.plot(x, y, "-o", color=color, linewidth=1.6, markersize=4.5, label=series_label, zorder=3)
    ax.axhline(ucl, color=P["red"],   linewidth=1.5, linestyle="--", label=f"UCL = {ucl:.4f}")
    ax.axhline(cl,  color=P["slate"], linewidth=1.0, linestyle=":",  label=f"CL  = {cl:.4f}")
    ax.axhline(lcl, color=P["red"],   linewidth=1.5, linestyle="--", label=f"LCL = {lcl:.4f}")
    for b in ooc:
        ax.plot(x[b-1], y[b-1], "v", color=P["red"], markersize=11,
                zorder=5, markeredgecolor="white", markeredgewidth=1)
    rng = max(y)-min(y) if max(y) != min(y) else 0.01
    x_end = x[-1]
    ax.text(x_end+0.4, ucl+rng*0.02, f"UCL={ucl:.3f}", color=P["red"],   fontsize=8, va="bottom")
    ax.text(x_end+0.4, cl,            f"CL ={cl:.3f}",  color=P["slate"], fontsize=8, va="center")
    ax.text(x_end+0.4, lcl-rng*0.02, f"LCL={lcl:.3f}", color=P["red"],   fontsize=8, va="top")
    ax.set_xlim(0.5, x_end+2.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7, edgecolor="#e2e8f0")
    ax.grid(axis="y", linestyle=":", alpha=0.4, color="#94a3b8")

def cpk_badge(val):
    if val >= 1.33: return f'<span class="kpi-good">Cpk = {val:.3f} ✓ Capable</span>'
    if val >= 1.00: return f'<span class="kpi-warn">Cpk = {val:.3f} △ Marginal</span>'
    return f'<span class="kpi-bad">Cpk = {val:.3f} ✗ Not Capable</span>'

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 SQC Dashboard")
    st.markdown("Upload your dataset to begin analysis.")
    st.divider()

    # ── File upload ──
    st.markdown("### 📂 Upload Dataset")
    uploaded = st.file_uploader(
        "CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="File must contain a **Subgroup** column and one or more **measurement columns**."
    )

    # ── Download sample ──
    st.markdown("### 📋 Need a template?")
    sample_url = "https://raw.githubusercontent.com/ericlin0616/sqc-dashboard/main/sample_data.csv"
    st.markdown(f"[⬇️ Download sample_data.csv]({sample_url})")
    st.divider()

    st.markdown("### 📐 Chart Settings")
    chart_type = st.radio("Chart Type", ["X-bar & R Chart", "X-bar & S Chart"],
                          label_visibility="collapsed")
    st.divider()

    st.markdown("### 📏 Specification Limits")
    usl = st.number_input("USL", value=33.0, step=0.1, format="%.3f",
                          help="Upper Specification Limit")
    lsl = st.number_input("LSL", value=31.0, step=0.1, format="%.3f",
                          help="Lower Specification Limit")
    st.divider()

    st.markdown("### ⚙️ Column Mapping")
    st.caption("Configure after uploading a file.")

# ════════════════════════════════════════════════════════════════
# MAIN AREA — before upload
# ════════════════════════════════════════════════════════════════
st.markdown("# 🏭 Statistical Quality Control Dashboard")
st.divider()

if uploaded is None:
    st.markdown("""
    <div class="upload-box">
        <h3>📂 Upload your dataset to get started</h3>
        <p style="color:#475569; margin:0">
            Accepted formats: <strong>CSV</strong> or <strong>Excel (.xlsx / .xls)</strong><br><br>
            Your file should have:<br>
            • A <strong>Subgroup</strong> column (integer, e.g. 1, 2, 3 …)<br>
            • One or more <strong>measurement columns</strong> (numeric values)
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Expected file format")
    example = pd.DataFrame({
        "Subgroup": [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3],
        "Value":    [32.1,31.9,32.3,32.0,31.8,
                     32.5,32.2,32.7,32.1,32.4,
                     31.7,31.5,31.9,31.6,31.8]
    })
    st.dataframe(example, hide_index=True, use_container_width=False)
    st.caption("Each row = one measurement. All rows with the same Subgroup number form one subgroup.")

    # Provide a downloadable sample CSV right here too
    csv_sample = example.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Download sample_data.csv", csv_sample,
                       file_name="sample_data.csv", mime="text/csv")
    st.stop()

# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════
try:
    if uploaded.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded)
    else:
        raw_df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"❌ Could not read file: {e}")
    st.stop()

# ── Column mapping ────────────────────────────────────────────────
all_cols = raw_df.columns.tolist()
numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()

with st.sidebar:
    st.markdown("### ⚙️ Column Mapping")
    # Guess subgroup column
    default_sg = next((c for c in all_cols if "subgroup" in c.lower() or "batch" in c.lower() or "group" in c.lower()), all_cols[0])
    subgroup_col = st.selectbox("Subgroup column", all_cols,
                                index=all_cols.index(default_sg))
    # Guess value column
    remaining = [c for c in numeric_cols if c != subgroup_col]
    default_val = next((c for c in remaining if "value" in c.lower() or "measure" in c.lower() or "waist" in c.lower()), remaining[0] if remaining else numeric_cols[0])
    value_col = st.selectbox("Measurement column", remaining if remaining else numeric_cols,
                              index=0)

# ── Validate ──────────────────────────────────────────────────────
if subgroup_col == value_col:
    st.error("❌ Subgroup column and Measurement column must be different.")
    st.stop()

try:
    df = raw_df[[subgroup_col, value_col]].copy()
    df.columns = ["Subgroup", "Value"]
    df["Subgroup"] = df["Subgroup"].astype(int)
    df["Value"]    = pd.to_numeric(df["Value"], errors="coerce")
    df.dropna(inplace=True)
except Exception as e:
    st.error(f"❌ Column error: {e}")
    st.stop()

if df.empty:
    st.error("❌ No valid data found after parsing. Check your column selections.")
    st.stop()

# ── Subgroup stats ────────────────────────────────────────────────
sg = df.groupby("Subgroup")["Value"].agg(
    xbar="mean",
    R=lambda x: x.max()-x.min(),
    S="std",
    n="count"
).reset_index()

# Detect subgroup size (use most common n)
n = int(sg["n"].mode()[0])
all_vals = df["Value"].values
batches   = sg["Subgroup"].tolist()
xbar_vals = sg["xbar"].tolist()
xbar_bar  = float(np.mean(xbar_vals))

# ── Control limits ────────────────────────────────────────────────
if chart_type == "X-bar & R Chart":
    if n in XR_CONST:
        c = XR_CONST[n]
        A2, D3, D4, d2 = c["A2"], c["D3"], c["D4"], c["d2"]
    else:
        # fallback for large n: use S chart constants implicitly
        st.warning(f"⚠️ n={n} is outside R Chart lookup table (n=2–10). Switching to S Chart constants.")
        chart_type = "X-bar & S Chart"

if chart_type == "X-bar & R Chart":
    R_vals = sg["R"].tolist(); R_bar = float(np.mean(R_vals))
    sigma_hat = R_bar / d2
    UCLx = xbar_bar + A2*R_bar;  LCLx = xbar_bar - A2*R_bar
    UCLv = D4*R_bar;              LCLv = D3*R_bar
    ooc_x = get_ooc(xbar_vals, UCLx, LCLx)
    ooc_v = get_ooc(R_vals,    UCLv, LCLv)
    var_label="Range R"; var_vals=R_vals; cl_var=R_bar; var_color=P["indigo"]
else:
    _c4 = c4(n); A3 = 3/(_c4*np.sqrt(n))
    B3 = max(0.0, 1 - 3/(_c4*np.sqrt(2*(n-1)))); B4 = 1 + 3/(_c4*np.sqrt(2*(n-1)))
    S_vals = sg["S"].tolist(); S_bar = float(np.mean(S_vals))
    sigma_hat = S_bar / _c4
    UCLx = xbar_bar + A3*S_bar;  LCLx = xbar_bar - A3*S_bar
    UCLv = B4*S_bar;              LCLv = B3*S_bar
    ooc_x = get_ooc(xbar_vals, UCLx, LCLx)
    ooc_v = get_ooc(S_vals,    UCLv, LCLv)
    var_label="Std Dev S"; var_vals=S_vals; cl_var=S_bar; var_color=P["purple"]

# ── Process capability ────────────────────────────────────────────
cp  = (usl-lsl)/(6*sigma_hat) if sigma_hat>0 else 0
cpu = (usl-xbar_bar)/(3*sigma_hat) if sigma_hat>0 else 0
cpl = (xbar_bar-lsl)/(3*sigma_hat) if sigma_hat>0 else 0
cpk = min(cpu, cpl)

# ════════════════════════════════════════════════════════════════
# KPI ROW
# ════════════════════════════════════════════════════════════════
st.markdown(
    f"**File:** `{uploaded.name}` &nbsp;|&nbsp; **Chart:** {chart_type} &nbsp;|&nbsp; "
    f"**Subgroups:** {len(batches)} &nbsp;|&nbsp; **Subgroup size n:** {n} &nbsp;|&nbsp; "
    f"**Total measurements:** {len(df)}"
)
st.divider()

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Grand Mean X̄",   f"{xbar_bar:.4f}")
k2.metric("Est. Std Dev σ̂", f"{sigma_hat:.4f}")
k3.metric("UCL (X̄)",        f"{UCLx:.4f}")
k4.metric("LCL (X̄)",        f"{LCLx:.4f}")
k5.metric("Cp",              f"{cp:.3f}")
k6.metric("Cpk",             f"{cpk:.3f}",
          delta="Capable" if cpk>=1.33 else ("Marginal" if cpk>=1.0 else "Not Capable"),
          delta_color="normal" if cpk>=1.33 else "inverse")

st.divider()

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈  Control Charts", "📊  Process Capability", "🗂️  Data Table"])

# ── Tab 1: Control Charts ─────────────────────────────────────────
with tab1:
    fig, (ax1, ax2) = make_fig(2)
    fig.suptitle(
        f"{chart_type}  |  {uploaded.name}  |  n={n},  Subgroups={len(batches)}",
        fontsize=12, fontweight="bold", color="#1e293b", y=1.01)

    draw_chart(ax1, batches, xbar_vals, UCLx, xbar_bar, LCLx, ooc_x,
               P["blue"], "Subgroup Mean", f"Mean {value_col}")
    ax1.set_title("X-bar Chart", fontsize=11, fontweight="bold")

    draw_chart(ax2, batches, var_vals, UCLv, cl_var, LCLv, ooc_v,
               var_color, f"Subgroup {var_label}", f"{var_label}")
    ax2.set_title(f"{var_label} Chart", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Subgroup Number", fontsize=10)

    st.pyplot(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-header">X-bar Control Limits</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Limit": ["UCL","CL","LCL"],
            "Value": [f"{UCLx:.4f}", f"{xbar_bar:.4f}", f"{LCLx:.4f}"]
        }), hide_index=True, use_container_width=True)
        st.error(f"⚠️ OOC subgroups: {ooc_x}") if ooc_x else st.success("✅ All in control")

    with col_b:
        st.markdown(f'<p class="section-header">{var_label} Control Limits</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Limit": ["UCL","CL","LCL"],
            "Value": [f"{UCLv:.4f}", f"{cl_var:.4f}", f"{LCLv:.4f}"]
        }), hide_index=True, use_container_width=True)
        st.error(f"⚠️ OOC subgroups: {ooc_v}") if ooc_v else st.success("✅ All in control")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button("⬇️  Download Chart (PNG)", buf.getvalue(),
                       file_name="control_chart.png", mime="image/png")

# ── Tab 2: Process Capability ─────────────────────────────────────
with tab2:
    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        st.markdown('<p class="section-header">Capability Indices</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Index": ["USL","LSL","X̄","σ̂","Cp","Cpu","Cpl","Cpk"],
            "Value": [f"{usl:.4f}", f"{lsl:.4f}",
                      f"{xbar_bar:.4f}", f"{sigma_hat:.4f}",
                      f"{cp:.3f}", f"{cpu:.3f}", f"{cpl:.3f}", f"{cpk:.3f}"]
        }), hide_index=True, use_container_width=True)

        st.markdown("**Verdict:**")
        st.markdown(cpk_badge(cpk), unsafe_allow_html=True)
        st.markdown("""
        <small style='color:#64748b'>
        Cpk ≥ 1.33 → Capable &nbsp;|&nbsp;
        1.00 ≤ Cpk < 1.33 → Marginal &nbsp;|&nbsp;
        Cpk < 1.00 → Not Capable
        </small>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<p class="section-header">Process Distribution</p>', unsafe_allow_html=True)

        fig2, ax = plt.subplots(figsize=(8, 4.5), tight_layout=True)
        fig2.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8f9fb")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#cbd5e1")

        ax.hist(all_vals, bins=min(30, len(all_vals)//3+1),
                color=P["blue"], alpha=0.5, edgecolor="white",
                linewidth=0.6, density=True, label="Measurements")
        x_r = np.linspace(all_vals.min()-0.5, all_vals.max()+0.5, 300)
        pdf = stats.norm.pdf(x_r, xbar_bar, sigma_hat)
        ax.plot(x_r, pdf, color=P["indigo"], linewidth=2.2,
                label=f"Normal fit  μ={xbar_bar:.3f}, σ={sigma_hat:.3f}")
        ax.axvline(usl, color=P["red"],   linewidth=1.8, linestyle="--", label=f"USL={usl:.3f}")
        ax.axvline(lsl, color=P["red"],   linewidth=1.8, linestyle="--", label=f"LSL={lsl:.3f}")
        ax.axvline(xbar_bar, color=P["amber"], linewidth=1.5, linestyle=":",
                   label=f"X̄={xbar_bar:.3f}")
        ax.fill_betweenx([0, pdf.max()*1.15], lsl-2, lsl, color=P["red"], alpha=0.07)
        ax.fill_betweenx([0, pdf.max()*1.15], usl, usl+2, color=P["red"], alpha=0.07)

        ax.set_xlabel(f"{value_col}", fontsize=10, color="#475569")
        ax.set_ylabel("Density", fontsize=10, color="#475569")
        ax.set_title("Distribution with Spec Limits", fontsize=11,
                     fontweight="bold", color="#1e293b")
        ax.legend(fontsize=8, framealpha=0.8, edgecolor="#e2e8f0")
        ax.tick_params(colors="#475569", labelsize=9)
        st.pyplot(fig2, use_container_width=True)

        pct_above = (1 - stats.norm.cdf(usl, xbar_bar, sigma_hat)) * 100
        pct_below = stats.norm.cdf(lsl, xbar_bar, sigma_hat) * 100
        dppm = (pct_above + pct_below) * 10000
        d1, d2_, d3 = st.columns(3)
        d1.metric("Est. % Above USL", f"{pct_above:.3f}%")
        d2_.metric("Est. % Below LSL", f"{pct_below:.3f}%")
        d3.metric("Est. DPPM", f"{dppm:,.0f}")

# ── Tab 3: Data Table ─────────────────────────────────────────────
with tab3:
    st.markdown('<p class="section-header">Subgroup Statistics</p>', unsafe_allow_html=True)
    disp = sg[["Subgroup","xbar","R","S"]].copy()
    disp.columns = ["Subgroup","Mean X̄","Range R","Std Dev S"]
    disp["Status"] = disp["Subgroup"].apply(
        lambda b: "⚠️ OOC" if b in ooc_x or b in ooc_v else "✅ OK")
    for col in ["Mean X̄","Range R","Std Dev S"]:
        disp[col] = disp[col].round(4)
    st.dataframe(disp, hide_index=True, use_container_width=True, height=420)

    csv_out = disp.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Download Subgroup Stats (CSV)", csv_out,
                       file_name="subgroup_statistics.csv", mime="text/csv")

    st.divider()
    st.markdown('<p class="section-header">Raw Uploaded Data</p>', unsafe_allow_html=True)
    st.dataframe(raw_df, hide_index=True, use_container_width=True, height=350)

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"File: {uploaded.name} · Subgroup col: '{subgroup_col}' · "
    f"Measurement col: '{value_col}' · LSL={lsl} · USL={usl} · "
    f"Chart: {chart_type}"
)
