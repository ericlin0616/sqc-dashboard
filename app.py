import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import io

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="SQC Dashboard | Jeans Waist Process",
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
    [data-testid="stMetricLabel"]  { font-size:0.78rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; }
    [data-testid="stMetricValue"]  { font-size:1.6rem;  font-weight:700; color:#1e293b; }
    .section-header {
        font-size:1.05rem; font-weight:700; color:#1e293b;
        border-bottom:2px solid #3b82f6; display:inline-block;
        padding-bottom:4px; margin-bottom:8px;
    }
    .kpi-good { background:#dcfce7; color:#166534; border-radius:6px; padding:3px 10px; font-weight:700; }
    .kpi-warn { background:#fef9c3; color:#854d0e; border-radius:6px; padding:3px 10px; font-weight:700; }
    .kpi-bad  { background:#fee2e2; color:#991b1b; border-radius:6px; padding:3px 10px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
ABNORMAL_BASE = [15, 16, 17, 32, 33]
P = {
    "blue":   "#3b82f6",
    "indigo": "#6366f1",
    "green":  "#22c55e",
    "red":    "#ef4444",
    "amber":  "#f59e0b",
    "purple": "#a855f7",
    "slate":  "#64748b",
}
XR_CONST = {
    3: {"A2":1.023,"D3":0.000,"D4":2.574,"d2":1.693},
    4: {"A2":0.729,"D3":0.000,"D4":2.282,"d2":2.059},
    5: {"A2":0.577,"D3":0.000,"D4":2.114,"d2":2.326},
}

def c4(n): return 4*(n-1)/(4*n-3)

# ── Helpers ───────────────────────────────────────────────────────
def generate_data(mean, std, shift, n, subgroups):
    abnormal = [b for b in ABNORMAL_BASE if b <= subgroups]
    records = []
    for b in range(1, subgroups+1):
        m = mean+shift if b in abnormal else mean
        for v in np.random.normal(m, std, n):
            records.append({"Subgroup": b, "Value": round(float(v), 4)})
    return pd.DataFrame(records), abnormal

def get_ooc(values, ucl, lcl):
    return [i+1 for i,v in enumerate(values) if v > ucl or v < lcl]

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
    ax.text(x_end+0.6, ucl+rng*0.02, f"UCL={ucl:.3f}", color=P["red"],   fontsize=8, va="bottom")
    ax.text(x_end+0.6, cl,            f"CL ={cl:.3f}",  color=P["slate"], fontsize=8, va="center")
    ax.text(x_end+0.6, lcl-rng*0.02, f"LCL={lcl:.3f}", color=P["red"],   fontsize=8, va="top")
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
    st.markdown("Jeans Waist Process Control")
    st.divider()

    st.markdown("### 📐 Chart Type")
    chart_type = st.radio("", ["X-bar & R Chart", "X-bar & S Chart"],
                          label_visibility="collapsed")
    st.divider()

    st.markdown("### ⚙️ Process Parameters")
    mean  = st.slider("Target Mean (inch)",         30.0, 34.0, 32.0, 0.1)
    std   = st.slider("Process Std Dev (inch)",      0.05,  1.0,  0.3, 0.05)
    shift = st.slider("Assignable Cause Shift (inch)", 0.0, 3.0,  1.0, 0.1)
    st.divider()

    st.markdown("### 🔢 Sampling Plan")
    if chart_type == "X-bar & R Chart":
        n         = st.slider("Subgroup Size n  (3–5)",      3,  5,  5, 1)
        subgroups = st.slider("Number of Subgroups",        20, 40, 40, 1)
    else:
        n         = st.slider("Subgroup Size n  (>10)",     10, 30, 12, 1)
        subgroups = st.slider("Number of Subgroups",        20, 60, 40, 1)
    st.divider()

    st.markdown("### 📏 Specification Limits")
    usl = st.number_input("USL (inch)", value=float(round(mean+1.0, 1)), step=0.1, format="%.2f")
    lsl = st.number_input("LSL (inch)", value=float(round(mean-1.0, 1)), step=0.1, format="%.2f")
    st.divider()

    regenerate = st.button("🔄  Regenerate Simulation", use_container_width=True, type="primary")
    abn_shown = [b for b in ABNORMAL_BASE if b <= subgroups]
    st.caption(f"Injected OOC subgroups: {abn_shown}")

# ════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════
if "df" not in st.session_state or regenerate:
    st.session_state.df, st.session_state.abnormal = generate_data(
        mean, std, shift, n, subgroups)

df, abnormal = st.session_state.df, st.session_state.abnormal
all_vals = df["Value"].values

sg = df.groupby("Subgroup")["Value"].agg(
    xbar="mean", R=lambda x: x.max()-x.min(), S="std"
).reset_index()

batches   = sg["Subgroup"].tolist()
xbar_vals = sg["xbar"].tolist()
xbar_bar  = float(np.mean(xbar_vals))

# Limits
if chart_type == "X-bar & R Chart":
    c   = XR_CONST[n]
    R_vals = sg["R"].tolist(); R_bar = float(np.mean(R_vals))
    sigma_hat = R_bar / c["d2"]
    UCLx=xbar_bar+c["A2"]*R_bar; LCLx=xbar_bar-c["A2"]*R_bar
    UCLv=c["D4"]*R_bar;          LCLv=c["D3"]*R_bar
    ooc_x=get_ooc(xbar_vals,UCLx,LCLx); ooc_v=get_ooc(R_vals,UCLv,LCLv)
    var_label="Range R"; var_vals=R_vals; cl_var=R_bar; var_color=P["indigo"]
else:
    _c4=c4(n); A3=3/(_c4*np.sqrt(n))
    B3=max(0.0,1-3/(_c4*np.sqrt(2*(n-1)))); B4=1+3/(_c4*np.sqrt(2*(n-1)))
    S_vals=sg["S"].tolist(); S_bar=float(np.mean(S_vals))
    sigma_hat=S_bar/_c4
    UCLx=xbar_bar+A3*S_bar; LCLx=xbar_bar-A3*S_bar
    UCLv=B4*S_bar;           LCLv=B3*S_bar
    ooc_x=get_ooc(xbar_vals,UCLx,LCLx); ooc_v=get_ooc(S_vals,UCLv,LCLv)
    var_label="Std Dev S"; var_vals=S_vals; cl_var=S_bar; var_color=P["purple"]

cp  = (usl-lsl)/(6*sigma_hat) if sigma_hat>0 else 0
cpu = (usl-xbar_bar)/(3*sigma_hat) if sigma_hat>0 else 0
cpl = (xbar_bar-lsl)/(3*sigma_hat) if sigma_hat>0 else 0
cpk = min(cpu, cpl)

# ════════════════════════════════════════════════════════════════
# HEADER + KPIs
# ════════════════════════════════════════════════════════════════
st.markdown("# 🏭 Statistical Quality Control Dashboard")
st.markdown(
    f"**Process:** Jeans waist &nbsp;|&nbsp; **Chart:** {chart_type} &nbsp;|&nbsp; "
    f"**n = {n}** &nbsp;|&nbsp; **Subgroups = {subgroups}** &nbsp;|&nbsp; "
    f"**Total units = {n*subgroups}**"
)
st.divider()

c1,c2,c3,c4_,c5,c6 = st.columns(6)
c1.metric("Grand Mean X̄",    f"{xbar_bar:.4f} in")
c2.metric("Est. Std Dev σ̂",  f"{sigma_hat:.4f} in")
c3.metric("UCL (X̄)",         f"{UCLx:.4f}")
c4_.metric("LCL (X̄)",        f"{LCLx:.4f}")
c5.metric("Cp",               f"{cp:.3f}")
c6.metric("Cpk",              f"{cpk:.3f}",
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
        f"{chart_type}  |  Target={mean} in,  σ={std} in,  Shift={shift} in",
        fontsize=12, fontweight="bold", color="#1e293b", y=1.01)

    draw_chart(ax1, batches, xbar_vals, UCLx, xbar_bar, LCLx, ooc_x,
               P["blue"], "Subgroup Mean", "Mean Waist (inch)")
    ax1.set_title("X-bar Chart", fontsize=11, fontweight="bold")

    draw_chart(ax2, batches, var_vals, UCLv, cl_var, LCLv, ooc_v,
               var_color, f"Subgroup {var_label}", f"{var_label} (inch)")
    ax2.set_title(f"{var_label} Chart", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Subgroup Number", fontsize=10)

    st.pyplot(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="section-header">X-bar Control Limits</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Limit":["UCL","CL","LCL"],
            "Value":[f"{UCLx:.4f}", f"{xbar_bar:.4f}", f"{LCLx:.4f}"]
        }), hide_index=True, use_container_width=True)
        st.error(f"⚠️ OOC subgroups: {ooc_x}") if ooc_x else st.success("✅ All in control")

    with col_b:
        st.markdown(f'<p class="section-header">{var_label} Control Limits</p>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Limit":["UCL","CL","LCL"],
            "Value":[f"{UCLv:.4f}", f"{cl_var:.4f}", f"{LCLv:.4f}"]
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
            "Index":["USL","LSL","X̄","σ̂","Cp","Cpu","Cpl","Cpk"],
            "Value":[f"{usl:.4f} in", f"{lsl:.4f} in",
                     f"{xbar_bar:.4f} in", f"{sigma_hat:.4f} in",
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

        ax.hist(all_vals, bins=25, color=P["blue"], alpha=0.5,
                edgecolor="white", linewidth=0.6, density=True, label="Measurements")
        x_r = np.linspace(all_vals.min()-0.5, all_vals.max()+0.5, 300)
        pdf = stats.norm.pdf(x_r, xbar_bar, sigma_hat)
        ax.plot(x_r, pdf, color=P["indigo"], linewidth=2.2,
                label=f"Normal fit  μ={xbar_bar:.3f}, σ={sigma_hat:.3f}")
        ax.axvline(usl, color=P["red"],   linewidth=1.8, linestyle="--", label=f"USL={usl:.3f}")
        ax.axvline(lsl, color=P["red"],   linewidth=1.8, linestyle="--", label=f"LSL={lsl:.3f}")
        ax.axvline(xbar_bar, color=P["amber"], linewidth=1.5, linestyle=":",
                   label=f"X̄={xbar_bar:.3f}")
        ax.fill_betweenx([0, pdf.max()*1.15], lsl-1, lsl,  color=P["red"], alpha=0.07)
        ax.fill_betweenx([0, pdf.max()*1.15], usl, usl+1,  color=P["red"], alpha=0.07)

        ax.set_xlabel("Waist Measurement (inch)", fontsize=10, color="#475569")
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
    disp = sg.copy()
    disp.columns = ["Subgroup","Mean X̄","Range R","Std Dev S"]
    disp["Status"] = disp["Subgroup"].apply(
        lambda b: "⚠️ OOC" if b in ooc_x or b in ooc_v else "✅ OK")
    for col in ["Mean X̄","Range R","Std Dev S"]:
        disp[col] = disp[col].round(4)
    st.dataframe(disp, hide_index=True, use_container_width=True, height=420)

    csv = disp.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Download CSV", csv,
                       file_name="subgroup_statistics.csv", mime="text/csv")

    st.divider()
    st.markdown('<p class="section-header">Raw Measurements (first 50 rows)</p>', unsafe_allow_html=True)
    st.dataframe(df.head(50), hide_index=True, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────
st.divider()
abn_str = ", ".join(str(b) for b in abnormal) if abnormal else "None"
st.caption(
    f"Simulation · Mean={mean} in · σ={std} in · Shift={shift} in · "
    f"LSL={lsl:.2f} · USL={usl:.2f} · Injected OOC: {abn_str}"
)
