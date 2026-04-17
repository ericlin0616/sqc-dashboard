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
        margin: 16px 0;
    }
    .mode-badge-sim  { background:#dbeafe; color:#1e40af; border-radius:8px; padding:4px 14px; font-weight:700; font-size:0.9rem; }
    .mode-badge-data { background:#dcfce7; color:#166534; border-radius:8px; padding:4px 14px; font-weight:700; font-size:0.9rem; }
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

ABNORMAL_BASE = [15, 16, 17, 32, 33]

def c4(n): return 4*(n-1)/(4*n-3)

# ── Shared helpers ────────────────────────────────────────────────
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

def compute_limits(chart_type, xbar_bar, n, sg):
    if chart_type == "X-bar & R Chart":
        if n in XR_CONST:
            c = XR_CONST[n]
            R_vals = sg["R"].tolist(); R_bar = float(np.mean(R_vals))
            sigma_hat = R_bar / c["d2"]
            UCLx = xbar_bar + c["A2"]*R_bar; LCLx = xbar_bar - c["A2"]*R_bar
            UCLv = c["D4"]*R_bar;             LCLv = c["D3"]*R_bar
            ooc_x = get_ooc(sg["xbar"].tolist(), UCLx, LCLx)
            ooc_v = get_ooc(R_vals, UCLv, LCLv)
            return dict(sigma_hat=sigma_hat, UCLx=UCLx, LCLx=LCLx, UCLv=UCLv, LCLv=LCLv,
                        ooc_x=ooc_x, ooc_v=ooc_v, var_label="Range R",
                        var_vals=R_vals, cl_var=R_bar, var_color=P["indigo"],
                        chart_type=chart_type)
        else:
            chart_type = "X-bar & S Chart"  # fallback

    _c4 = c4(n); A3 = 3/(_c4*np.sqrt(n))
    B3 = max(0.0, 1-3/(_c4*np.sqrt(2*(n-1)))); B4 = 1+3/(_c4*np.sqrt(2*(n-1)))
    S_vals = sg["S"].tolist(); S_bar = float(np.mean(S_vals))
    sigma_hat = S_bar / _c4
    UCLx = xbar_bar + A3*S_bar; LCLx = xbar_bar - A3*S_bar
    UCLv = B4*S_bar;             LCLv = B3*S_bar
    ooc_x = get_ooc(sg["xbar"].tolist(), UCLx, LCLx)
    ooc_v = get_ooc(S_vals, UCLv, LCLv)
    return dict(sigma_hat=sigma_hat, UCLx=UCLx, LCLx=LCLx, UCLv=UCLv, LCLv=LCLv,
                ooc_x=ooc_x, ooc_v=ooc_v, var_label="Std Dev S",
                var_vals=S_vals, cl_var=S_bar, var_color=P["purple"],
                chart_type=chart_type)

def render_tabs(df, sg, lim, all_vals, usl, lsl, n, chart_type, value_col, subtitle):
    batches   = sg["Subgroup"].tolist()
    xbar_vals = sg["xbar"].tolist()
    xbar_bar  = float(np.mean(xbar_vals))

    cp  = (usl-lsl)/(6*lim["sigma_hat"]) if lim["sigma_hat"]>0 else 0
    cpu = (usl-xbar_bar)/(3*lim["sigma_hat"]) if lim["sigma_hat"]>0 else 0
    cpl = (xbar_bar-lsl)/(3*lim["sigma_hat"]) if lim["sigma_hat"]>0 else 0
    cpk = min(cpu, cpl)

    # KPIs
    sigma_val = lim["sigma_hat"]
    UCLx_val  = lim["UCLx"]
    LCLx_val  = lim["LCLx"]
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Grand Mean X̄",   f"{xbar_bar:.4f}")
    k2.metric("Est. Std Dev σ̂", f"{sigma_val:.4f}")
    k3.metric("UCL (X̄)",        f"{UCLx_val:.4f}")
    k4.metric("LCL (X̄)",        f"{LCLx_val:.4f}")
    k5.metric("Cp",              f"{cp:.3f}")
    k6.metric("Cpk",             f"{cpk:.3f}",
              delta="Capable" if cpk>=1.33 else ("Marginal" if cpk>=1.0 else "Not Capable"),
              delta_color="normal" if cpk>=1.33 else "inverse")
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📈  Control Charts", "📊  Process Capability", "🗂️  Data Table", "📦  Box Plot"])

    # ── Tab 1 ──
    with tab1:
        fig, (ax1, ax2) = make_fig(2)
        fig.suptitle(subtitle, fontsize=12, fontweight="bold", color="#1e293b", y=1.01)

        draw_chart(ax1, batches, xbar_vals,
                   lim["UCLx"], xbar_bar, lim["LCLx"], lim["ooc_x"],
                   P["blue"], "Subgroup Mean", f"Mean {value_col}")
        ax1.set_title("X-bar Chart", fontsize=11, fontweight="bold")

        vl = lim["var_label"]
        draw_chart(ax2, batches, lim["var_vals"],
                   lim["UCLv"], lim["cl_var"], lim["LCLv"], lim["ooc_v"],
                   lim["var_color"], "Subgroup " + vl, vl)
        ax2.set_title(vl + " Chart", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Subgroup Number", fontsize=10)

        st.pyplot(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<p class="section-header">X-bar Control Limits</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Limit":["UCL","CL","LCL"],
                "Value":[f"{UCLx_val:.4f}", f"{xbar_bar:.4f}", f"{LCLx_val:.4f}"]
            }), hide_index=True, use_container_width=True)
            ooc_x_list = lim["ooc_x"]
            if ooc_x_list:
                st.error("OOC subgroups (X-bar): " + str(ooc_x_list))
            else:
                st.success("All X-bar subgroups in control")

        with col_b:
            var_label_str = lim["var_label"]
            st.markdown(f'<p class="section-header">{var_label_str} Control Limits</p>', unsafe_allow_html=True)
            UCLv_val = lim["UCLv"]; cl_var_val = lim["cl_var"]; LCLv_val = lim["LCLv"]
            st.dataframe(pd.DataFrame({
                "Limit":["UCL","CL","LCL"],
                "Value":[f"{UCLv_val:.4f}", f"{cl_var_val:.4f}", f"{LCLv_val:.4f}"]
            }), hide_index=True, use_container_width=True)
            ooc_v_list = lim["ooc_v"]
            if ooc_v_list:
                st.error("OOC subgroups (" + var_label_str + "): " + str(ooc_v_list))
            else:
                st.success("All " + var_label_str + " subgroups in control")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        st.download_button("⬇️  Download Chart (PNG)", buf.getvalue(),
                           file_name="control_chart.png", mime="image/png")

    # ── Tab 2 ──
    with tab2:
        col_l, col_r = st.columns([1, 1.6])
        with col_l:
            st.markdown('<p class="section-header">Capability Indices</p>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Index":["USL","LSL","X̄","σ̂","Cp","Cpu","Cpl","Cpk"],
                "Value":[f"{usl:.4f}", f"{lsl:.4f}",
                         f"{xbar_bar:.4f}", f"{sigma_val:.4f}",
                         f"{cp:.3f}", f"{cpu:.3f}", f"{cpl:.3f}", f"{cpk:.3f}"]
            }), hide_index=True, use_container_width=True)
            st.markdown("**Verdict:**")
            st.markdown(cpk_badge(cpk), unsafe_allow_html=True)
            st.markdown("""<small style='color:#64748b'>
            Cpk ≥ 1.33 → Capable &nbsp;|&nbsp;
            1.00 ≤ Cpk &lt; 1.33 → Marginal &nbsp;|&nbsp;
            Cpk &lt; 1.00 → Not Capable
            </small>""", unsafe_allow_html=True)

        with col_r:
            st.markdown('<p class="section-header">Process Distribution</p>', unsafe_allow_html=True)
            fig2, ax = plt.subplots(figsize=(8, 4.5), tight_layout=True)
            fig2.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#f8f9fb")
            ax.spines[["top","right"]].set_visible(False)
            ax.spines[["left","bottom"]].set_color("#cbd5e1")

            ax.hist(all_vals, bins=min(30, max(5, len(all_vals)//5)),
                    color=P["blue"], alpha=0.5, edgecolor="white",
                    linewidth=0.6, density=True, label="Measurements")
            x_r = np.linspace(all_vals.min()-0.5, all_vals.max()+0.5, 300)
            pdf = stats.norm.pdf(x_r, xbar_bar, sigma_val)
            ax.plot(x_r, pdf, color=P["indigo"], linewidth=2.2,
                    label=f"Normal fit  mu={xbar_bar:.3f}, sigma={sigma_val:.3f}")
            ax.axvline(usl, color=P["red"],   linewidth=1.8, linestyle="--", label=f"USL={usl:.3f}")
            ax.axvline(lsl, color=P["red"],   linewidth=1.8, linestyle="--", label=f"LSL={lsl:.3f}")
            ax.axvline(xbar_bar, color=P["amber"], linewidth=1.5, linestyle=":",
                       label=f"X̄={xbar_bar:.3f}")
            ax.fill_betweenx([0, pdf.max()*1.15], lsl-2, lsl, color=P["red"], alpha=0.07)
            ax.fill_betweenx([0, pdf.max()*1.15], usl, usl+2, color=P["red"], alpha=0.07)
            ax.set_xlabel(value_col, fontsize=10, color="#475569")
            ax.set_ylabel("Density", fontsize=10, color="#475569")
            ax.set_title("Distribution with Spec Limits", fontsize=11,
                         fontweight="bold", color="#1e293b")
            ax.legend(fontsize=8, framealpha=0.8, edgecolor="#e2e8f0")
            ax.tick_params(colors="#475569", labelsize=9)
            st.pyplot(fig2, use_container_width=True)

            pct_above = (1 - stats.norm.cdf(usl, xbar_bar, sigma_val)) * 100
            pct_below = stats.norm.cdf(lsl, xbar_bar, sigma_val) * 100
            dppm = (pct_above + pct_below) * 10000
            d1, d2_, d3 = st.columns(3)
            d1.metric("Est. % Above USL", f"{pct_above:.3f}%")
            d2_.metric("Est. % Below LSL", f"{pct_below:.3f}%")
            d3.metric("Est. DPPM", f"{dppm:,.0f}")

    # ── Tab 3 ──
    with tab3:
        st.markdown('<p class="section-header">Subgroup Statistics</p>', unsafe_allow_html=True)
        disp = sg[["Subgroup","xbar","R","S"]].copy()
        disp.columns = ["Subgroup","Mean X̄","Range R","Std Dev S"]
        disp["Status"] = disp["Subgroup"].apply(
            lambda b: "⚠️ OOC" if b in lim["ooc_x"] or b in lim["ooc_v"] else "✅ OK")
        for col in ["Mean X̄","Range R","Std Dev S"]:
            disp[col] = disp[col].round(4)
        st.dataframe(disp, hide_index=True, use_container_width=True, height=420)
        csv_out = disp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️  Download Subgroup Stats (CSV)", csv_out,
                           file_name="subgroup_statistics.csv", mime="text/csv")
        st.divider()
        st.markdown('<p class="section-header">Raw Data</p>', unsafe_allow_html=True)
        st.dataframe(df, hide_index=True, use_container_width=True, height=320)

    # ── Tab 4: Box Plot ──
    with tab4:
        st.markdown('<p class="section-header">Box Plot by Subgroup</p>', unsafe_allow_html=True)

        # Options row
        col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 2])
        with col_opt1:
            show_points = st.checkbox("Show individual points", value=True)
        with col_opt2:
            show_mean = st.checkbox("Show subgroup mean", value=True)
        with col_opt3:
            max_sg_display = st.slider("Max subgroups to display", 10, min(60, len(batches)), min(40, len(batches)), 5)

        # Subset batches for readability
        display_batches = batches[:max_sg_display]
        batch_data = [df.loc[df["Subgroup"] == b, "Value"].values for b in display_batches]

        # Colour each box: red if OOC, blue otherwise
        ooc_set = set(lim["ooc_x"]) | set(lim["ooc_v"])
        box_colors = [P["red"] if b in ooc_set else P["blue"] for b in display_batches]

        # Dynamic figure width
        fig_w = max(12, len(display_batches) * 0.45)
        fig3, ax = plt.subplots(figsize=(fig_w, 5), tight_layout=True)
        fig3.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8f9fb")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd5e1")

        bp = ax.boxplot(
            batch_data,
            positions=display_batches,
            widths=0.6,
            patch_artist=True,
            medianprops=dict(color="#ffffff", linewidth=2),
            whiskerprops=dict(color="#94a3b8", linewidth=1.2),
            capprops=dict(color="#94a3b8", linewidth=1.2),
            flierprops=dict(marker="o", markersize=3,
                            markerfacecolor=P["amber"], markeredgecolor="none", alpha=0.7),
        )

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Individual points (jitter)
        if show_points:
            for b, data in zip(display_batches, batch_data):
                jitter = np.random.uniform(-0.18, 0.18, size=len(data))
                ax.scatter(b + jitter, data, s=18, color=P["slate"],
                           alpha=0.45, zorder=3, linewidths=0)

        # Subgroup means
        if show_mean:
            means = [d.mean() for d in batch_data]
            ax.plot(display_batches, means, "D", color=P["amber"],
                    markersize=5, zorder=5, label="Subgroup mean")

        # Grand mean & spec lines
        ax.axhline(xbar_bar, color=P["blue"],  linewidth=1.2, linestyle="--",
                   label=f"Grand mean = {xbar_bar:.3f}")
        ax.axhline(usl, color=P["red"],  linewidth=1.4, linestyle="--", label=f"USL = {usl:.3f}")
        ax.axhline(lsl, color=P["red"],  linewidth=1.4, linestyle="--", label=f"LSL = {lsl:.3f}")

        # Shade OOC boxes lightly
        for b in display_batches:
            if b in ooc_set:
                ax.axvspan(b - 0.4, b + 0.4, color=P["red"], alpha=0.06, zorder=0)

        ax.set_xlabel("Subgroup Number", fontsize=10, color="#475569")
        ax.set_ylabel(value_col, fontsize=10, color="#475569")
        ax.set_title(
            f"Box Plot by Subgroup  |  n={n}  |  Red = OOC subgroup",
            fontsize=11, fontweight="bold", color="#1e293b"
        )
        ax.set_xticks(display_batches)
        ax.set_xticklabels(display_batches, fontsize=7 if len(display_batches) > 30 else 9)
        ax.tick_params(colors="#475569")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8, edgecolor="#e2e8f0")
        ax.grid(axis="y", linestyle=":", alpha=0.4, color="#94a3b8")

        st.pyplot(fig3, use_container_width=True)

        # Summary stats per subgroup
        with st.expander("View subgroup summary statistics"):
            summary = pd.DataFrame({
                "Subgroup": display_batches,
                "Min":    [d.min() for d in batch_data],
                "Q1":     [float(np.percentile(d, 25)) for d in batch_data],
                "Median": [float(np.median(d)) for d in batch_data],
                "Mean":   [d.mean() for d in batch_data],
                "Q3":     [float(np.percentile(d, 75)) for d in batch_data],
                "Max":    [d.max() for d in batch_data],
                "Std Dev":[d.std() for d in batch_data],
                "Status": ["OOC" if b in ooc_set else "OK" for b in display_batches],
            }).round(4)
            st.dataframe(summary, hide_index=True, use_container_width=True)

            csv_box = summary.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Download Box Plot Stats (CSV)", csv_box,
                               file_name="boxplot_stats.csv", mime="text/csv")

        buf3 = io.BytesIO()
        fig3.savefig(buf3, format="png", dpi=150, bbox_inches="tight")
        st.download_button("⬇️  Download Box Plot (PNG)", buf3.getvalue(),
                           file_name="boxplot.png", mime="image/png")


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 SQC Dashboard")
    st.divider()

    # ── Mode selector ──
    st.markdown("### 🔀 Mode")
    mode = st.radio("", ["📂  Upload Dataset", "🎛️  Simulation"],
                    label_visibility="collapsed")
    st.divider()

    # ── Chart type (shared) ──
    st.markdown("### 📐 Chart Type")
    chart_type = st.radio(" ", ["X-bar & R Chart", "X-bar & S Chart"],
                          label_visibility="collapsed")
    st.divider()

    # ── Spec limits (shared) ──
    st.markdown("### 📏 Specification Limits")
    usl = st.number_input("USL", value=33.0, step=0.1, format="%.3f")
    lsl = st.number_input("LSL", value=31.0, step=0.1, format="%.3f")
    st.divider()

    # ── Mode-specific controls ──
    if "Upload" in mode:
        st.markdown("### 📂 Upload File")
        uploaded = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"])
        st.markdown("### ⚙️ Column Mapping")
        st.caption("Available after upload.")
    else:
        uploaded = None
        st.markdown("### ⚙️ Process Parameters")
        sim_mean  = st.slider("Target Mean",            28.0, 36.0, 32.0, 0.1)
        sim_std   = st.slider("Process Std Dev",         0.05,  2.0,  0.3, 0.05)
        sim_shift = st.slider("Assignable Cause Shift",  0.0,   3.0,  1.0, 0.1)
        st.divider()
        st.markdown("### 🔢 Sampling Plan")
        if chart_type == "X-bar & R Chart":
            sim_n  = st.slider("Subgroup Size n  (2–10)", 2, 10, 5, 1)
            sim_sg = st.slider("Number of Subgroups",    10, 60, 40, 1)
        else:
            sim_n  = st.slider("Subgroup Size n  (>10)", 10, 30, 12, 1)
            sim_sg = st.slider("Number of Subgroups",    10, 60, 40, 1)
        abn = [b for b in ABNORMAL_BASE if b <= sim_sg]
        st.caption(f"OOC subgroups injected at: {abn}")
        regenerate = st.button("🔄  Regenerate", use_container_width=True, type="primary")

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
st.markdown("# 🏭 Statistical Quality Control Dashboard")
st.divider()

# ─────────────────────────────────────────────
# UPLOAD MODE
# ─────────────────────────────────────────────
if "Upload" in mode:
    st.markdown('<span class="mode-badge-data">📂 Upload Mode</span>', unsafe_allow_html=True)
    st.markdown(" ")

    if uploaded is None:
        st.markdown("""
        <div class="upload-box">
            <h3>📂 Upload your dataset to begin</h3>
            <p style="color:#475569; margin:0">
                Accepted: <strong>CSV</strong> or <strong>Excel (.xlsx / .xls)</strong><br><br>
                Required columns:<br>
                • <strong>Subgroup</strong> — integer group ID (1, 2, 3 …)<br>
                • <strong>Measurement column</strong> — numeric values
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("#### Example format")
        ex = pd.DataFrame({
            "Subgroup":[1,1,1,1,1,2,2,2,2,2],
            "Value":   [32.1,31.9,32.3,32.0,31.8,32.5,32.2,32.7,32.1,32.4]
        })
        st.dataframe(ex, hide_index=True, use_container_width=False)
        csv_s = ex.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️  Download sample_data.csv", csv_s,
                           file_name="sample_data.csv", mime="text/csv")
        st.stop()

    # Load file
    try:
        raw_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}"); st.stop()

    all_cols     = raw_df.columns.tolist()
    numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()

    # Column mapping in sidebar
    with st.sidebar:
        default_sg  = next((c for c in all_cols if any(k in c.lower() for k in ["subgroup","batch","group"])), all_cols[0])
        subgroup_col = st.selectbox("Subgroup column", all_cols, index=all_cols.index(default_sg))
        remaining    = [c for c in numeric_cols if c != subgroup_col]
        value_col    = st.selectbox("Measurement column", remaining or numeric_cols)

    if subgroup_col == value_col:
        st.error("❌ Subgroup and Measurement columns must differ."); st.stop()

    try:
        df = raw_df[[subgroup_col, value_col]].copy()
        df.columns = ["Subgroup","Value"]
        df["Subgroup"] = df["Subgroup"].astype(int)
        df["Value"]    = pd.to_numeric(df["Value"], errors="coerce")
        df.dropna(inplace=True)
    except Exception as e:
        st.error(f"❌ {e}"); st.stop()

    if df.empty:
        st.error("❌ No valid data after parsing."); st.stop()

    sg = df.groupby("Subgroup")["Value"].agg(
        xbar="mean", R=lambda x: x.max()-x.min(), S="std", n="count"
    ).reset_index()

    n         = int(sg["n"].mode()[0])
    all_vals  = df["Value"].values
    xbar_bar  = float(np.mean(sg["xbar"]))
    lim       = compute_limits(chart_type, xbar_bar, n, sg)
    actual_ct = lim["chart_type"]
    if actual_ct != chart_type:
        st.warning(f"⚠️ n={n} is outside R Chart table. Switched to S Chart automatically.")

    st.markdown(
        f"**File:** `{uploaded.name}` &nbsp;|&nbsp; **Chart:** {actual_ct} &nbsp;|&nbsp; "
        f"**Subgroups:** {len(sg)} &nbsp;|&nbsp; **n:** {n} &nbsp;|&nbsp; "
        f"**Total measurements:** {len(df)}"
    )

    render_tabs(df, sg, lim, all_vals, usl, lsl, n, actual_ct, value_col,
                subtitle=f"{actual_ct}  |  {uploaded.name}  |  n={n}")

    st.divider()
    st.caption(f"File: {uploaded.name} · Subgroup col: '{subgroup_col}' · "
               f"Measurement col: '{value_col}' · LSL={lsl} · USL={usl}")

# ─────────────────────────────────────────────
# SIMULATION MODE
# ─────────────────────────────────────────────
else:
    st.markdown('<span class="mode-badge-sim">🎛️ Simulation Mode</span>', unsafe_allow_html=True)
    st.markdown(" ")

    # Generate / cache data
    if "sim_df" not in st.session_state or regenerate:
        abn = [b for b in ABNORMAL_BASE if b <= sim_sg]
        records = []
        for b in range(1, sim_sg+1):
            m = sim_mean + sim_shift if b in abn else sim_mean
            for v in np.random.normal(m, sim_std, sim_n):
                records.append({"Subgroup": b, "Value": round(float(v), 4)})
        st.session_state.sim_df  = pd.DataFrame(records)
        st.session_state.sim_abn = abn

    df       = st.session_state.sim_df
    abnormal = st.session_state.sim_abn
    all_vals = df["Value"].values

    sg = df.groupby("Subgroup")["Value"].agg(
        xbar="mean", R=lambda x: x.max()-x.min(), S="std", n="count"
    ).reset_index()

    xbar_bar = float(np.mean(sg["xbar"]))
    lim      = compute_limits(chart_type, xbar_bar, sim_n, sg)
    actual_ct = lim["chart_type"]

    abn_str = ", ".join(str(b) for b in abnormal) if abnormal else "None"
    st.markdown(
        f"**Chart:** {actual_ct} &nbsp;|&nbsp; **Target mean:** {sim_mean} &nbsp;|&nbsp; "
        f"**σ:** {sim_std} &nbsp;|&nbsp; **Shift:** {sim_shift} &nbsp;|&nbsp; "
        f"**n:** {sim_n} &nbsp;|&nbsp; **Subgroups:** {sim_sg} &nbsp;|&nbsp; "
        f"**Injected OOC:** {abn_str}"
    )

    render_tabs(df, sg, lim, all_vals, usl, lsl, sim_n, actual_ct,
                value_col="Value",
                subtitle=f"{actual_ct}  |  Simulation  |  Mean={sim_mean}, σ={sim_std}, Shift={sim_shift}")

    st.divider()
    st.caption(f"Simulation · Mean={sim_mean} · σ={sim_std} · Shift={sim_shift} · "
               f"Injected OOC: {abn_str} · LSL={lsl} · USL={usl}")
