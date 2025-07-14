#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import streamlit_utils as stut

# 0) page config must be first!
st.set_page_config(layout="wide", page_title="AIH Boundary Analysis")
st.title("Boundary analysis for AIH & Criticality Index")

# ————————————— HELPERS —————————————
def assign_severities_boundaries(df_freq, case):
    """
    Given df_freq[['stakeholders','freq']], assign severities 1…M so as to
    minimize ('best') or maximize ('worst') inequality.
    """
    df = df_freq.sort_values("freq", ascending=False).reset_index(drop=True)
    M = len(df)
    if case == "worst":
        df["severity"] = list(range(M, 0, -1))
    elif case == "best":
        df["severity"] = list(range(1, M + 1))
    else:
        raise ValueError("case must be 'best' or 'worst'")
    return df

def plot_combined_lorenz_curve_ord(category,
                                   df_worst, df_best,
                                   sev_map_worst, sev_map_best,
                                   category_col):
    # filter to just that category
    df_w_cat = df_worst[df_worst[category_col] == category]
    df_b_cat = df_best [df_best [category_col] == category]

    # plot with exact same stut call (pads zeros for missing)
    gini_best, fig_best = stut.plot_lorenz_curves_cumulative_probability(
        df_b_cat, category_col,
        total_stakeholders=len(sev_map_best),
        default_severity=sev_map_best
    )
    gini_worst, fig_worst = stut.plot_lorenz_curves_cumulative_probability(
        df_w_cat, category_col,
        total_stakeholders=len(sev_map_worst),
        default_severity=sev_map_worst
    )

    # pull out the two single traces + equality
    best_trace  = fig_best.data[0]
    worst_trace = fig_worst.data[0]
    eq_trace    = fig_best.data[1]

    best_trace.name  = f"Best Case AUC={gini_best[category]:.3f}"
    best_trace.line.color  = "blue"
    worst_trace.name = f"Worst Case AUC={gini_worst[category]:.3f}"
    worst_trace.line.color = "red"
    eq_trace.name    = "Equality (AUC=0.500)"

    fig = go.Figure([best_trace, worst_trace, eq_trace])
    fig.update_layout(
        title=f"Harm Category: {category}",
        xaxis_title="Cumulative Share of Stakeholders",
        yaxis_title="Severity",
        width=900, height=600,
        legend=dict(font=dict(size=14), orientation="h", yanchor="bottom", y=1.02)
    )
    return fig

# ————————————— 1) Load & choose —————————————
choice = st.selectbox("Analyze Categories or Subcategories?", ["Categories","Subcategories"])
if choice == "Categories":
    df = pd.read_csv("../../results/categories_stakeholders_detailed.csv")
    category_col = "harm_category"
else:
    df = pd.read_csv("../../results/subcategories_stakeholders_detailed.csv")
    category_col = "harm_subcategory"

# Pull the full universe of stakeholders once
all_stakeholders = df["stakeholders"].unique()

# ————————————— 2–4) per-category extremes —————————————
boundary_data = {
    "Best-Case AIH (AUC)":  {},
    "Worst-Case AIH (AUC)": {},
    "Best-Case CI":                  {},
    "Worst-Case CI":                 {}
}

for cat in df[category_col].unique():
    # slice to this category
    df_cat = df[df[category_col] == cat]

    # frequency by stakeholder for this category
    df_freq_cat = df_cat.groupby("stakeholders", as_index=False)["freq"].sum()

    # ─── pad zero-frequency rows for *all* stakeholders ───
    df_freq_full = pd.DataFrame({"stakeholders": all_stakeholders})
    df_freq_full = (
        df_freq_full
        .merge(df_freq_cat, on="stakeholders", how="left")
        .fillna({"freq": 0})
    )

    # get best/worst severity mappings for this category
    df_best_map_cat  = assign_severities_boundaries(df_freq_full,  "best")
    df_worst_map_cat = assign_severities_boundaries(df_freq_full, "worst")
    sev_best  = df_best_map_cat.set_index("stakeholders")["severity"].to_dict()
    sev_worst = df_worst_map_cat.set_index("stakeholders")["severity"].to_dict()

    # apply the mappings
    df_cat_best  = df_cat.assign(severity=lambda d: d["stakeholders"].map(sev_best))
    df_cat_worst = df_cat.assign(severity=lambda d: d["stakeholders"].map(sev_worst))

    # compute Gini (AUC) for best/worst
    g_best, _  = stut.plot_lorenz_curves_cumulative_probability(
        df_cat_best, category_col,
        total_stakeholders=len(all_stakeholders),
        default_severity=sev_best
    )
    g_worst, _ = stut.plot_lorenz_curves_cumulative_probability(
        df_cat_worst, category_col,
        total_stakeholders=len(all_stakeholders),
        default_severity=sev_worst
    )
    boundary_data["Best-Case AIH (AUC)"][cat]  = g_best.loc[cat]
    boundary_data["Worst-Case AIH (AUC)"][cat] = g_worst.loc[cat]

    # compute Criticality Index for best/worst
    ci_best  = stut.calculate_criticality_index(df_cat_best,  category_col)
    ci_worst = stut.calculate_criticality_index(df_cat_worst, category_col)
    boundary_data["Best-Case CI"][cat]  = ci_best.loc[cat]
    boundary_data["Worst-Case CI"][cat] = ci_worst.loc[cat]

# ————————————— 5) Display table of extremes —————————————
boundary_df = pd.DataFrame(boundary_data)
st.markdown("### Extreme possible values by harm category")
st.dataframe(boundary_df.style.format("{:.3f}"))

# ————————————— 6) Pick one category & overlay curves —————————————
selected = st.selectbox("Pick harm category:", boundary_df.index.tolist())

# Recompute padded freq & severities for the selected category
df_sel      = df[df[category_col] == selected]
df_freq_sel = df_sel.groupby("stakeholders", as_index=False)["freq"].sum()
df_freq_sel = (
    pd.DataFrame({"stakeholders": all_stakeholders})
    .merge(df_freq_sel, on="stakeholders", how="left")
    .fillna({"freq": 0})
)

df_best_map  = assign_severities_boundaries(df_freq_sel,  "best")
df_worst_map = assign_severities_boundaries(df_freq_sel, "worst")
sev_best_sel  = df_best_map .set_index("stakeholders")["severity"].to_dict()
sev_worst_sel = df_worst_map.set_index("stakeholders")["severity"].to_dict()

df_sel_best  = df_sel.assign(severity=lambda d: d["stakeholders"].map(sev_best_sel))
df_sel_worst = df_sel.assign(severity=lambda d: d["stakeholders"].map(sev_worst_sel))

fig = plot_combined_lorenz_curve_ord(
    category       = selected,
    df_worst       = df_sel_worst,
    df_best        = df_sel_best,
    sev_map_worst  = sev_worst_sel,
    sev_map_best   = sev_best_sel,
    category_col   = category_col
)
st.plotly_chart(fig, use_container_width=True)
