#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit_utils as stut
from numpy.random import randint

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
PAGE_TITLE = "Sensitivity Analysis: AIH vs Criticality Index"
stut.set_page_config(PAGE_TITLE)

st.markdown("""
### Sensitivity Analysis: AIH<sub>ord</sub> vs Criticality Index  
We draw 10 random “severity” scenarios, then for each:
1. Compute the **AIH** (via our Lorenz‐curve/AUC method).  
2. Compute the **Criticality Index** (mean of cumulative shares Fₖ).  

We then show boxplots, summary stats, and a final scatter of means.
""", unsafe_allow_html=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
option_category = st.selectbox(
    "Select analysis per Harms **Categories** or **Subcategories**:",
    ["Categories", "Subcategories"]
)
if option_category == "Categories":
    df = pd.read_csv("../../results/categories_stakeholders_detailed.csv")
    category_col = "harm_category"
else:
    df = pd.read_csv("../../results/subcategories_stakeholders_detailed.csv")
    category_col = "harm_subcategory"

# ─── RANDOM SEVERITY SCENARIOS ────────────────────────────────────────────────
stakeholders = sorted(df["stakeholders"].unique())
severity_cols = [f"Scenario {i}" for i in range(1, 11)]
severity_data = {
    col: randint(1, 10, size=len(stakeholders))
    for col in severity_cols
}
severity_df = pd.DataFrame({
    "stakeholders": stakeholders,
    **severity_data
})
st.markdown("#### Random Severity Scenarios")
st.dataframe(severity_df)

# helper to get mapping dict for a given scenario
def get_mapping(col):
    return dict(zip(severity_df["stakeholders"], severity_df[col]))

# ─── METHOD 1: AIHₒᵣᵈ Gini ─────────────────────────────────────────────────────
st.markdown("## Method 1: AIH", unsafe_allow_html=True)

def calculate_gini_indices(base_df):
    results = {}
    for col in severity_cols:
        mapping = get_mapping(col)
        # merge in this scenario's severities
        df2 = base_df.merge(
            severity_df[["stakeholders", col]],
            on="stakeholders"
        ).rename(columns={col: "severity"})
        # compute cumulative table + padded Lorenz/AUC
        cum = stut.get_cumulative_df_categories(df2, category_col)
        gini_ser, _ = stut.plot_lorenz_curves_cumulative_probability(
            cum,
            category_col,
            total_stakeholders=len(stakeholders),
            default_severity=mapping
        )
        results[col] = gini_ser.to_dict()
    return pd.DataFrame(results)

gini_df = calculate_gini_indices(df)
gini_stats = stut.get_gini_stats_df(gini_df)

st.dataframe(gini_df, height=300)
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(gini_stats, height=300)
with col2:
    # boxplot helper reused from original
    def boxplot(df_idx, stats, label):
        cats = sorted(df_idx.index)
        fig = go.Figure()
        for cat in cats:
            fig.add_trace(go.Box(
                y=df_idx.loc[cat], name=cat,
                boxpoints="all", jitter=0.5, whiskerwidth=0.5,
                marker=dict(size=5, opacity=0.7), line=dict(width=1)
            ))
        fig.add_trace(go.Scatter(
            x=cats, y=stats["MEAN"],
            mode="markers+lines", name=f"Mean {label}",
            marker=dict(size=8), line=dict(dash="dash", width=2),
            error_y=dict(type="data", array=stats["CONF. INTERVAL"], visible=True)
        ))
        fig.add_trace(go.Scatter(
            x=cats, y=stats["VaR (95%)"],
            mode="markers+lines", name="VaR (95%)",
            marker=dict(symbol="x", size=8), line=dict(dash="dot", width=2)
        ))
        fig.update_layout(
            xaxis_title="Categories", yaxis_title=label,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            width=900, height=500
        )
        return fig

    st.plotly_chart(boxplot(gini_df, gini_stats, "AIH"))

st.markdown(
    "<hr style='"
    "border: none;"
    "height: 2px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)

# ─── METHOD 2: Criticality Index ─────────────────────────────────────────────
st.markdown("## Method 2: Criticality Index")

def calculate_ci_indices(base_df):
    results = {}
    for col in severity_cols:
        # merge in this scenario's severities
        df2 = base_df.merge(
            severity_df[["stakeholders", col]],
            on="stakeholders"
        ).rename(columns={col: "severity"})
        cum = stut.get_cumulative_df_categories(df2, category_col)
        ci_ser = stut.calculate_criticality_index(cum, category_col)
        results[col] = ci_ser.to_dict()
    return pd.DataFrame(results)

ci_df    = calculate_ci_indices(df)
ci_stats = stut.get_gini_stats_df(ci_df)

st.dataframe(ci_df, height=300)
col3, col4 = st.columns([1, 2])
with col3:
    st.dataframe(ci_stats, height=300)
with col4:
    st.plotly_chart(boxplot(ci_df, ci_stats, "Criticality Index"))

st.markdown(
    "<hr style='"
    "border: none;"
    "height: 4px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)

# ─── MEAN VS MEAN SCATTER ────────────────────────────────────────────────────
st.markdown("### Mean Comparison: AIH vs Criticality Index", unsafe_allow_html=True)
scatter = stut.plot_gini_vs_criticality(
    gini_stats["MEAN"],
    ci_stats["MEAN"]
)
st.plotly_chart(scatter)