#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
from numpy.random import exponential
import plotly.graph_objs as go
import streamlit_utils as stut

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
PAGE_TITLE = "Sensitivity of actual values of severity (Ordinal Gini vs Criticality Index)"
stut.set_page_config(PAGE_TITLE)

st.markdown("""
### Sensitivity Analysis of Gini Index (AIH) and Criticality Index  
Here we draw 20 exponential‐cumulated severity scenarios per stakeholder, then:
1. Compute **Method 1: AIH**  
2. Compute **Method 2: Criticality Index**  

We display the raw scenario tables, summary statistics, boxplots, and a final scatter of means.
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

# list of all stakeholders
distinct_stakeholders = sorted(df["stakeholders"].unique())

# ─── EXPONENTIAL SAMPLING ─────────────────────────────────────────────────────
st.markdown("### Exponential Severity Sampling")

def generate_exponential_severities(stakeholders, scale=1.0, scenarios=20):
    data = {}
    for i in range(1, scenarios+1):
        # draw exponential(λ=1/scale), then cumsum
        draws = exponential(scale=scale, size=len(stakeholders))
        data[f"Scenario {i}"] = np.cumsum(draws)
    return data

severity_cols = [f"Scenario {i}" for i in range(1, 21)]
severity_data = generate_exponential_severities(distinct_stakeholders, scale=1.0, scenarios=20)
severity_df = pd.DataFrame({"stakeholders": distinct_stakeholders, **severity_data})
st.dataframe(severity_df, use_container_width=True)

# convenience to build a mapping dict for padding
def sev_map(col):
    return dict(zip(distinct_stakeholders, severity_df[col]))

# ─── METHOD 1: AIH ────────────────────────────────────────
st.markdown("## Method 1: AIH", unsafe_allow_html=True)

def calculate_gini(df_base):
    all_results = {}
    for col in severity_cols:
        m = sev_map(col)
        df2 = (
            df_base
            .merge(severity_df[["stakeholders", col]], on="stakeholders")
            .rename(columns={col: "severity"})
        )
        cum = stut.get_cumulative_df_categories(df2, category_col)
        gser, _ = stut.plot_lorenz_curves_cumulative_probability(
            cum,
            category_col,
            total_stakeholders=len(distinct_stakeholders),
            default_severity=m
        )
        all_results[col] = gser.to_dict()
    return pd.DataFrame(all_results)

gini_df    = calculate_gini(df)
gini_stats = stut.get_gini_stats_df(gini_df)

st.dataframe(gini_df, height=250)
c1, c2 = st.columns([1,2])
with c1:
    st.dataframe(gini_stats, height=250)
with c2:
    # error‐bar chart of means & CIs, using original helper
    st.plotly_chart(stut.plot_mean_errorbar_with_var(gini_stats))

# compute “reference” Gini from original 1–9 mapping
original_sev = {
    "Artists/content creators": 1, "General public": 9, "Government/public sector": 8,
    "Users": 6, "Vulnerable groups": 7, "Workers": 5, "Business": 3,
    "Investors": 4, "Subjects": 2
}
df_ref = df.copy()
df_ref["severity"] = df_ref["stakeholders"].map(original_sev)
cum_ref = stut.get_cumulative_df_categories(df_ref, category_col)
gini_ref, _ = stut.plot_lorenz_curves_cumulative_probability(
    cum_ref, category_col,
    total_stakeholders=len(distinct_stakeholders),
    default_severity=original_sev
)

# boxplot with reference‐line
def plot_box_with_ref(df_vals, ref_series, line_label, ylabel):
    cats = sorted(df_vals.index)
    fig = go.Figure()
    for cat in cats:
        fig.add_trace(go.Box(
            y=df_vals.loc[cat],
            name=cat,
            boxpoints="all",
            jitter=0.5,
            whiskerwidth=0.5,
            marker=dict(size=5, opacity=0.7),
            line=dict(width=1),
        ))
    fig.add_trace(go.Scatter(
        x=cats,
        y=[ref_series[cat] for cat in cats],
        mode="lines+markers",
        name=line_label,
        line=dict(color="green", width=2),
        marker=dict(size=8, symbol="circle"),
    ))
    fig.update_layout(
        xaxis_title="Categories",
        yaxis_title=ylabel,
        xaxis=dict(
            title=dict(font=dict(size=22, weight="bold")),
            tickfont=dict(size=22),
            categoryorder="array",
            categoryarray=cats,
        ),
        yaxis=dict(
            title=dict(font=dict(size=22, weight="bold")),
            tickfont=dict(size=22),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(size=22),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#DDDDDD",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        width=1500,
        height=800,
        showlegend=True
    )
    return fig

st.markdown("### Box plots with Reference Gini")
st.plotly_chart(plot_box_with_ref(gini_df, gini_ref, "Reference Gini", "Gini Index"))

st.markdown(
    "<hr style='"
    "border: none;"
    "height: 2px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)

# ─── METHOD 2: CRITICALITY INDEX ─────────────────────────────────────────────
st.markdown("## Method 2: Criticality Index")

def calculate_ci(df_base):
    all_results = {}
    for col in severity_cols:
        df2 = (
            df_base
            .merge(severity_df[["stakeholders", col]], on="stakeholders")
            .rename(columns={col: "severity"})
        )
        cum = stut.get_cumulative_df_categories(df2, category_col)
        cser = stut.calculate_criticality_index(cum, category_col)
        all_results[col] = cser.to_dict()
    return pd.DataFrame(all_results)

ci_df    = calculate_ci(df)
ci_stats = stut.get_gini_stats_df(ci_df)

st.dataframe(ci_df, height=250)
c3, c4 = st.columns([1,2])
with c3:
    st.dataframe(ci_stats, height=250)
with c4:
    st.plotly_chart(stut.plot_mean_errorbar_with_var(ci_stats))

# reference CI from original severity
ci_ref = stut.calculate_criticality_index(cum_ref, category_col)

st.markdown("### Box plots with Reference CI")
st.plotly_chart(plot_box_with_ref(ci_df, ci_ref, "Reference CI", "Criticality Index"))

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