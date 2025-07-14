#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit_utils as stut


DEFAULT_SEVERITY = {
        "Artists/content creators": 1, "General public": 9, "Government/public sector": 8,
        "Users": 6, "Vulnerable groups": 7, "Workers": 5, "Business": 3, "Investors": 4,
        "Subjects": 2
    }

# --- Scenario Generation ----------------------------------------------------
def generate_fixed_removal_scenarios_with_base(df, removal_percentages=[0.1, 0.2, 0.5, 0.8]):
    scenarios = {}

    # Scenario 0: full data with fixed severities
    base = df.copy()
    base['severity'] = base['stakeholders'].map(DEFAULT_SEVERITY)
    scenarios["All Annotations"] = base

    total = len(base)
    for i, pct in enumerate(removal_percentages, start=1):
        df2 = base.copy()
        remove_n = int(pct * total)
        if len(df2) > remove_n:
            df2 = df2.sample(n=len(df2)-remove_n)
        scenarios[f"{int(pct * 100)}% Removed"] = df2

    return scenarios


# --- Plot Helpers -----------------------------------------------------------
def plot_trend_interactive(df, stats, yaxis_title, key):
    fig = go.Figure()
    for cat in df.index:
        ci = stats.loc[cat, "CONF. INTERVAL"]
        fig.add_trace(go.Scatter(
            x=df.columns,
            y=df.loc[cat],
            mode='lines+markers',
            name=cat,
            line=dict(width=2),
            marker=dict(size=8),
            error_y=dict(
                type='data',
                array=[ci] * len(df.columns),
                visible=True,
                thickness=2,
                width=4,
            )
        ))
    fig.update_layout(
        xaxis_title="Scenarios",
        yaxis_title=yaxis_title,
        xaxis=dict(
            title_font=dict(size=22, weight='bold'),
            tickfont=dict(size=22),
        ),
        yaxis=dict(
            title_font=dict(size=22, weight='bold'),
            tickfont=dict(size=22)
        ),
        legend=dict(title="Categories", font=dict(size=20), orientation='v', x=1.05, y=1.0),
        margin=dict(l=50, r=150, t=50, b=50)
    )

    st.plotly_chart(fig, use_container_width=True, key=key)


def plot_box_with_stats(df, stats, yaxis_title, key):
    fig = go.Figure()
    # Box for each category
    for cat in df.index:
        fig.add_trace(go.Box(
            y=df.loc[cat], name=cat,
            boxpoints='outliers', marker=dict(size=5, opacity=0.7), line=dict(width=1)
        ))
    # Overlay mean+CI+VaR
    fig.add_trace(go.Scatter(
        x=stats.index,
        y=stats['MEAN'],
        mode='markers+lines',
        name='Mean',
        marker=dict(color='red', size=8),
        line=dict(color='red', dash='dash', width=2),
        error_y=dict(type='data', array=stats['CONF. INTERVAL'], visible=True, color='red')
    ))
    fig.add_trace(go.Scatter(
        x=stats.index,
        y=stats['VaR (95%)'],
        mode='markers',
        name='VaR (95%)',
        marker=dict(color='black', size=6, symbol='diamond')
    ))
    fig.update_layout(
        xaxis_title='Categories',
        yaxis_title=yaxis_title,
        xaxis=dict(title_font=dict(size=22, weight='bold'), tickfont=dict(size=22)),
        yaxis=dict(title_font=dict(size=22, weight='bold'), tickfont=dict(size=22)),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True, key=key)

# --- Main -------------------------------------------------------------------
stut.set_page_config("Sensitivity Analysis with Incremental Annotation Removal")

option = st.selectbox("Select analysis per Harms **Categories** or **Subcategories**:",
                      ["Categories", "Subcategories"])
if option == 'Categories':
    df = pd.read_csv('../../results/categories_stakeholders_detailed.csv')
    cat_col = 'harm_category'
else:
    df = pd.read_csv('../../results/subcategories_stakeholders_detailed.csv')
    cat_col = 'harm_subcategory'

# Generate scenarios
scenarios = generate_fixed_removal_scenarios_with_base(df)
all_stakeholders = sorted(df['stakeholders'].unique())

# --- Method 1: AIH -----------------------------------------------
st.markdown("## Method 1: AIH", unsafe_allow_html=True)

gini_dict = {}
for name, df_scen in scenarios.items():
    # Dynamically create a severity map for this scenario
    dynamic_severity_map = df_scen.groupby('stakeholders')['severity'].first().to_dict()

    cum = stut.get_cumulative_df_categories(df_scen, cat_col)
    gser, _ = stut.plot_lorenz_curves_cumulative_probability(
        cum, cat_col,
        total_stakeholders = len(all_stakeholders),
        default_severity = dynamic_severity_map  # ✅ now matches actual scenario!
    )
    gini_dict[name] = gser


gini_df = pd.DataFrame(gini_dict)
st.dataframe(gini_df)

stats_gini = stut.get_gini_stats_df(gini_df)

st.markdown("### Trend: AIH Across Scenarios", unsafe_allow_html=True)
plot_trend_interactive(gini_df, stats_gini, "AIH", "trend_aih")

st.markdown("### Boxplot + Mean, CI, VaR: AIH", unsafe_allow_html=True)
plot_box_with_stats(gini_df, stats_gini, "AIH", "box_aih")

st.markdown(
    "<hr style='"
    "border: none;"
    "height: 2px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)

# --- Method 2: Criticality Index -------------------------------------------
st.markdown("## Method 2: Criticality Index")

crit_dict = {}
for name, df_scen in scenarios.items():
    cum = stut.get_cumulative_df_categories(df_scen, cat_col)
    cidx = stut.calculate_criticality_index(cum, cat_col)
    crit_dict[name] = cidx

crit_df = pd.DataFrame(crit_dict)
st.dataframe(crit_df)

stats_crit = stut.get_gini_stats_df(crit_df)

st.markdown("### Trend: Criticality Index Across Scenarios")
plot_trend_interactive(crit_df, stats_crit, "Criticality Index", "trend_crit")

st.markdown("### Boxplot + Mean, CI, VaR: Criticality Index")
plot_box_with_stats(crit_df, stats_crit, "Criticality Index", "box_crit")

st.markdown(
    "<hr style='"
    "border: none;"
    "height: 4px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)

# --- Comparison Scatter -----------------------------------------------------
st.markdown("### Scatter: AIH vs Criticality Index", unsafe_allow_html=True)
mean_aih = stats_gini['MEAN']
mean_crit = stats_crit['MEAN']
fig_cmp = stut.plot_gini_vs_criticality(gini_series=mean_aih, criticality_series=mean_crit)
st.plotly_chart(fig_cmp, use_container_width=True, key="scatter_cmp")



st.markdown(
    "<hr style='"
    "border: none;"
    "height: 2px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)

# ─── SUBCATEGORY-LEVEL REMOVAL SENSITIVITY (only when “Subcategories” mode) ─────
if option == 'Subcategories':
    # load mapping of categories → subcategories
    mapping_df = pd.read_csv('../../results/category_subcategory_mapping.csv')

    st.markdown("## Subcategory-Level Removal Sensitivity")
    top_cat = st.selectbox(
        "Choose High-Level Category:",
        mapping_df['harm_category'].unique(),
        key='removal_drill_top'
    )

    # find its subcategories
    subs = mapping_df.loc[
        mapping_df['harm_category'] == top_cat,
        'harm_subcategory'
    ].unique().tolist()

    if not subs:
        st.warning(f"No subcategories found for {top_cat!r}.")
    else:
        # filter your original subcategory-detailed df
        df_sub = df[df[cat_col].isin(subs)].copy()

        # regenerate removal scenarios for just these subcategories
        sub_scenarios = generate_fixed_removal_scenarios_with_base(df_sub)

        # --- Gini on subcategories ---------------------------------------
        st.markdown(f"### AIH for “{top_cat}” Subcategories", unsafe_allow_html=True)
        gini_sub = {}
        for name, df_s in sub_scenarios.items():
            sev_map = df_s.groupby('stakeholders')['severity'].first().to_dict()
            cum = stut.get_cumulative_df_categories(df_s, cat_col)
            gser, _ = stut.plot_lorenz_curves_cumulative_probability(
                cum, cat_col,
                total_stakeholders=len(all_stakeholders),
                default_severity=sev_map
            )
            gini_sub[name] = gser
        gini_sub_df = pd.DataFrame(gini_sub)
        stats_sub_gini = stut.get_gini_stats_df(gini_sub_df)

        plot_trend_interactive(gini_sub_df, stats_sub_gini,
                               "AIH", key="trend_aih_sub")
        plot_box_with_stats(gini_sub_df, stats_sub_gini,
                            "AIH", key="box_aih_sub")

        st.markdown("<hr/>", unsafe_allow_html=True)

        # --- CI on subcategories ------------------------------------------
        st.markdown(f"### CI for “{top_cat}” Subcategories", unsafe_allow_html=True)
        ci_sub = {}
        for name, df_s in sub_scenarios.items():
            cum = stut.get_cumulative_df_categories(df_s, cat_col)
            ci_sub[name] = stut.calculate_criticality_index(cum, cat_col)
        ci_sub_df = pd.DataFrame(ci_sub)
        stats_sub_ci = stut.get_gini_stats_df(ci_sub_df)

        plot_trend_interactive(ci_sub_df, stats_sub_ci,
                               "Criticality Index", key="trend_ci_sub")
        plot_box_with_stats(ci_sub_df, stats_sub_ci,
                            "Criticality Index", key="box_ci_sub")

        st.markdown("<hr/>", unsafe_allow_html=True)

        # --- comparison scatter --------------------------------------------
        st.markdown("### Scatter: AIH vs CI (Subcategories)", unsafe_allow_html=True)
        mean_gini_sub = stats_sub_gini['MEAN']
        mean_ci_sub   = stats_sub_ci['MEAN']
        fig_cmp_sub = stut.plot_gini_vs_criticality(
            gini_series=mean_gini_sub,
            criticality_series=mean_ci_sub
        )
        st.plotly_chart(fig_cmp_sub, use_container_width=True, key="scatter_cmp_sub")
