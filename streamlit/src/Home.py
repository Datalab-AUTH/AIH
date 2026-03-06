#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit_utils as stut

##### Setting page config #####
PAGE_TITLE = "AIAAIC incidents: Harms Severity analysis"
stut.set_page_config(PAGE_TITLE)

# Load data and define categories
option_category = st.selectbox(
    "Select analysis per Harms **Categories** or **Subcategories**:",
    ["Categories", "Subcategories"]
)
if option_category == 'Categories':
    df = pd.read_csv('../../results/categories_stakeholders_detailed.csv')
    category_col = 'harm_category'
elif option_category == 'Subcategories':
    df = pd.read_csv('../../results/subcategories_stakeholders_detailed.csv')
    category_col = 'harm_subcategory'
else:
    df = pd.DataFrame()
    st.error('Please select a category')

# Dropdown for selecting severity case
severity_case = st.selectbox("Select Severity Case", ["Original (1-9)", "Severity (2nd case: 50-450)"])

# Map severities based on selected case
if severity_case == "Original (1-9)":
    DEFAULT_SEVERITY = {
        "Artists/content creators": 1, "Subjects": 2, "Business": 3,
        "Investors": 4, "Workers": 5, "Users": 6, "Vulnerable groups": 7,
        "Government/public sector": 8, "General public": 9
    }
    MAX_SEVERITY = 10
    step_value = 1
else:
    DEFAULT_SEVERITY = {
        "General public": 50, "Government/public sector": 100,
        "Vulnerable groups": 150, "Users": 200, "Workers": 250, "Investors": 300,
        "Business": 350, "Subjects": 400, "Artists/content creators": 450
    }
    MAX_SEVERITY = 450
    step_value = 50

st.markdown(f"""
### Severity per Stakeholder group ({severity_case})
Below you can see and redefine the severity for each type of stakeholders.
""")

list_of_stakeholders = sorted(df['stakeholders'].unique().tolist())

# Check for missing stakeholders in severity mapping
missing_stakeholders = set(list_of_stakeholders) - set(DEFAULT_SEVERITY.keys())
if missing_stakeholders:
    st.warning(f"Missing stakeholders in severity mapping: {missing_stakeholders}")
    for s in missing_stakeholders:
        DEFAULT_SEVERITY[s] = 1 if severity_case == "Original (1-9)" else 50

# Sidebar for user-defined severity levels
dict_stakeholders_severity = {}
col1, col2 = st.columns([1, 2])
with col2:
    with st.expander(f"Select severity levels ({severity_case}):"):
        for s in list_of_stakeholders:
            dict_stakeholders_severity[s] = st.number_input(
                label=f"Group **{s}**; severity level:",
                value=DEFAULT_SEVERITY[s],
                step=step_value,
                min_value=1 if severity_case == "Original (1-9)" else 50,
                max_value=MAX_SEVERITY
            )

with col1:
    st.write(pd.Series(dict_stakeholders_severity, name='Severity'))

st.markdown("### Data")

# Map severity levels from the user's input
df['severity'] = df['stakeholders'].map(dict_stakeholders_severity)
df_with_cumulative = stut.get_cumulative_df_categories(df, category_col)

# Display the DataFrame with the additional cumulative columns
st.dataframe(df_with_cumulative)




# ------------------------------
# Method 1: Gini & Lorenz
# ------------------------------
df_gini_probability, plot_lorenz_probability = stut.plot_lorenz_curves_cumulative_probability(
    df_with_cumulative, category_col, total_stakeholders=9, default_severity=DEFAULT_SEVERITY
)



plot_lorenz_probability.update_layout(
    xaxis=dict(
        title=dict(
            text='Cumulative Share of Stakeholders',
            font=dict(size=22, weight='bold')  # Match the first plot's font style
        ),
        tickfont=dict(size=22, color='#666666'),
        showgrid=True,
        gridcolor='#DDDDDD',
        zeroline=False,
        range=[0, 1.1],  # Extend range slightly beyond 1
        tickmode='linear',  # Add additional tick values
        dtick=0.2          # Tick interval set to 0.2
    ),
    yaxis=dict(
        title=dict(
            text='Severity',
            font=dict(size=22, weight='bold')  # Match the first plot's font style
        ),
        tickfont=dict(size=22, color='#666666'),
        showgrid=True,
        gridcolor='#DDDDDD',
        zeroline=False
    ),
    autosize=False,  # Disable automatic sizing
    width=1000,       # Set fixed width
    height=600,      # Set fixed height to match width
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    paper_bgcolor='rgba(255,255,255,1)',  # White paper background
    legend=dict(
        title=dict(text='Harm Categories', font=dict(size=18, color='#333333', family='Courier New')),
        font=dict(size=16, family='Courier New'),
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='#DDDDDD',
        borderwidth=1,
        x=1.05,  # Move legend to the right of the plot
        y=0.5,
        xanchor='left',
        yanchor='middle',
        itemwidth=30
    )
)

st.markdown(
    f"### Lorenz curves (AIH): Normalized Severity Rank - {severity_case})", unsafe_allow_html=True
)
col_c, col_d = st.columns([1, 2])
with col_c:
    # Show the Gini table
    st.dataframe(df_gini_probability)
with col_d:
    # Show the Lorenz chart with the updated legend position
    st.plotly_chart(plot_lorenz_probability, key="method2_chart")


st.markdown(
    "<hr style='"
    "border: none;"
    "height: 2px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)

# ------------------------------
# Criticality Index Calculation
# ------------------------------
st.markdown("### Criticality Index per Harm Category")
# ------------------------------
# Debug: Inspect p_k and F_k values
# ------------------------------
with st.expander("🔍 Inspect $p_k$ and $F_k$ values per category"):
    df_ci_debug, debug_details = stut.calculate_criticality_index(df_with_cumulative, category_col, verbose=True)

    selected_cat = st.selectbox("Select category to inspect:", debug_details.keys())
    st.dataframe(debug_details[selected_cat])

# Compute Criticality Index using corrected method (mean of F_k)
df_criticality_index = stut.calculate_criticality_index(df_with_cumulative, category_col)

st.dataframe(df_criticality_index)


st.markdown(
    "<hr style='border:none;height:4px;background-color:#333;margin:2rem 0;'/>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# Additional inequality metrics: Pietra, Theil, Atkinson, Gini (numeric)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Additional Inequality Metrics")
st.markdown(
    "**AIH & Pietra** use *ordinal* severities (already defined above). "
    "**Gini, Theil, & Atkinson** additionally use *numeric* severities with meaningful distances. "
    "Both scales share the same **ordering** of stakeholders — only the spacing changes."
)

# ── Default numeric severity (equal spacing, same ordering as ordinal) ───────
DEFAULT_NUMERIC_SEVERITY = {
    "General public": 1,
    "Government/public sector": 5,
    "Vulnerable groups": 10,
    "Users": 15,
    "Workers": 20,
    "Investors": 25,
    "Business": 30,
    "Subjects": 35,
    "Artists/content creators": 40,
}

# ── Editable severity inputs ─────────────────────────────────────────────────
# Stakeholders in ordinal order (lowest → highest vulnerability)
ordered_stakeholders = [
    "General public", "Government/public sector", "Vulnerable groups",
    "Users", "Workers", "Investors", "Business", "Subjects",
    "Artists/content creators"
]

with st.expander("⚙️ Review & edit severity scales used for each metric"):
    st.markdown("**Ordinal severities (AIH, Pietra, CI)** — change in the panel at the top of this page.")
    st.markdown("**Numeric severities (Gini, Theil, Atkinson)** — edit below:")
    num_sev_cols = st.columns(3)
    numeric_severity_user = {}
    for idx, stakeholder in enumerate(ordered_stakeholders):
        col = num_sev_cols[idx % 3]
        numeric_severity_user[stakeholder] = col.number_input(
            label=stakeholder,
            value=float(DEFAULT_NUMERIC_SEVERITY.get(stakeholder, 1)),
            step=1.0,
            min_value=0.001,
            key=f"num_sev_{stakeholder}"
        )

NUMERIC_SEVERITY = numeric_severity_user

import plotly.graph_objs as go

# ── Build df with numeric severities ─────────────────────────────────────────
df_numeric = df_with_cumulative.copy()
df_numeric['severity'] = df_numeric['stakeholders'].map(NUMERIC_SEVERITY)

# ── Compute metrics ───────────────────────────────────────────────────────────
df_pietra   = stut.compute_pietra(df_with_cumulative, category_col, default_severity=DEFAULT_SEVERITY)
df_gini_num = stut.compute_gini(df_numeric, category_col, default_severity=NUMERIC_SEVERITY)
df_theil    = stut.compute_theil(df_numeric, category_col, default_severity=NUMERIC_SEVERITY)
df_atkinson = stut.compute_atkinson(df_numeric, category_col, epsilon=0.5, default_severity=NUMERIC_SEVERITY)

# ── Combined summary table ────────────────────────────────────────────────────
st.markdown("#### Summary: all inequality metrics")
df_summary = pd.DataFrame({
    "AIH":              df_gini_probability.round(4),
    "Pietra":           df_pietra.round(4),
    "Gini(numeric)":    df_gini_num.round(4),
    "Theil":            df_theil.round(4),
    "Atkinson(ε=0.5)":  df_atkinson.round(4),
    "CI":               df_criticality_index.round(4),
})
df_summary.index.name = "Harm Category"
st.dataframe(df_summary, use_container_width=True)

# ── Metrics Comparison Chart ─────────────────────────────────────────────────
fig_metrics = go.Figure()
colors = ['#1f77b4', '#ff7f0e', '#8c564b', '#2ca02c', '#d62728', '#9467bd']
for idx, col in enumerate(df_summary.columns):
    fig_metrics.add_trace(go.Bar(
        name=col,
        x=df_summary.index,
        y=df_summary[col],
        marker_color=colors[idx % len(colors)]
    ))

fig_metrics.update_layout(
    title="Comparison of Inequality Metrics by Harm Category",
    barmode="group",
    xaxis={'categoryorder':'total descending'},
    legend=dict(title="Metrics", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500,
    plot_bgcolor="white",
    hovermode="x unified"
)
fig_metrics.update_yaxes(showgrid=True, gridcolor="#DDDDDD")
st.plotly_chart(fig_metrics, use_container_width=True, key="metrics_comparison_chart")


# ── Interactive Scatter Plot ──────────────────────────────────────────────────
st.markdown("#### Interactive Metric Comparison")
st.markdown("Select two metrics to compare them in a scatter plot.")

metrics_list = list(df_summary.columns)
col1, col2 = st.columns(2)
with col1:
    x_metric = st.selectbox("X-axis Metric", options=metrics_list, index=metrics_list.index("CI") if "CI" in metrics_list else 0)
with col2:
    y_metric = st.selectbox("Y-axis Metric", options=metrics_list, index=metrics_list.index("AIH") if "AIH" in metrics_list else 1)

scatter_interactive = stut.plot_gini_vs_criticality(
    df_summary[y_metric], 
    df_summary[x_metric]
)
# Update layout to deeply override the hardcoded axis titles from the utility
scatter_interactive.update_layout(
    title=f"Scatter Plot: {y_metric} vs {x_metric}",
    xaxis=dict(
        title=dict(text=x_metric, font=dict(size=23, weight='bold'))
    ),
    yaxis=dict(
        title=dict(text=y_metric, font=dict(size=23, weight='bold'))
    )
)
st.plotly_chart(scatter_interactive, use_container_width=True, key="interactive_scatter_chart")






