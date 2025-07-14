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
        "Artists/content creators": 1, "General public": 9, "Government/public sector": 8,
        "Users": 6, "Vulnerable groups": 7, "Workers": 5, "Business": 3, "Investors": 4,
        "Subjects": 2
    }
    MAX_SEVERITY = 10
    step_value = 1
else:
    DEFAULT_SEVERITY = {
        "Artists/content creators": 50, "General public": 450, "Government/public sector": 400,
        "Users": 300, "Vulnerable groups": 350, "Workers": 250, "Business": 150, "Investors": 200,
        "Subjects": 100
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
    "<hr style='"
    "border: none;"
    "height: 4px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)
st.markdown("### Scatter Plot: AIH vs Criticality Index")



scatter_gini_ci = stut.plot_gini_vs_criticality(df_gini_probability, df_criticality_index)
st.plotly_chart(scatter_gini_ci, key="gini_vs_ci")




