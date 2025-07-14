#!/usr/bin/env python3

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import streamlit_utils as stut
import plotly.express as px

##### METHODS #####

# Modify the DEFAULT_SEVERITY dictionary to align with the Home page script
DEFAULT_SEVERITY = {
    "Artists/content creators": 1, "General public": 9, "Government/public sector": 8,
    "Users": 6, "Vulnerable groups": 7, "Workers": 5, "Business": 3, "Investors": 4,
    "Subjects": 2
}
@st.cache_data(show_spinner=False)
def calculate_mean_gini_across_permutations(scenarios, gini_dict):
    mean_gini_dict = {}
    ci_dict = {}

    permutation_groups = {
        "Original": ["Original"],
        "1 Permutation": [f"Perm{i}" for i in range(1, 6)],
        "2 Permutations": [f"Perm{i}" for i in range(6, 11)],
        "5 Permutations": [f"Perm{i}" for i in range(11, 21)],
    }

    for group_name, group_permutations in permutation_groups.items():
        group_mean_values = []
        group_ci_values = []

        for scenario_name, scenario in scenarios.items():
            if group_name == "Original" and scenario_name == "Scenario 0":
                gini_values = gini_dict[scenario_name].loc[:, "Original"]
                group_mean_values.append(gini_values)
                group_ci_values.append(pd.Series(0, index=gini_values.index))
            else:
                gini_values = gini_dict[scenario_name].loc[:, group_permutations]
                mean_values = gini_values.mean(axis=1)
                std_dev = gini_values.std(axis=1)
                ci_values = 1.96 * std_dev / np.sqrt(len(group_permutations))

                group_mean_values.append(mean_values)
                group_ci_values.append(ci_values)

        mean_gini_dict[group_name] = pd.concat(group_mean_values, axis=1).mean(axis=1)
        ci_dict[group_name] = pd.concat(group_ci_values, axis=1).mean(axis=1)

    return pd.DataFrame(mean_gini_dict), pd.DataFrame(ci_dict)


def calculate_gini_indices_with_ci(df, method, severity_scenarios, category_col):
    """
    Compute AIH (method1) or Criticality Index (method2) for each severity scenario.
    Returns (gini_df, ci_df).
    """
    stakeholders = sorted(df['stakeholders'].unique())
    gini_dict = {}
    ci_dict   = {}

    for severity_col, severity_values in severity_scenarios.items():
        # 1) build a DataFrame & merge so every stakeholder is present
        mapping_df = pd.DataFrame({
            'stakeholders': stakeholders,
            severity_col: severity_values
        })
        df2 = df.merge(mapping_df, on='stakeholders').rename(
            columns={severity_col: 'severity'}
        )

        # 2) cumulative shares
        cum = stut.get_cumulative_df_categories(df2, category_col)

        if method == 'method1':
            # pad zeros so the Lorenz curve sees all stakeholders
            default_severity = dict(zip(stakeholders, severity_values))
            gser, _ = stut.plot_lorenz_curves_cumulative_probability(
                cum,
                category_col,
                total_stakeholders = len(stakeholders),
                default_severity  = default_severity
            )
            vals = gser.to_dict()

        elif method == 'method2':
            vals = stut.calculate_criticality_index(cum, category_col).to_dict()

        else:
            raise ValueError("Invalid method. Choose 'method1' or 'method2'.")

        series = pd.Series(vals)
        gini_dict[severity_col] = series
        ci_dict[severity_col]   = pd.Series(
            1.96 * series.std() / np.sqrt(len(series)),
            index=series.index
        )

    return pd.DataFrame(gini_dict), pd.DataFrame(ci_dict)


@st.cache_data(show_spinner=False)
def calculate_gini_indices_for_scenarios(df, severity_scenarios, method, category_col):
    gini_dict = {}

    for scenario_name, permutations in severity_scenarios.items():
        scenario_ginis = {}

        for perm_name, perm_values in permutations.items():
            severity_df = pd.DataFrame({'stakeholders': list_of_stakeholders, 'severity': perm_values})
            df_merged = pd.merge(df, severity_df, on='stakeholders')
            df_merged['severity'] = pd.to_numeric(df_merged['severity'], errors='coerce')
            df_with_cumulative = stut.get_cumulative_df_categories(df_merged, category_col)

            if method == 'method1':
                # build a mapping of all stakeholders → their permuted severity
                default_severity = dict(zip(list_of_stakeholders, perm_values))

                gini_series, _ = stut.plot_lorenz_curves_cumulative_probability(
                    df_with_cumulative,
                    category_col,
                    total_stakeholders=len(list_of_stakeholders),
                    default_severity=default_severity
                )
                gini_values = gini_series.to_dict()

            elif method == 'method2':
                gini_values = stut.calculate_criticality_index(
                    df_with_cumulative, category_col
                )

            else:
                raise ValueError("Invalid method. Choose either 'method1' or 'method2'.")

            scenario_ginis[perm_name] = pd.Series(gini_values)

        gini_dict[scenario_name] = pd.DataFrame(scenario_ginis)

    return gini_dict



def plot_gini_sensitivity_with_ci_for_scenarios(mean_gini_df, ci_df, unique_key, *, yaxis_title="Mean AIH"
):
    """
    Creates a line plot with error bars (confidence intervals) for AIH sensitivity, grouped by scenarios.
    """
    fig = go.Figure()

    # Iterate over the rows (now permutations are represented as lines)
    for permutation in mean_gini_df.index:
        # Extract values for the permutation across scenarios
        mean_values = mean_gini_df.loc[permutation].values
        conf_intervals = ci_df.loc[permutation].values

        fig.add_trace(go.Scatter(
            x=mean_gini_df.columns,  # Scenarios (now x-axis)
            y=mean_values,  # Mean AIH values
            mode="lines+markers",
            name=permutation,  # Permutation group (e.g., Original, 1 Permutation, etc.)
            line=dict(width=2),
            marker=dict(size=8),
            error_y=dict(
                type="data",
                array=conf_intervals,
                visible=True,
                thickness=2.0,
                width=4,
            ),
        ))

    fig.update_layout(
        #title=title,
        xaxis_title="Scenarios",
        yaxis_title=yaxis_title,
        xaxis=dict(title=dict(font=dict(size=24, weight="bold")), tickfont=dict(size=24), tickmode="array", tickvals=mean_gini_df.columns  # Rotate x-axis labels
),
        yaxis=dict(title=dict(font=dict(size=24, weight="bold")), tickfont=dict(size=24)),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
        legend=dict(
            font=dict(size=22),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#DDDDDD',
            borderwidth=1,
        ),     margin=dict(l=50, r=150, t=50, b=50),
        width=1000,
        height=800,

    )

    st.plotly_chart(fig, use_container_width=True, key=unique_key)

def generate_scenario_permutations(stakeholders, num_scenarios=3):
    """
    Generates scenarios with varying number of permutations for each scenario.
    Scenario 1: 1 permutation for each of the 20 perm columns.
    Scenario 2: 2 permutations for each of the 20 perm columns.
    Scenario 3: 5 permutations for each of the 20 perm columns.
    """
    sorted_severity = [DEFAULT_SEVERITY[stakeholder] for stakeholder in stakeholders]
    scenarios = {}

    for scenario_num in range(1, num_scenarios + 1):
        scenario = {"Original": sorted_severity.copy()}  # Start with predefined severities

        # Set the number of permutations based on the scenario number
        num_permutations = scenario_num  # Scenario 1 has 1 permutation, Scenario 2 has 2, etc.

        for perm_num in range(1, 21):  # 20 perm columns per scenario
            perm = sorted_severity.copy()
            for _ in range(num_permutations):  # Apply permutations for each column based on scenario
                valid_indices = [(i, j) for i in range(len(perm)) for j in range(i + 1, len(perm)) if
                                 abs(perm[i] - perm[j]) == 1]
                if valid_indices:
                    ind1, ind2 = valid_indices[np.random.choice(len(valid_indices))]
                    perm[ind1], perm[ind2] = perm[ind2], perm[ind1]  # Swap the values
            scenario[f"Perm{perm_num}"] = perm

        scenarios[f"Scenario {scenario_num}"] = scenario

    return scenarios


def plot_gini_permutation_lineplot(gini_df, title, yaxis_title="AIH"):
    fig = go.Figure()

    for category in gini_df.index:
        fig.add_trace(go.Scatter(
            x=gini_df.columns,
            y=gini_df.loc[category],
            mode='lines+markers',
            name=category
        ))

    fig.update_layout(
        xaxis_title="Permutations",
        yaxis_title=yaxis_title,
        xaxis=dict(title=dict(font=dict(size=22, weight='bold')), tickfont=dict(size=22)),
        yaxis=dict(title=dict(font=dict(size=22, weight='bold')), tickfont=dict(size=22)),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
        legend_title="Categories",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_mean_gini_line(mean_gini, title, yaxis_title="AIH"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mean_gini.index,
        y=mean_gini.values,
        mode='lines+markers',
        name='Mean AIH'
    ))

    fig.update_layout(
        xaxis_title="Permutation",
        yaxis_title=yaxis_title,
        xaxis=dict(title=dict(font=dict(size=22, weight='bold')), tickfont=dict(size=22)),
        yaxis=dict(title=dict(font=dict(size=22, weight='bold')), tickfont=dict(size=22)),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_boxplot_with_mean_line(gini_df, title, yaxis_title="AIH"):
    fig = go.Figure()

    for category in gini_df.index:
        fig.add_trace(go.Box(
            y=gini_df.loc[category],
            name=category,
            boxpoints="outliers",
            marker=dict(size=4, opacity=0.7),
            line=dict(width=1.5),
            showlegend=False
        ))

    mean_gini_values = gini_df.mean(axis=1)
    fig.add_trace(go.Scatter(
        x=gini_df.index,
        y=mean_gini_values,
        mode='lines+markers',
        name='Mean AIH',
        line=dict(color='red', width=2),
        marker=dict(size=8, symbol='circle')
    ))

    fig.update_layout(
        xaxis_title="Categories",
        yaxis_title=yaxis_title,
        xaxis=dict(title=dict(font=dict(size=22, weight='bold')), tickfont=dict(size=22)),
        yaxis=dict(title=dict(font=dict(size=22, weight='bold')), tickfont=dict(size=22)),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True
    )
    return fig


def plot_mean_gini_comparison(mean_gini_method1, mean_gini_method2, categories):
    """
    Create a square scatter plot comparing mean AIH for two methods with larger text and dots.
    """
    fig = go.Figure()

    # Add scatter plot points
    fig.add_trace(go.Scatter(
        x=mean_gini_method1,
        y=mean_gini_method2,
        mode='markers+text',
        text=categories,
        textposition='top center',
        textfont=dict(size=19),  # Increased font size for category labels on dots
        marker=dict(size=16, color='#0072B2', line=dict(width=0, color='white')),  # Increased marker size
        name="Categories"
    ))

    # Determine max Gini for equal scaling
    max_gini = max(max(mean_gini_method1), max(mean_gini_method2)) + 0.05

    # Add 45-degree line
    fig.add_trace(go.Scatter(
        x=[0, max_gini],
        y=[0, max_gini],
        mode='lines',
        line=dict(dash='dash', color='#999999', width=3),  # Increased line width
        name="45-degree line"
    ))

    # Update layout to enforce square aspect ratio and larger text
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Mean AIH (Method 2: Cumulative Severity)",
                font=dict(size=23, weight='bold')
            ),
            tickfont=dict(size=22, color='#666666'),
            showgrid=True,
            gridcolor='#DDDDDD',
            range=[0, max_gini]
        ),
        yaxis=dict(
            title=dict(
                text="Mean AIH (Method 1: Severity)",
                font=dict(size=23, weight='bold')
            ),
            tickfont=dict(size=22, color='#666666'),
            showgrid=True,
            gridcolor='#DDDDDD',
            range=[0, max_gini]
        ),
        legend=dict(
            font=dict(size=22),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#DDDDDD',
            borderwidth=1
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=50, r=50, t=60, b=50),
        width=1350,  # Set fixed width
        height=980  # Set fixed height to enforce square aspect ratio
    )

    return fig

def plot_gini_boxplots_for_scenarios_with_no_permutations_v2(
    gini_dict, mean_gini_no_permutations, category, unique_key, yaxis_title="AIH"
):
    """
    Creates boxplots of AIH values for a specific harm category across scenarios,
    including the "No Permutations" scenario, using its calculated AIH value,
    and applies a consistent color palette.
    """
    boxplot_data = []

    # Define a consistent color palette
    color_palette = {
        "No Permutations": "#636EFA",  # Blue
        "1 Permutation": "#EF553B",   # Red
        "2 Permutations": "#00CC96",  # Green
        "5 Permutations": "#AB63FA",  # Purple
    }

    # Collect Gini values for the selected category across all scenarios
    for scenario_name, gini_df in gini_dict.items():
        if category not in gini_df.index:
            st.warning(f"Category '{category}' not found in {scenario_name}.")
            continue

        # Collect Gini values for all permutations in the scenario
        gini_values = gini_df.loc[category].values.flatten()

        # Append scenario and Gini values
        for gini_value in gini_values:
            boxplot_data.append({"Scenario": scenario_name, "AIH Value": gini_value})

    # Add "No Permutations" as a single-value scenario using the pre-computed mean Gini
    boxplot_data.append({"Scenario": "No Permutations", "AIH Value": mean_gini_no_permutations[category]})

    if not boxplot_data:
        st.error(f"No AIH data available for category '{category}'.")
        return

    # Convert to DataFrame
    boxplot_df = pd.DataFrame(boxplot_data)

    # Map scenario names to readable labels
    scenario_map = {
        "No Permutations": "No Permutations",
        "Scenario 1": "1 Permutation",
        "Scenario 2": "2 Permutations",
        "Scenario 3": "5 Permutations",
    }
    boxplot_df["Scenario"] = boxplot_df["Scenario"].map(scenario_map)

    # Ensure the order of scenarios on the x-axis
    ordered_scenarios = ["No Permutations", "1 Permutation", "2 Permutations", "5 Permutations"]
    boxplot_df["Scenario"] = pd.Categorical(boxplot_df["Scenario"], categories=ordered_scenarios, ordered=True)

    # Create the Plotly boxplot with the color palette
    fig = go.Figure()

    for scenario in ordered_scenarios:
        group = boxplot_df[boxplot_df["Scenario"] == scenario]
        fig.add_trace(
            go.Box(
                y=group["AIH Value"],
                name=scenario,
                boxpoints=False if scenario == "No Permutations" else "outliers",
                marker=dict(color=color_palette[scenario], size=4, opacity=0.7),
                line=dict(color=color_palette[scenario], width=1.5),
            )
        )

    fig.update_layout(
        #title=title,
        xaxis_title="Scenarios",
        yaxis_title=yaxis_title,
        xaxis=dict(title=dict(font=dict(size=22, weight="bold")), tickfont=dict(size=22)),
        yaxis=dict(title=dict(font=dict(size=22, weight="bold")), tickfont=dict(size=22)),
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background
        paper_bgcolor='rgba(0, 0, 0, 0)',  # Transparent paper background
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, key=unique_key)

def plot_combined_boxplots_with_overlap(
    gini_dict_cumprob, gini_dict_critidx,
    mean_gini_no_permutations_cumprob, mean_gini_no_permutations_critidx,
    selected_category, unique_key
):
    """
    Creates a combined boxplot for Method 1 (AIH) and Method 2 (Criticality Index) for a specific harm category,
    with overlapping boxes for the same scenarios and a properly styled legend.
    """
    combined_data = []

    # Define color palettes for both methods
    color_palette_cumprob = {
        "No Permutations": "#636EFA",
        "1 Permutation": "#EF553B",
        "2 Permutations": "#00CC96",
        "5 Permutations": "#AB63FA",
    }
    color_palette_critidx = {
        "No Permutations": "#B3C6FA",
        "1 Permutation": "#FF999B",
        "2 Permutations": "#7FF7C3",
        "5 Permutations": "#D5B3FA",
    }

    # Add "No Permutations" scenario
    combined_data.append({"Scenario": "No Permutations", "Method": "AIH", "Value": mean_gini_no_permutations_cumprob[selected_category]})
    combined_data.append({"Scenario": "No Permutations", "Method": "Criticality Index", "value": mean_gini_no_permutations_critidx[selected_category]})

    for scenario_name, gini_df_cumprob in gini_dict_cumprob.items():
        descriptive_name = scenario_name.replace("Scenario 1", "1 Permutation").replace("Scenario 2", "2 Permutations").replace("Scenario 3", "5 Permutations")

        if selected_category in gini_dict_critidx[scenario_name].index:
            combined_data.extend([{"Scenario": descriptive_name, "Method": "Criticality Index", "AIH Value": val} for val in gini_dict_critidx[scenario_name].loc[selected_category]])
        if selected_category in gini_df_cumprob.index:
            combined_data.extend([{"Scenario": descriptive_name, "Method": "Cumulative Probability", "AIH Value": val} for val in gini_df_cumprob.loc[selected_category]])

    if not combined_data:
        st.error(f"No AIH data available for category '{selected_category}'.")
        return

    combined_df = pd.DataFrame(combined_data)
    ordered_scenarios = ["No Permutations", "1 Permutation", "2 Permutations", "5 Permutations"]
    combined_df["Scenario"] = pd.Categorical(combined_df["Scenario"], categories=ordered_scenarios, ordered=True)

    fig = go.Figure()

    for method, color_palette in zip(["Cumulative Probability", "Criticality Index"], [color_palette_cumprob, color_palette_critidx]):
        for scenario in ordered_scenarios:
            filtered = combined_df[(combined_df["Scenario"] == scenario) & (combined_df["Method"] == method)]
            fig.add_trace(
                go.Box(
                    y=filtered["AIH Value"],
                    name=scenario,
                    boxpoints="outliers",
                    marker=dict(color=color_palette.get(scenario, "#000000")),
                    line=dict(color=color_palette.get(scenario, "#000000")),
                    legendgroup=method,
                    showlegend=False,
                )
            )

    for method, color_palette in zip(["Cumulative Probability:", "Criticality Index:"], [color_palette_cumprob, color_palette_critidx]):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(color="black"), name=f"{method}", legendgroup=method, showlegend=True))
        for scenario, color in color_palette.items():
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=color, size=12), name=f"  - {scenario}", legendgroup=method, showlegend=True))

    fig.update_layout(
        xaxis_title="Scenarios",
        yaxis_title="AIH / Criticality Index",
        xaxis=dict(title=dict(font=dict(size=22, weight="bold")), tickfont=dict(size=22)),
        yaxis=dict(title=dict(font=dict(size=22, weight="bold")), tickfont=dict(size=22)),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=40, r=40, t=60, b=40),
        width=1200,
        height=900,
        legend=dict(
            title=dict(text="Methods and Scenarios", font=dict(size=22)),
            font=dict(size=22),
            itemsizing="constant",
            x=1.05,
            y=1.0
        ),
    )

    st.plotly_chart(fig, use_container_width=True, key=unique_key)



##### PAGE CONFIG #####
PAGE_TITLE = "AIH Sensitivity Analysis Across Multiple Scenarios"
stut.set_page_config(PAGE_TITLE)

st.markdown("""
    Explore **multiple scenarios**, each with an increasing number of random permutations, to observe AIH sensitivity variations.
""")

# Load data and define categories
option_category = st.selectbox("Select analysis per Harms **Categories** or **Subcategories**:",
                               ["Categories", "Subcategories"])
if option_category == 'Categories':
    df = pd.read_csv('../../results/categories_stakeholders_detailed.csv')
    category_col = 'harm_category'
elif option_category == 'Subcategories':
    df = pd.read_csv('../../results/subcategories_stakeholders_detailed.csv')
    category_col = 'harm_subcategory'
else:
    st.error('Please select a category')
    st.stop()

# Get the list of stakeholders and assign predefined severities
list_of_stakeholders = sorted(df['stakeholders'].unique().tolist())

# Use st.session_state to persist permutations
if "scenarios" not in st.session_state:
    st.session_state.scenarios = generate_scenario_permutations(list_of_stakeholders, num_scenarios=3)

scenarios = st.session_state.scenarios

# Dropdown to select scenario
selected_scenario = st.selectbox("Select a Scenario:", list(scenarios.keys()))
severity_scenarios = scenarios[selected_scenario]

# Define severity_cols after generating severity_scenarios
severity_cols = list(severity_scenarios.keys())

# Convert to DataFrame
severity_df = pd.DataFrame({'stakeholders': list_of_stakeholders, **severity_scenarios})
st.markdown(f"### Generated Permutations for {selected_scenario}")
st.dataframe(severity_df)

# alias for all of our “Perm…” columns
perm_df = severity_df.copy()


# ─── FULL STAKEHOLDER UNIVERSE ────────────────────────────────────────────────
stakeholders = sorted(df['stakeholders'].unique())
# ─── METHOD 1: AIH ───────────────────────────────────────────
st.markdown("## Method 1: AIH", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def compute_gini_df(category_col):
    out = {}
    for name in severity_scenarios:
        # merge in the permuted severity
        df2 = df.merge(
            perm_df[["stakeholders", name]],
            on="stakeholders"
        ).rename(columns={name: "severity"})

        # build cumulative table
        cum = stut.get_cumulative_df_categories(df2, category_col)

        # compute AIH *and* pad with zeros for missing stakeholders
        gser, _ = stut.plot_lorenz_curves_cumulative_probability(
            cum,
            category_col,
            total_stakeholders = len(stakeholders),
            default_severity  = dict(zip(stakeholders, severity_scenarios[name]))
        )
        out[name] = gser

    # drop any all‐NaN columns so stats never blow up
    return pd.DataFrame(out).dropna(axis=1, how="all")

# build & describe
gini_df = compute_gini_df(category_col)
gini_stats = stut.get_gini_stats_df(gini_df)

# display
st.dataframe(gini_df, height=240)
c1, c2 = st.columns([1,2])
st.dataframe(gini_stats, height=240)


# Existing plots (now using our freshly‐computed gini_df)
st.plotly_chart(
    plot_gini_permutation_lineplot(gini_df, title="AIH per Permutation (Method 1)"),
    key="gini_perm_method1"
)
st.plotly_chart(
    plot_mean_gini_line(gini_df.mean(axis=0), title="Mean AIH per Permutation (Method 1)"),
    key="mean_gini_method1"
)
st.plotly_chart(
    plot_boxplot_with_mean_line(gini_df, title="Boxplot of AIH (Method 1)"),
    key="boxplot_method1"
)



##### Main Script #####

# Calculate Gini indices for each scenario
gini_scenario_dict_method1 = calculate_gini_indices_for_scenarios(df, scenarios, 'method1', category_col)
gini_scenario_dict_method2 = calculate_gini_indices_for_scenarios(df, scenarios, 'method2', category_col)

# Dropdown for selecting the harm category for Method 1
selected_category_method1 = st.selectbox("Select a Harm Category for Method 1:", df[category_col].unique())

# ─── RECOMPUTE CROSS-SCENARIO MEANS ────────────────────────────────
mean_gini_df_method1, _ = calculate_mean_gini_across_permutations(
    scenarios,
    calculate_gini_indices_for_scenarios(df, scenarios, 'method1', category_col)
)
mean_gini_df_method2, _ = calculate_mean_gini_across_permutations(
    scenarios,
    calculate_gini_indices_for_scenarios(df, scenarios, 'method2', category_col)
)


mean_gini_df_method1, ci_df_method1 = calculate_mean_gini_across_permutations(
    scenarios,
    calculate_gini_indices_for_scenarios(df, scenarios, 'method1', category_col)
)
mean_gini_df_method2, ci_df_method2 = calculate_mean_gini_across_permutations(
    scenarios,
    calculate_gini_indices_for_scenarios(df, scenarios, 'method2', category_col)
)


# Calculate mean Gini for the "No Permutations" scenario
mean_gini_no_permutations_method1 = mean_gini_df_method1["Original"]
mean_gini_no_permutations_method2 = mean_gini_df_method2["Original"]

# Generate boxplots for Method 1, including "No Permutations"
plot_gini_boxplots_for_scenarios_with_no_permutations_v2(
    gini_scenario_dict_method1,
    mean_gini_no_permutations_method1,
    selected_category_method1,
    #title=f"Gini Distribution for {selected_category_method1} Across Scenarios (Method 1, Including No Permutations)",
    unique_key="boxplot_harm_category_with_no_permutations_method1_v2"
)

# plot “mean Gini” sensitivity for Method 1
st.markdown("### Mean Gini Sensitivity Across Scenarios (Method 1)")
plot_gini_sensitivity_with_ci_for_scenarios(
    mean_gini_df_method1,
    ci_df_method1,
    unique_key="method1_sensitivity_across_scenarios"
)


# ─── SPEARMAN RANK CORRELATION HEATMAP ──────────────────────────────────────

# Add a markdown header like your other sections
st.markdown("### Spearman ρ between Scenarios (Gini Index)")


# 1. compute correlation matrix
corr = mean_gini_df_method1.corr(method='spearman')

# 2. build a Plotly heatmap, but without an internal title
fig = px.imshow(
    corr,
    text_auto=".2f",
    zmin=0.9,
    zmax=1.0,
    color_continuous_scale="Blues"
)

# 3. style axes exactly like your other figures
fig.update_layout(
    xaxis_title="Scenario",
    yaxis_title="Scenario",
    xaxis=dict(
        title=dict(font=dict(size=23, weight="bold")),
        tickfont=dict(size=22)
    ),
    yaxis=dict(
        title=dict(font=dict(size=23, weight="bold")),
        tickfont=dict(size=22)
    ),
    coloraxis_colorbar=dict(
        title="ρ",
        tickfont=dict(size=22),
        title_font=dict(size=23, weight="bold")
    ),
    margin=dict(l=50, r=50, t=20, b=50),
    width=600,
    height=600,
    font=dict(size=22)
)

# 4. render it in Streamlit
st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "<hr style='"
    "border: none;"
    "height: 2px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)

# ─── METHOD 2: Criticality Index ────────────────────────────────────────────
st.markdown("## Method 2: Criticality Index", unsafe_allow_html=True)


# ─── BOX‐WITH‐REFERENCE‐LINE HELPER ────────────────────────────────────────

@st.cache_data(show_spinner=False)
def compute_ci_df(category_col):
    out = {}
    for name in severity_scenarios:
        df2 = df.merge(
            perm_df[["stakeholders", name]],
            on="stakeholders"
        ).rename(columns={name: "severity"})

        cum = stut.get_cumulative_df_categories(df2, category_col)

        # Criticality Index (padding happens inside the function)
        cser = stut.calculate_criticality_index(cum, category_col)
        out[name] = cser

    return pd.DataFrame(out).dropna(axis=1, how="all")

ci_df = compute_ci_df(category_col)

ci_stats = stut.get_gini_stats_df(ci_df)

st.dataframe(ci_df, height=240)
st.dataframe(ci_stats, height=240)

# ─── BOXPLOT WITH REFERENCE LINE ─────────────────────────────────────────────
df_ref = df.copy()
df_ref["severity"] = df_ref["stakeholders"].map(DEFAULT_SEVERITY)
cum_ref = stut.get_cumulative_df_categories(df_ref, category_col)
gini_ref, _ = stut.plot_lorenz_curves_cumulative_probability(
    cum_ref,
    category_col,
    total_stakeholders=len(stakeholders),
    default_severity=DEFAULT_SEVERITY
)


ci_ref = stut.calculate_criticality_index(cum_ref, category_col)




# 2) Permutation‐by‐permutation lineplot (CI)
st.plotly_chart(
    plot_gini_permutation_lineplot(
        ci_df,
        title="CI per Permutation (Method 2)",
        yaxis_title="Criticality Index"
    ),
    key="gini_perm_method2"
)


# 3) Mean‐over‐permutations lineplot (CI)
st.plotly_chart(
    plot_mean_gini_line(
        ci_df.mean(axis=0),
        title="Mean Criticality Index per Permutation (Method 2)",
        yaxis_title="Mean Criticality Index"
    ), key="mean_critidx_method2"
)

st.plotly_chart(
    plot_boxplot_with_mean_line(
        ci_df,
        title="Boxplot of Criticality Index Values (Method 2)",
        yaxis_title="Criticality Index"
    ), key="boxplot_critidx_method2"
)

# 5) “No permutations” boxplot (CI)
selected_category_ci = st.selectbox(
    "Select a Harm Category for Method 2:",
    df[category_col].unique(),
    key="method2_dropdown"
)
plot_gini_boxplots_for_scenarios_with_no_permutations_v2(
    gini_scenario_dict_method2,
        mean_gini_no_permutations_method2,
        selected_category_ci,
        unique_key = "boxplot_no_permutations_method2",
    yaxis_title = "Criticality Index"
)

# plot “mean CI” sensitivity for Method 2
st.markdown("### Mean CI Sensitivity Across Scenarios (Method 2)")
plot_gini_sensitivity_with_ci_for_scenarios(
    mean_gini_df_method2,
    ci_df_method2,
    unique_key="method2_sensitivity_across_scenarios",
    yaxis_title="Mean Criticality Index"
)




# ─── SPEARMAN ρ FOR CRITICALITY-INDEX HEATMAP ────────────────────────────────
st.markdown("### Spearman ρ between Scenarios (Criticality Index)")


corr_ci = mean_gini_df_method2.corr(method='spearman')

import plotly.express as px
fig_ci = px.imshow(
    corr_ci,
    text_auto=".2f",
    zmin=0.9,
    zmax=1.0,
    color_continuous_scale="Blues"
)

fig_ci.update_layout(
    xaxis_title="Scenario",
    yaxis_title="Scenario",
    xaxis=dict(title=dict(font=dict(size=23, weight="bold")), tickfont=dict(size=22)),
    yaxis=dict(title=dict(font=dict(size=23, weight="bold")), tickfont=dict(size=22)),
    coloraxis_colorbar=dict(title="ρ", tickfont=dict(size=22), title_font=dict(size=23, weight="bold")),
    margin=dict(l=50, r=50, t=20, b=50),
    width=600,
    height=600,
    font=dict(size=22)
)

st.plotly_chart(
    fig_ci,
    use_container_width=True,
    key="spearman_ci"        # ← make this unique
)
st.markdown(
    "<hr style='border:none;height:2px;background-color:#333;margin:2rem 0;'/>",
    unsafe_allow_html=True
)


st.markdown(
    "<hr style='"
    "border: none;"
    "height: 2px;"
    "background-color: #333;"
    "margin: 2rem 0;"
    "'/>",
    unsafe_allow_html=True
)


st.markdown(
    "<hr style='"
    "border: none;"
    "height: 4px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)


# ─── COMBINED BOXPLOT: Method 1 vs Method 2 ───────────────────────
selected_category_combined = st.selectbox(
    "Select a Harm Category for Combined Comparison:",
    df[category_col].unique(),
    key="combined_dropdown"
)
plot_combined_boxplots_with_overlap(
    gini_scenario_dict_method1,
    gini_scenario_dict_method2,
    mean_gini_no_permutations_method1,
    mean_gini_no_permutations_method2,
    selected_category_combined,
    unique_key="combined_boxplot_methods_overlap"
)

# ─── SCATTER PLOT COMPARISON ────────────────────────────────────────────────
st.markdown("### Scatter Plot: AIH vs Criticality Index", unsafe_allow_html=True)

# grab the scenario‐specific AIH & CI tables
df1 = gini_scenario_dict_method1[selected_scenario]
df2 = gini_scenario_dict_method2[selected_scenario]

# compute per‐category means
mean_method1 = df1.mean(axis=1)
mean_method2 = df2.mean(axis=1)

# plot
scatter_ci = stut.plot_gini_vs_criticality(
    gini_series=mean_method1,
    criticality_series=mean_method2
)
st.plotly_chart(scatter_ci, key="gini_vs_criticality_comparison")



st.markdown(
    "<hr style='"
    "border: none;"
    "height: 2px;"          # thickness
    "background-color: #333;"  # color
    "margin: 2rem 0;"       # vertical spacing
    "'/>",
    unsafe_allow_html=True
)




# ─── SUBCATEGORY DRILL-DOWN (only when “Subcategories” mode is active) ─────
if option_category == 'Subcategories':
    # load the precomputed mapping
    mapping_df = pd.read_csv('../../results/category_subcategory_mapping.csv')

    st.markdown("## Subcategory Drill-Down")

    # pick a top-level category
    top_category = st.selectbox(
        "Choose High-Level Category:",
        mapping_df['harm_category'].unique(),
        key='drill_top_category'
    )

    # get its subcategories
    subs = (
        mapping_df
        .loc[mapping_df['harm_category'] == top_category, 'harm_subcategory']
        .unique()
        .tolist()
    )
    if not subs:
        st.warning(f"No subcategories found for {top_category!r}.")
    else:
        # let the user pick one or “All”
        choice = st.selectbox(
            "Choose Subcategory:",
            ['All'] + subs,
            key='drill_subcategory'
        )
        targets = subs if choice == 'All' else [choice]

        # helper to filter your AIH/CI dicts by index
        def _filter(gdict):
            return {
                sc: gdf.loc[gdf.index.isin(targets)]
                for sc, gdf in gdict.items()
            }

        fm1 = _filter(gini_scenario_dict_method1)
        fm2 = _filter(gini_scenario_dict_method2)

        # pick out the “no-perm” means
        mnp1 = mean_gini_no_permutations_method1.loc[targets]
        mnp2 = mean_gini_no_permutations_method2.loc[targets]

        # ─ Method 1 drill-down plots ─────────────────────────
        st.markdown("### Method 1: AIH (Selected Subcategories)")
        plot_gini_boxplots_for_scenarios_with_no_permutations_v2(
            fm1, mnp1, targets[0], unique_key='drill_boxplot_m1'
        )
        plot_gini_sensitivity_with_ci_for_scenarios(
            *calculate_mean_gini_across_permutations(scenarios, fm1),
            unique_key='drill_sensitivity_m1'
        )

        # ─ Method 2 drill-down plots ─────────────────────────
        st.markdown("### Method 2: Criticality Index (Selected Subcategories)")
        plot_gini_boxplots_for_scenarios_with_no_permutations_v2(
            fm2, mnp2, targets[0], unique_key='drill_boxplot_m2'
        )
        plot_gini_sensitivity_with_ci_for_scenarios(
            *calculate_mean_gini_across_permutations(scenarios, fm2),
            unique_key='drill_sensitivity_m2'
        )

        # ─ Combined scatter ─────────────────────────────────
        st.markdown("### Scatter: AIH vs Criticality (Selected Subcategories)")
        df1_sub = fm1[selected_scenario]
        df2_sub = fm2[selected_scenario]
        mean1 = df1_sub.mean(axis=1).loc[targets]
        mean2 = df2_sub.mean(axis=1).loc[targets]
        st.plotly_chart(
            stut.plot_gini_vs_criticality(mean1, mean2),
            key='drill_scatter'
        )
