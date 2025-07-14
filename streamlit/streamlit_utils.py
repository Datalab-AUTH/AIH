import streamlit as st
from PIL import Image
import io
import pandas as pd
import numpy as np
import plotly.graph_objs as go

MAX_NB_DFs = 10
TTL_CACHED = 300  # 5 minutes

def set_page_config(page_title):
    logo = Image.open('./content/datalab_logo.png')
    st.set_page_config(layout="wide",
                       page_title=page_title,
                       page_icon=logo)
    st.image(logo, width=200)
    st.title(page_title)
    st.info(
        "This web app and its data analysis are experimental. Use the results responsibly. "
        "We hold no responsibility for their accuracy or any consequences arising from their use.",
        icon="ℹ️"
    )

    st.sidebar.markdown(
        """
        <style>
        [data-testid='stSidebarNav'] > ul {
            min-height: 22em;
        } 
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_data(show_spinner="Creating file ...", ttl=TTL_CACHED)
def convert_df(df, file_type):
    if file_type == 'csv':
        data_file = df.to_csv(index=False).encode('utf-8')
    elif file_type == 'xlsx':
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer) as writer:
            df.to_excel(writer, index=False)
            writer.save()
            data_file = buffer
    else:
        raise Exception('Not implemented file type')
    return data_file

def download_button_block(df, group_by, key, file_name, file_types=['csv', 'xlsx']):
    file_type = st.radio(label='Select file type:', options=file_types, index=0, horizontal=True,
                         key=f"radio-button-{key}")
    data_file = convert_df(df, file_type=file_type)
    st.download_button(label="Download data",
                       data=data_file,
                       file_name=file_name + '.' + file_type,
                       key=f"download-button-{key}")

def check_credentials(username, password):
    if (username in st.secrets["passwords"]) and (password == st.secrets["passwords"][username]):
        return True
    else:
        return False

def check_password():
    if ("password_correct" in st.session_state) and (st.session_state["password_correct"]):
        return True
    else:
        if ("username" in st.session_state) \
           and ("password" in st.session_state) \
           and check_credentials(st.session_state["username"], st.session_state["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
            del st.session_state["username"]
            return True
        else:
            st.session_state["password_correct"] = False
            return False

def get_cumulative_df_categories(df, category_col):
    cumulative_list = []
    for category in df[category_col].unique():
        df_category = df[df[category_col] == category].copy()
        df_category = df_category.sort_values(by='severity')

        df_category['percentage_freq'] = df_category['freq'] / df_category['freq'].sum()
        df_category['cumulative freq'] = df_category['percentage_freq'].cumsum()

        df_category['perc_freq_x_severity'] = df_category['percentage_freq'] * df_category['severity']
        df_category['cumulative freq x severity'] = (
            df_category['perc_freq_x_severity'].cumsum() / df_category['perc_freq_x_severity'].sum()
        )

        cumulative_list.append(df_category)

    df_with_cumulative = pd.concat(cumulative_list, ignore_index=True)
    return df_with_cumulative





def plot_lorenz_curves_cumulative_probability(df, category_col, total_stakeholders=None, default_severity=None):
    """
    Plot Lorenz curves using cumulative probability method and calculate Gini.
    """
    fig = go.Figure()
    gini_vals = {}

    for category in df[category_col].unique():
        df_category = df[df[category_col] == category].copy()

        # Ensure missing stakeholders are included with default severity
        if default_severity is not None:
            missing_stakeholders = set(default_severity.keys()) - set(df_category['stakeholders'])
            for stakeholder in missing_stakeholders:
                df_category = pd.concat([df_category, pd.DataFrame([{
                    'stakeholders': stakeholder,
                    'freq': 0,
                    'severity': default_severity[stakeholder],
                    category_col: category
                }])], ignore_index=True)

        df_category = df_category.sort_values(by='severity')

        # Compute cumulative shares
        df_category['cumulative_share_counts'] = df_category['freq'].cumsum() / df_category['freq'].sum()
        df_category['severity_rank'] = df_category['severity'].rank(method='first')
        df_category['normalized_severity_rank'] = df_category['severity_rank'] / len(df_category)

        # Extract X and Y values for Lorenz curve
        x_vals = [0] + df_category['cumulative_share_counts'].tolist()
        y_vals = [0] + df_category['normalized_severity_rank'].tolist()

        # Compute AUC and Gini
        auc = np.trapz(y=y_vals, x=x_vals)
        gini_vals[category] = auc

        # Add Lorenz curve to the figure
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='lines+markers',
            name=category,
            line=dict(width=3),
        ))

    # Add equality line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='black'),
        name="Equality Line"
    ))

    # Update layout with larger ticks
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Cumulative Share of Stakeholders",
                font=dict(size=22, weight='bold')
            ),
            tickfont=dict(size=22, color='#666666'),
            showgrid=True,
            gridcolor='#DDDDDD',
            zeroline=False
        ),
        yaxis=dict(
            title=dict(
                text="Cumulative Probability",
                font=dict(size=22, weight='bold')
            ),
            tickfont=dict(size=22, color='#666666'),
            showgrid=True,
            gridcolor='#DDDDDD',
            zeroline=False
        ),
        legend=dict(
            font=dict(size=20),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#DDDDDD',
            borderwidth=1
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=50, r=50, t=60, b=50)
    )

    return pd.Series(gini_vals, name="AIH"), fig

def plot_mean_gini_scatter(mean_gini_method1, mean_gini_method2, categories, lorenz_curve_figure):
    """
    Create a scatter plot comparing Gini indices for two methods.
    Automatically match dot colors to the Lorenz curve colors.
    """
    fig = go.Figure()

    # Extract colors from the Lorenz curve figure
    trace_colors = {trace['name']: trace['line']['color'] for trace in lorenz_curve_figure['data'] if 'line' in trace}

    # Add scatter plot points with matching colors
    for category, x_val, y_val in zip(categories, mean_gini_method1, mean_gini_method2):
        color = trace_colors.get(category, '#000000')  # Default to black if no color is found
        fig.add_trace(go.Scatter(
            x=[x_val],
            y=[y_val],
            mode='markers+text',
            text=[category],
            textposition='top center',
            textfont=dict(size=19),
            marker=dict(
                size=16,
                color=color,
                line=dict(width=0, color='white')
            ),
            name=category
        ))

    # Add 45-degree line
    max_gini = max(max(mean_gini_method1), max(mean_gini_method2)) + 0.05
    fig.add_trace(go.Scatter(
        x=[0, max_gini],
        y=[0, max_gini],
        mode='lines',
        line=dict(dash='dash', color='#999999', width=3),
        name="45-degree line"
    ))

    # Update layout
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="AIH",
                font=dict(size=23, weight='bold'),

            ),
            tickfont=dict(size=22, color='#666666'),
            showgrid=True,
            gridcolor='#DDDDDD',
            range=[0, max_gini]
        ),
        yaxis=dict(
            title=dict(
                text="AIH",
                font=dict(size=23, weight='bold'),
            ),
            tickfont=dict(size=22, color='#666666'),
            showgrid=True,
            gridcolor='#DDDDDD',
            range=[0, max_gini]
        ),
        legend=dict(
            font=dict(size=20),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#DDDDDD',
            borderwidth=1
        ),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(255, 255, 255, 1)',
        margin=dict(l=50, r=50, t=60, b=50),
        width=1350,
        height=980
    )

    return fig



def get_gini_stats_df(df):
    gini_df_stats = pd.DataFrame()
    gini_df_stats['MEAN'] = df.mean(axis=1)
    gini_df_stats['VARIANCE'] = df.var(axis=1)
    gini_df_stats['CONF. INTERVAL'] = 1.96 * df.std(axis=1) / np.sqrt(df.shape[1])
    gini_df_stats['VaR (95%)'] = df.apply(lambda x: np.percentile(x.dropna(), 95), axis=1)
    return gini_df_stats

def calculate_criticality_index(df, category_col, verbose=False):
    """
    Compute the Criticality Index per category using:
    CI = sum(F_k) / M
    where F_k is the cumulative frequency of stakeholder k and M is total number of stakeholders.
    If verbose=True, returns a debug dict with intermediate p_k and F_k values for inspection.
    """
    ci_vals = {}
    debug_info = {}

    all_stakeholders = df['stakeholders'].unique()

    for cat in df[category_col].unique():
        df_cat = df[df[category_col] == cat].copy()

        # Ensure all stakeholders are included
        missing = set(all_stakeholders) - set(df_cat['stakeholders'])
        for s in missing:
            df_cat = pd.concat([df_cat, pd.DataFrame([{
                category_col: cat,
                'stakeholders': s,
                'freq': 0,
                'severity': df[df['stakeholders'] == s]['severity'].iloc[0] if not df[df['stakeholders'] == s].empty else 0
            }])], ignore_index=True)

        # Sort stakeholders by severity ascending
        df_cat = df_cat.sort_values(by='severity')

        # Calculate p_k and F_k
        total_freq = df_cat['freq'].sum()
        if total_freq > 0:
            df_cat['p_k'] = df_cat['freq'] / total_freq
        else:
            df_cat['p_k'] = 0

        df_cat['F_k'] = df_cat['p_k'].cumsum()

        # Save for debug
        if verbose:
            debug_info[cat] = df_cat[['stakeholders', 'freq', 'p_k', 'F_k']]

        # Final Criticality Index: mean of F_k
        ci_vals[cat] = 1 - ((df_cat['F_k'].sum() - 1) / (len(df_cat) - 1))

    result = pd.Series(ci_vals, name="Criticality Index")

    if verbose:
        return result, debug_info
    else:
        return result

def plot_mean_errorbar_with_var(df):
    """
    Creates a bar chart showing mean index values with confidence intervals and VaR (95%).
    Displays both positive and negative values.
    """
    fig = go.Figure()

    # Add Mean with Confidence Intervals
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MEAN'].values,
        error_y=dict(
            type='data',
            array=df['CONF. INTERVAL'].values,
            color='blue'
        ),
        marker=dict(color='#0072B2'),
        text=df['MEAN'].round(2),
        textposition='outside',
        name='Mean'
    ))

    # Add VaR (95%) markers
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['VaR (95%)'].values,
        mode='lines+markers',
        name='VaR (95%)',
        line=dict(color='red', dash='dot', width=2),
        marker=dict(size=8, symbol='circle')
    ))

    # Update layout
    y_min = min(df['MEAN'].min(), df['VaR (95%)'].min()) - 0.05
    y_max = max(df['MEAN'].max(), df['VaR (95%)'].max()) + 0.05

    fig.update_layout(
        xaxis_title="Categories",
        yaxis_title="Index Value",
        xaxis=dict(title=dict(font=dict(size=19, weight='bold'))),
        yaxis=dict(title=dict(font=dict(size=19, weight='bold'))),
        plot_bgcolor='#F8F8F8',
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=True
    )
    fig.update_yaxes(range=[y_min, y_max])

    return fig

def plot_gini_vs_criticality(gini_series, criticality_series):
    """
    Scatter plot comparing Gini (ordinal) vs Criticality Index
    with on-chart labels, and axis fonts matching other plots.
    """
    fig = go.Figure()
    max_val = max(gini_series.max(), criticality_series.max()) + 0.05

    # cycle through these positions for label placement:
    positions = ['top center', 'bottom center', 'middle right', 'middle left']

    for i, cat in enumerate(gini_series.index):
        x = criticality_series[cat]
        y = gini_series[cat]
        pos = positions[i % len(positions)]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=16),
            text=[cat],
            textposition=pos,
            textfont=dict(size=18),
            name=cat,
            showlegend=False
        ))

    # 45° reference line (hidden from legend)
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        showlegend=False
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Criticality Index",
                font=dict(size=23, weight='bold')
            ),
            tickfont=dict(size=22, color='#666666'),
            range=[0, max_val],
            showgrid=True,
            gridcolor='#DDDDDD'
        ),
        yaxis=dict(
            title=dict(
                text="AIH",
                font=dict(size=23, weight='bold')
            ),
            tickfont=dict(size=22, color='#666666'),
            range=[0, max_val],
            showgrid=True,
            gridcolor='#DDDDDD'
        ),
        width=1350,
        height=980,
        plot_bgcolor='white',
        legend=dict(
            font=dict(size=20),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#DDD',
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )

    return fig
