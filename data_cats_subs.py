import pandas as pd
filename = "./data/AI_Harm_Taxonomy_Streamlit - Annotations__12-07-2024.csv"
df = pd.read_csv(filename)
df

group_columns = ['annotator','incident_ID','harm_category']
df_categories = df.groupby(by=group_columns).count().reset_index()[group_columns]
df_categories

# df_categories_freq = df_categories['harm_category'].value_counts()
df_categories_freq = df_categories.groupby(by='harm_category').size().reset_index().rename(columns={0:'freq'})
df_categories_freq

group_columns = ['annotator','incident_ID','harm_subcategory']
df_subcategories = df.groupby(by=group_columns).size().reset_index()[group_columns]
df_subcategories

# df_subcategories_freq = df_subcategories['harm_subcategory'].value_counts()
df_subcategories_freq = df_subcategories.groupby(by='harm_subcategory').size().reset_index().rename(columns={0:'freq'})
df_subcategories_freq

group_columns = ['annotator','incident_ID','harm_category','stakeholders']
df_categories_stakeholders = df.groupby(by=group_columns).size().reset_index()[group_columns]
df_categories_stakeholders

df_categories_stakeholders_total = df_categories_stakeholders[['harm_category','stakeholders']].groupby(by='harm_category').count().reset_index()
df_categories_stakeholders_total

df_categories_freq_stakeholders = df_categories_freq.merge(df_categories_stakeholders_total,on='harm_category')
df_categories_freq_stakeholders

#subcategories
group_columns = ['annotator','incident_ID','harm_subcategory','stakeholders']
df_subcategories_stakeholders = df.groupby(by=group_columns).size().reset_index()[group_columns]
df_subcategories_stakeholders

df_subcategories_stakeholders_total = df_subcategories_stakeholders[['harm_subcategory','stakeholders']].groupby(by='harm_subcategory').count().reset_index()
df_subcategories_stakeholders_total

df_subcategories_freq_stakeholders = df_subcategories_freq.merge(df_subcategories_stakeholders_total,on='harm_subcategory')
df_subcategories_freq_stakeholders


# Extra data granularity: per category and stakeholder

def get_freq_per_category_and_stakeholder_detailed(df, category_col):
    '''
    category_col: 'harm_category' or 'harm_subcategory'
    '''
    group_columns = ['annotator','incident_ID',category_col,'stakeholders']
    _df = df.groupby(by=group_columns).size().reset_index()[group_columns]
    return _df.groupby(by=[category_col,'stakeholders']).size().reset_index().rename(columns={0:'freq'})

df_categories_stakeholders_detailed = get_freq_per_category_and_stakeholder_detailed(df, 'harm_category')
df_categories_stakeholders_detailed


df_subcategories_stakeholders_detailed = get_freq_per_category_and_stakeholder_detailed(df, 'harm_subcategory')
df_subcategories_stakeholders_detailed


# … everything you already have …

# -------------------------------------------------------------------
# Finally, save a lookup of which subcategories belong to which categories
mapping = df[['harm_category','harm_subcategory']].drop_duplicates()
mapping.to_csv('results/category_subcategory_mapping.csv', index=False)
