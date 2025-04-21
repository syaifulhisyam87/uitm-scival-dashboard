import streamlit as st
import pandas as pd
import altair as alt

# Page configuration
st.set_page_config(
    page_title='UiTM SciVal Dashboard',
    layout='wide',
)

@st.cache_data
# Load data from the cleaned SciVal export
def load_data():
    df = pd.read_csv('All_Topic_Clusters_Cleaned.csv')
    return df

# Load the dataset
df = load_data()

# Dashboard Title
st.title('UiTM SciVal Dashboard')
st.markdown(
    """
    This interactive dashboard visualizes key research impact metrics
    from the UiTM SciVal dataset, allowing exploration of scholarly output,
    citation impact, prominence, and publication growth across topic clusters.
    """
)

# Sidebar filters
st.sidebar.header('Filters')

# Scholarly Output filter
min_output = int(df['Scholarly Output'].min())
max_output = int(df['Scholarly Output'].max())
output_range = st.sidebar.slider(
    'Scholarly Output',
    min_output, max_output,
    (min_output, max_output)
)

# Field-Weighted Citation Impact filter
min_fwci = float(df['Field-Weighted Citation Impact'].min())
max_fwci = float(df['Field-Weighted Citation Impact'].max())
fwci_range = st.sidebar.slider(
    'Field-Weighted Citation Impact (FWCI)',
    min_fwci, max_fwci,
    (min_fwci, max_fwci)
)

# Topic Cluster selection
cluster_options = df['Topic Cluster'].unique()
selected_clusters = st.sidebar.multiselect(
    'Topic Clusters', cluster_options, default=[]
)

# Apply filters to the dataframe
filtered_df = df[
    (df['Scholarly Output'] >= output_range[0]) &
    (df['Scholarly Output'] <= output_range[1]) &
    (df['Field-Weighted Citation Impact'] >= fwci_range[0]) &
    (df['Field-Weighted Citation Impact'] <= fwci_range[1])
]
if selected_clusters:
    filtered_df = filtered_df[filtered_df['Topic Cluster'].isin(selected_clusters)]

# Display filtered data table
st.subheader('Filtered Dataset')
st.dataframe(filtered_df)

# Chart 1: Top 10 Topic Clusters by Scholarly Output
st.subheader('Top 10 Topic Clusters by Scholarly Output')
top_output = filtered_df.nlargest(10, 'Scholarly Output')
bar_output = alt.Chart(top_output).mark_bar().encode(
    x=alt.X('Scholarly Output:Q', title='Scholarly Output'),
    y=alt.Y('Topic Cluster:N', sort='-x', title='Topic Cluster'),
    tooltip=['Topic Cluster', 'Scholarly Output']
).properties(width=700, height=400)
st.altair_chart(bar_output, use_container_width=True)

# Chart 2: FWCI vs Prominence Scatter Plot
st.subheader('FWCI vs Prominence Percentile')
scatter = alt.Chart(filtered_df).mark_circle(opacity=0.7, size=100).encode(
    x=alt.X('Field-Weighted Citation Impact:Q', title='FWCI'),
    y=alt.Y('Prominence percentile:Q', title='Prominence Percentile'),
    color=alt.Color('Scholarly Output:Q', title='Scholarly Output'),
    tooltip=['Topic Cluster', 'Field-Weighted Citation Impact', 'Prominence percentile', 'Scholarly Output']
).properties(width=700, height=400)
st.altair_chart(scatter, use_container_width=True)

# Chart 3: Top 10 Topic Clusters by Publication Share Growth
st.subheader('Top 10 Topic Clusters by Publication Share Growth (%)')
top_growth = filtered_df.nlargest(10, 'Publication Share growth (%)')
bar_growth = alt.Chart(top_growth).mark_bar(color='purple').encode(
    x=alt.X('Publication Share growth (%):Q', title='Growth (%)'),
    y=alt.Y('Topic Cluster:N', sort='-x', title='Topic Cluster'),
    tooltip=['Topic Cluster', 'Publication Share growth (%)']
).properties(width=700, height=400)
st.altair_chart(bar_growth, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Dashboard generated using Streamlit and Altair.*")
