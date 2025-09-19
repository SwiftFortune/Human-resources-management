# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Industry Clustering Dashboard",
                   layout="wide",
                   page_icon="📊")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\sachin\Python\Human Resource Management\industry_clusters.csv")

data = load_data()

# -----------------------------
# Rename Clusters
# -----------------------------
cluster_names = {
    0: "Retail",
    1: "Poultry",
    2: "Agriculture",
    3: "Manufacturing",
    4: "Services",
    5: "Others"
}
data["Cluster_Name"] = data["Cluster"].map(cluster_names)

# -----------------------------
# Sidebar - Questions
# -----------------------------
st.sidebar.title("🔎 Explore Questions")
QUESTIONS = [
    "Which industries dominate each cluster?",
    "Retail businesses by state",
    "Poultry businesses by state",
    "Manufacturing businesses by state",
    "Cluster distribution pie chart",
    "Cluster distribution donut chart",
    "Workers stacked bar chart by cluster",
    "Correlation heatmap of numeric features",
    "WordCloud: Top terms in Retail cluster",
    "WordCloud: Top terms in Poultry cluster",
    "WordCloud: Top terms in Agriculture cluster",
    "WordCloud: Top terms in Manufacturing cluster",
    "Top 10 industries per cluster",
    "Cluster vs Group comparison",
    "Cluster vs Division comparison",
    "Top 5 states per cluster",
    "Distribution of marginal workers",
    "Distribution of main workers",
    "Cluster boxplot of workers",
    "Pairplot of numeric features",
    "Bar chart of industries by Class",
    "Map showing all clusters together",
    "Line chart: State vs Workers count",
    "Bubble chart: State vs Workers vs Cluster",
    "Treemap: Industries per cluster",
    "Sunburst chart of state-cluster breakdown",
    "Time trend (dummy) for industries"
]
question_selected = st.sidebar.radio("Select a Question:", QUESTIONS)

# -----------------------------
# Title
# -----------------------------
st.title("📊 Industry Clustering Dashboard")
st.markdown("Interactive maps, charts, and tables of Indian industries, clusters, and states.")

# =============================
# Dataset Preview
# =============================
with st.expander("📋 Dataset Preview"):
    st.dataframe(data.head(20))

# =============================
# Helper Function for Maps
# =============================
def plot_india_state_map_safe(cluster_name):
    df = data[data["Cluster_Name"] == cluster_name].groupby("India/States").size().reset_index(name="Count")
    try:
        fig = px.choropleth(
            df,
            locations="India/States",
            locationmode="country names",
            color="Count",
            hover_name="India/States",
            title=f"{cluster_name} Businesses by State",
            color_continuous_scale="Viridis"
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Map could not be rendered. Showing table instead.")
        st.table(df)
    st.dataframe(df)

# =============================
# Main Question Handling
# =============================
if question_selected == "Which industries dominate each cluster?":
    fig = px.histogram(data, x="Cluster_Name", color="Cluster_Name", title="Industries per Cluster")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(data["Cluster_Name"].value_counts().reset_index().rename(columns={"index":"Cluster","Cluster_Name":"Count"}))

elif question_selected == "Retail businesses by state":
    plot_india_state_map_safe("Retail")

elif question_selected == "Poultry businesses by state":
    plot_india_state_map_safe("Poultry")

elif question_selected == "Manufacturing businesses by state":
    plot_india_state_map_safe("Manufacturing")

elif question_selected in ["Cluster distribution pie chart", "Cluster distribution donut chart"]:
    hole = 0.4 if "donut" in question_selected.lower() else 0
    fig = px.pie(data, names="Cluster_Name", hole=hole, title=question_selected)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(data["Cluster_Name"].value_counts().reset_index().rename(columns={"index":"Cluster","Cluster_Name":"Count"}))

elif question_selected == "Workers stacked bar chart by cluster":
    worker_cols = [c for c in data.columns if "Workers" in c]
    if worker_cols:
        cluster_workers = data.groupby("Cluster_Name")[worker_cols].sum().reset_index()
        cluster_workers_long = cluster_workers.melt(id_vars="Cluster_Name", var_name="Worker Type", value_name="Count")
        fig = px.bar(cluster_workers_long, x="Cluster_Name", y="Count", color="Worker Type", barmode="stack",
                     title="Workers per Cluster")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cluster_workers)
    else:
        st.warning("No worker-related columns found.")

elif question_selected == "Correlation heatmap of numeric features":
    numeric_cols = data.select_dtypes(include=["int64","float64"]).columns
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns to plot heatmap. Showing correlation table instead.")
        st.dataframe(data[numeric_cols].corr())
    else:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

elif "WordCloud" in question_selected:
    cluster = question_selected.split(" ")[-2]
    text = " ".join(data[data["Cluster_Name"]==cluster]["NIC Name"].astype(str))
    if len(text.split()) < 5:
        st.warning("Not enough text to generate WordCloud. Showing top terms table instead.")
        top_terms = pd.Series(text.split()).value_counts().head(10)
        st.table(top_terms)
    else:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        st.image(wc.to_array(), use_column_width=True)

elif question_selected == "Top 10 industries per cluster":
    for cluster in data["Cluster_Name"].unique():
        top10 = data[data["Cluster_Name"]==cluster]["NIC Name"].value_counts().head(10)
        st.subheader(f"Top 10 Industries in {cluster} Cluster")
        fig = px.bar(top10, x=top10.index, y=top10.values, labels={'x':'Industry','y':'Count'})
        st.plotly_chart(fig, use_container_width=True)

elif question_selected == "Cluster vs Group comparison":
    if "Group" in data.columns:
        fig = px.box(data, x="Cluster_Name", y="Group", title="Cluster vs Group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Column 'Group' not found. Showing cluster counts instead.")
        st.dataframe(data["Cluster_Name"].value_counts())

elif question_selected == "Cluster vs Division comparison":
    if "Division" in data.columns:
        fig = px.box(data, x="Cluster_Name", y="Division", title="Cluster vs Division")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Column 'Division' not found. Showing cluster counts instead.")
        st.dataframe(data["Cluster_Name"].value_counts())

elif question_selected == "Top 5 states per cluster":
    for cluster in data["Cluster_Name"].unique():
        top_states = data[data["Cluster_Name"]==cluster]["India/States"].value_counts().head(5)
        st.subheader(f"Top 5 States in {cluster} Cluster")
        st.table(top_states)

elif question_selected == "Distribution of marginal workers":
    worker_col = "Marginal Workers - Total -  Persons"
    if worker_col in data.columns:
        fig = px.histogram(data, x=worker_col, color="Cluster_Name", title="Distribution of Marginal Workers")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Column '{worker_col}' not found.")

elif question_selected == "Distribution of main workers":
    worker_col = "Main Workers - Total -  Persons"
    if worker_col in data.columns:
        fig = px.histogram(data, x=worker_col, color="Cluster_Name", title="Distribution of Main Workers")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Column '{worker_col}' not found.")

elif question_selected == "Cluster boxplot of workers":
    worker_cols = [c for c in data.columns if "Workers" in c]
    if worker_cols:
        fig = px.box(data.melt(id_vars=["Cluster_Name"], value_vars=worker_cols, var_name="Worker Type", value_name="Count"),
                     x="Cluster_Name", y="Count", color="Worker Type", title="Cluster Boxplot of Workers")
        st.plotly_chart(fig, use_container_width=True)

elif question_selected == "Pairplot of numeric features":
    numeric_cols = data.select_dtypes(include=["int64","float64"]).columns
    if len(numeric_cols) > 5:
        st.warning("Too many numeric columns for pairplot. Showing correlation table instead.")
        st.dataframe(data[numeric_cols].corr())
    else:
        fig = sns.pairplot(data[numeric_cols])
        st.pyplot(fig)

elif question_selected == "Bar chart of industries by Class":
    if "Class" in data.columns:
        class_count = data.groupby("Class")["NIC Name"].count()
        fig = px.bar(class_count, x=class_count.index, y=class_count.values, labels={'x':'Class','y':'Count'}, title="Industries by Class")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Column 'Class' not found.")

elif question_selected == "Map showing all clusters together":
    df = data.groupby(["India/States","Cluster_Name"]).size().reset_index(name="Count")
    try:
        fig = px.choropleth(df, locations="India/States", locationmode="country names",
                            color="Cluster_Name", hover_name="India/States", title="All Clusters by State")
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Map could not be rendered. Showing table instead.")
        st.table(df)
    st.dataframe(df)

elif question_selected == "Line chart: State vs Workers count":
    worker_cols = [c for c in data.columns if "Workers" in c]
    if worker_cols:
        state_workers = data.groupby("India/States")[worker_cols].sum().sum(axis=1).reset_index()
        state_workers.columns = ["India/States","Total Workers"]
        try:
            fig = px.line(state_workers, x="India/States", y="Total Workers", title="State vs Total Workers")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Line chart could not be rendered. Showing table instead.")
            st.table(state_workers)

elif question_selected == "Bubble chart: State vs Workers vs Cluster":
    worker_cols = [c for c in data.columns if "Workers" in c]
    if worker_cols:
        df = data.groupby(["India/States","Cluster_Name"])[worker_cols].sum().sum(axis=1).reset_index()
        df.columns = ["India/States","Cluster_Name","Total Workers"]
        try:
            fig = px.scatter(df, x="India/States", y="Total Workers", size="Total Workers", color="Cluster_Name", title="Bubble chart")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Bubble chart could not be rendered. Showing table instead.")
            st.table(df)

elif question_selected == "Treemap: Industries per cluster":
    try:
        fig = px.treemap(data, path=["Cluster_Name","NIC Name"], title="Treemap: Industries per cluster")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Treemap could not be rendered. Showing table instead.")
        st.table(data[["Cluster_Name","NIC Name"]])

elif question_selected == "Sunburst chart of state-cluster breakdown":
    try:
        fig = px.sunburst(data, path=["India/States","Cluster_Name"], title="Sunburst chart")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.warning("Sunburst chart could not be rendered. Showing table instead.")
        st.table(data[["India/States","Cluster_Name"]])

elif question_selected == "Time trend (dummy) for industries":
    st.info("Time trend chart is a placeholder. Showing industry counts instead.")
    industry_counts = data["NIC Name"].value_counts().head(20)
    fig = px.bar(industry_counts, x=industry_counts.index, y=industry_counts.values, title="Industry counts")
    st.plotly_chart(fig, use_container_width=True)
