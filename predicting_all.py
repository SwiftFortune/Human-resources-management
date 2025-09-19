# complete_hr_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Industrial Human Resource Dashboard",
                   layout="wide",
                   page_icon="📊")

# -------------------------------
# Sidebar navigation
# -------------------------------
section = st.sidebar.radio(
    "📂 Navigate Sections",
    ["Introduction", "Data Visualization", "Total Worker Prediction",
     "Industry Classification Prediction", "Applications & Use Cases"]
)

# -------------------------------
# 1. Introduction Section
# -------------------------------
if section == "Introduction":
    st.title("🏭 Industrial Human Resource Geo-Visualization")
    st.markdown("""
    **Technologies:** EDA, Visualization, NLP  
    **Domain:** Resource Management
    """)

    st.subheader("Problem Statement")
    st.markdown("""
    In India, the industrial classification of the workforce is essential to understand the distribution of the labor force across various sectors.  
    The classification of main workers and marginal workers, other than cultivators and agricultural laborers, by sex and by section, division, and class, 
    has been traditionally used to understand the economic status and employment trends in the country.  

    However, the current data on this classification is outdated and may not accurately reflect the current state of the workforce.  
    The aim of this study is to update the information on the industrial classification of the main and marginal workers, other than cultivators and agricultural laborers, 
    by sex and by section, division, and class, to provide relevant and accurate data for policy making and employment planning.
    """)

    st.subheader("Importance & Use Cases")
    st.markdown("""
    This industrial HR analysis is used to:
    - **Government & Policy Making:** Allocate resources efficiently and design employment policies.
    - **Human Resource Planning:** Identify workforce trends and labor shortages.
    - **Industry Analysis:** Understand which sectors dominate in different regions.
    - **Economic Research:** Study employment patterns and economic health of various states.
    - **Business Strategy:** Companies can decide where to expand based on available workforce.
    """)

    st.subheader("Why This Study Matters")
    st.markdown("""
    By using up-to-date industrial HR data:
    - Policymakers can **target interventions** where they are most needed.
    - Businesses can **strategically expand operations** in states with skilled workforce.
    - Researchers can **analyze trends** in employment and workforce distribution over time.
    """)

# -------------------------------
# 2. Data Visualization Section
# -------------------------------
elif section == "Data Visualization":
    st.title("📊 Industry Clustering Dashboard")
    st.markdown("Interactive maps, charts, and tables of Indian industries, clusters, and states.")

    @st.cache_data
    def load_data():
        return pd.read_csv(r"C:\sachin\Python\Human Resource Management\industry_clusters.csv")

    data = load_data()

    # Rename clusters
    cluster_names = {0: "Retail", 1: "Poultry", 2: "Agriculture", 3: "Manufacturing", 4: "Services", 5: "Others"}
    data["Cluster_Name"] = data["Cluster"].map(cluster_names)

    # Sidebar questions
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

    # Dataset preview
    with st.expander("📋 Dataset Preview"):
        st.dataframe(data.head(20))

    # Helper function for maps
    def plot_india_state_map_safe(cluster_name):
        df = data[data["Cluster_Name"] == cluster_name].groupby("India/States").size().reset_index(name="Count")
        try:
            fig = px.choropleth(df, locations="India/States", locationmode="country names",
                                color="Count", hover_name="India/States",
                                title=f"{cluster_name} Businesses by State",
                                color_continuous_scale="Viridis")
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Map could not be rendered. Showing table instead.")
            st.table(df)
        st.dataframe(df)

    # Main visualization logic (simplified)
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

# -------------------------------
# 3. Total Worker Regression Prediction
# -------------------------------
elif section == "Total Worker Prediction":
    st.title("👷 Total Worker Prediction")
    MODEL_PATH = "total_worker_model.pkl"
    SCALER_PATH = "total_worker_scaler.pkl"
    STATES_CLASSES_PATH = "india_states_classes.npy"

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        states_classes = np.load(STATES_CLASSES_PATH, allow_pickle=True)
        st.success("Model, scaler, and states loaded successfully.")
    except:
        st.error("Model, scaler, or state classes not found. Please train or place files in folder.")
        st.stop()

    st.subheader("Enter feature values")
    state_code = st.number_input("State Code", value=1)
    district_code = st.number_input("District Code", value=1)
    selected_state = st.selectbox("India/States", options=states_classes)
    selected_state_encoded = int(np.where(states_classes == selected_state)[0][0])
    division = st.number_input("Division", value=1)
    group = st.number_input("Group", value=1)
    class_col = st.number_input("Class", value=1)
    marginal_total = st.number_input("Marginal Workers - Total - Persons", value=0)

    if st.button("🔮 Predict Total Workers"):
        features = np.array([[state_code, district_code, selected_state_encoded,
                              division, group, class_col, marginal_total]])
        try:
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]
            st.success(f"✅ Predicted Main Workers - Total - Persons: {int(round(prediction))}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# -------------------------------
# 4. Industry Classification Prediction
# -------------------------------
elif section == "Industry Classification Prediction":
    st.title("👨‍💼 Human Resource Classification App")

    @st.cache_data
    def load_model():
        model = joblib.load("naive_bayes_model.pkl")
        scaler = joblib.load("scaler.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, scaler, label_encoder

    model, scaler, label_encoder = load_model()

    @st.cache_data
    def load_data(path):
        data = pd.read_csv(path)
        data.columns = [col.strip().replace("`", "") for col in data.columns]
        return data

    data = load_data(r"C:\sachin\Python\Human Resource Management\output.csv")
    numeric_cols = ["Division", "Group", "Class",
                    "Marginal Workers - Total - Persons",
                    "Marginal Workers - Total - Males",
                    "Marginal Workers - Total - Females"]
    numeric_cols = [col for col in numeric_cols if col in data.columns]
    for col in numeric_cols:
        data[col] = data[col].astype(str).str.replace("`", "").str.extract("(\d+)")[0].astype(int)

    if data["India/States"].dtype == object:
        le_states = LabelEncoder()
        data["India/States_encoded"] = le_states.fit_transform(data["India/States"])
        state_options = data["India/States"].unique()
    else:
        le_states = None
        data["India/States_encoded"] = data["India/States"]
        state_options = data["India/States"].unique()

    def number_input_feature(column_name):
        if column_name in data.columns:
            min_val = int(data[column_name].min())
            max_val = int(data[column_name].max())
            default_val = min_val
            return st.number_input(f"{column_name} (min {min_val} - max {max_val})",
                                   min_value=min_val, max_value=max_val, value=default_val, step=1)
        else:
            return 0

    with st.form(key="input_form"):
        division = number_input_feature("Division")
        group = number_input_feature("Group")
        class_col = number_input_feature("Class")
        state_name = st.selectbox("India/States", options=state_options)
        india_states_encoded = le_states.transform([state_name])[0] if le_states else state_name
        marginal_total = number_input_feature("Marginal Workers - Total - Persons")
        marginal_males = number_input_feature("Marginal Workers - Total - Males")
        marginal_females = number_input_feature("Marginal Workers - Total - Females")
        submit_button = st.form_submit_button(label="🔍 Predict Industry")

    if submit_button:
        features = np.array([[division, group, class_col, india_states_encoded,
                              marginal_total, marginal_males, marginal_females]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        st.success(f"✅ Predicted NIC Name (Industry): **{predicted_label}**")

# -------------------------------
# 5. Applications & Use Cases Section
# -------------------------------
elif section == "Applications & Use Cases":
    st.title("💡 Applications & Use Cases")
    st.markdown("""
    **Industrial HR Analysis is used for:**  
    1. **Government Planning:** Allocate employment schemes and labor welfare policies effectively.  
    2. **Corporate Strategy:** Decide expansion locations based on workforce availability.  
    3. **Research & Academia:** Study employment patterns, workforce trends, and regional disparities.  
    4. **Skill Development Programs:** Identify areas requiring training programs for workforce upskilling.  
    5. **Economic Forecasting:** Predict workforce growth and sectoral employment trends.
    """)
