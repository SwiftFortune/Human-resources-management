# streamlit_hr_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Load Data and Models
# ----------------------------
df = pd.read_csv("processed_hr_data.csv")

# Load saved models
rf_clf = joblib.load("rf_industry_classifier.pkl")
xgb_reg = joblib.load("xgb_worker_regressor.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# ----------------------------
# Reverse Mappings for Names
# ----------------------------
state_mapping = df[['State Code', 'India/States']].drop_duplicates().set_index('State Code')['India/States'].to_dict()
district_mapping = df[['District Code', 'India/States', 'District Code']].drop_duplicates()  # dummy if needed

# Features used in models
features = [
    "State Code", "District Code", "Division", "Group", "Class",
    "Main Workers - Rural - Persons", "Main Workers - Urban - Persons",
    "Male_Female_Ratio", "Rural_Urban_Ratio"
]

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="HR Dashboard", layout="wide", page_icon="💼")
st.title("Human Resource Management Dashboard 💼")

menu = ["Introduction", "Data Analysis", "Industry Prediction", "Worker Prediction", "About Me"]
choice = st.sidebar.selectbox("Menu", menu)

# ----------------------------
# 1️⃣ Introduction
# ----------------------------
if choice == "Introduction":
    st.header("Introduction to Human Resource Management")
    try:
        st.image("hr_intro.jpg", use_column_width=True)
    except:
        st.markdown("🖼️ *HR Introduction Image Missing*")
    st.write("""
    **Why Human Resource Management (HRM) is Important:**
    - Ensures optimal utilization of workforce.
    - Helps in employee satisfaction and retention.
    - Aligns human capital with business goals.

    **Problems HRM Solves:**
    - Labor allocation inefficiencies.
    - Skill mismatch in industries.
    - Monitoring workforce distribution (Rural vs Urban, Male vs Female).

    **Applications:**
    - Planning workforce distribution across industries.
    - Predicting manpower requirements.
    - Data-driven HR policies.

    **Solutions from Our Analysis:**
    - Identifying high/low workforce industries.
    - Detecting gender and rural-urban imbalances.
    - Predicting industry classification and workforce numbers.
    """)

# ----------------------------
# 2️⃣ Data Analysis
# ----------------------------
elif choice == "Data Analysis":
    st.header("Exploratory Data Analysis & Insights")

    st.subheader("Top 15 Industries by Count")
    top_industries = df["NIC Name"].value_counts().head(15).reset_index()
    top_industries.columns = ["Industry", "Count"]
    fig1 = px.bar(top_industries, x="Industry", y="Count", color="Count", title="Top 15 Industries by Count")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Workers Distribution: Rural vs Urban")
    rural_urban = df[["Main Workers - Rural - Persons", "Main Workers - Urban - Persons"]].sum().reset_index()
    rural_urban.columns = ["Category", "Count"]
    fig2 = px.pie(rural_urban, names="Category", values="Count", title="Rural vs Urban Workers")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Workers Distribution: Male vs Female")
    male_female = df[["Main Workers - Total - Males", "Main Workers - Total - Females"]].sum().reset_index()
    male_female.columns = ["Category", "Count"]
    fig3 = px.pie(male_female, names="Category", values="Count", title="Male vs Female Workers",
                  color_discrete_sequence=["#1f77b4","#ff7f0e"])
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Rural/Urban Ratio Distribution")
    fig4 = px.histogram(df, x="Rural_Urban_Ratio", nbins=50, title="Rural to Urban Worker Ratio Distribution")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Top 15 Industries by Male/Female Ratio")
    male_female_ratio = df.groupby("NIC Name")[["Male_Female_Ratio"]].mean().sort_values("Male_Female_Ratio", ascending=False).head(15).reset_index()
    fig5 = px.bar(male_female_ratio, x="NIC Name", y="Male_Female_Ratio", color="Male_Female_Ratio",
                  title="Top 15 Industries by Male/Female Ratio")
    st.plotly_chart(fig5, use_container_width=True)

    # Additional 10+ insights
    st.subheader("Additional Insights")
    st.markdown("**1. Industry with highest total workers**")
    st.write(df.groupby("NIC Name")["Main Workers - Total - Persons"].sum().idxmax())

    st.markdown("**2. Industry with lowest total workers**")
    st.write(df.groupby("NIC Name")["Main Workers - Total - Persons"].sum().idxmin())

    st.markdown("**3. Top 5 states by total workforce**")
    st.dataframe(df.groupby("India/States")["Main Workers - Total - Persons"].sum().sort_values(ascending=False).head(5))

    st.markdown("**4. Distribution of workers by Industry Group**")
    group_counts = df["Industry_Group"].value_counts().reset_index()
    group_counts.columns = ["Industry Group", "Count"]
    fig6 = px.bar(group_counts, x="Industry Group", y="Count", color="Count")
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("**5. Top 5 districts by workforce**")
    top_districts = df.groupby("District Code")["Main Workers - Total - Persons"].sum().sort_values(ascending=False).head(5)
    st.bar_chart(top_districts)

    st.markdown("**6. State-wise Rural vs Urban Distribution**")
    state_rural_urban = df.groupby("India/States")[["Main Workers - Rural - Persons","Main Workers - Urban - Persons"]].sum().reset_index()
    fig7 = px.bar(state_rural_urban, x="India/States", y=["Main Workers - Rural - Persons","Main Workers - Urban - Persons"])
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("**7. Industry-wise Male vs Female Distribution**")
    industry_gender = df.groupby("NIC Name")[["Main Workers - Total - Males","Main Workers - Total - Females"]].sum().reset_index()
    fig8 = px.bar(industry_gender, x="NIC Name", y=["Main Workers - Total - Males","Main Workers - Total - Females"])
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("**8. Industry-wise Rural vs Urban Ratio (Top 15)**")
    ratio_top = df.groupby("NIC Name")[["Rural_Urban_Ratio"]].mean().sort_values("Rural_Urban_Ratio", ascending=False).head(15).reset_index()
    fig9 = px.bar(ratio_top, x="NIC Name", y="Rural_Urban_Ratio", color="Rural_Urban_Ratio")
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("**9. Industries with Male/Female ratio > 1.2**")
    st.dataframe(df[df["Male_Female_Ratio"] > 1.2][["NIC Name","Male_Female_Ratio"]])

    st.markdown("**10. Industries with Female-dominated workforce (Male/Female ratio < 0.8)**")
    st.dataframe(df[df["Male_Female_Ratio"] < 0.8][["NIC Name","Male_Female_Ratio"]])

# ----------------------------
# 3️⃣ Classification Prediction
# ----------------------------
elif choice == "Industry Prediction":
    st.header("Predict Industry Group for a Worker Entry")
    st.write("Enter details to predict Industry Group:")

    # Dropdowns using names instead of codes
    selected_state = st.selectbox("Select State", options=sorted(df["India/States"].unique()))
    state_code = df[df["India/States"]==selected_state]["State Code"].iloc[0]

    selected_district = st.selectbox("Select District", options=sorted(df[df["India/States"]==selected_state]["District Code"].unique()))
    district_code = selected_district

    division = st.number_input("Division", min_value=int(df["Division"].min()), max_value=int(df["Division"].max()), value=1)
    group = st.number_input("Group", min_value=int(df["Group"].min()), max_value=int(df["Group"].max()), value=1)
    class_ = st.number_input("Class", min_value=int(df["Class"].min()), max_value=int(df["Class"].max()), value=1)
    rural_persons = st.number_input("Main Workers - Rural - Persons", min_value=0, value=100)
    urban_persons = st.number_input("Main Workers - Urban - Persons", min_value=0, value=50)
    male_female_ratio = st.number_input("Male/Female Ratio", min_value=0.0, value=1.0)
    rural_urban_ratio = st.number_input("Rural/Urban Ratio", min_value=0.0, value=1.0)

    user_input = np.array([[state_code,district_code,division,group,class_,rural_persons,urban_persons,male_female_ratio,rural_urban_ratio]])
    
    try:
        prediction = rf_clf.predict(user_input)
        predicted_industry = le.inverse_transform(prediction)[0]
        st.success(f"Predicted Industry Group: {predicted_industry}")
    except:
        st.warning("Prediction not possible for this combination (unseen during training)")

# ----------------------------
# 4️⃣ Regression Prediction
# ----------------------------
elif choice == "Worker Prediction":
    st.header("Predict Number of Workers for an Industry Entry")
    st.write("Enter details to predict total workers:")

    selected_state = st.selectbox("Select State", options=sorted(df["India/States"].unique()))
    state_code = df[df["India/States"]==selected_state]["State Code"].iloc[0]

    selected_district = st.selectbox("Select District", options=sorted(df[df["India/States"]==selected_state]["District Code"].unique()))
    district_code = selected_district

    division = st.number_input("Division", min_value=int(df["Division"].min()), max_value=int(df["Division"].max()), value=1)
    group = st.number_input("Group", min_value=int(df["Group"].min()), max_value=int(df["Group"].max()), value=1)
    class_ = st.number_input("Class", min_value=int(df["Class"].min()), max_value=int(df["Class"].max()), value=1)
    rural_persons = st.number_input("Main Workers - Rural - Persons", min_value=0, value=100)
    urban_persons = st.number_input("Main Workers - Urban - Persons", min_value=0, value=50)
    male_female_ratio = st.number_input("Male/Female Ratio", min_value=0.0, value=1.0)
    rural_urban_ratio = st.number_input("Rural/Urban Ratio", min_value=0.0, value=1.0)

    user_input = pd.DataFrame([[state_code,district_code,division,group,class_,rural_persons,urban_persons,male_female_ratio,rural_urban_ratio]],
                              columns=features)
    prediction = xgb_reg.predict(user_input)[0]
    st.success(f"Predicted Total Workers: {int(prediction)}")

# ----------------------------
# 5️⃣ About Me
# ----------------------------
elif choice == "About Me":
    st.header("About Me")
    try:
        st.image("profile_image.jpg", width=200)
    except:
        st.markdown("👤 *Profile Image Missing*")
    st.write("""
    **Name:** Sachin Hembram  
    **Role:** Data Scientist / ML Engineer  
    **Expertise:** Machine Learning, Deep Learning, NLP, Data Analysis, Dashboard Development  
    **LinkedIn:** [linkedin.com/in/sachinhembram](https://www.linkedin.com/in/sachinhembram)  
    **GitHub:** [github.com/sachinhembram](https://github.com/sachinhembram)  
    """)
    st.write("This dashboard demonstrates HR analytics, predictive modeling, and interactive visualizations using Python, Streamlit, and Plotly.")
