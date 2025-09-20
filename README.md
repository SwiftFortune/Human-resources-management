# 📊 Industrial Human Resource Geo-Visualization

## 🚀 Project Overview
This project focuses on analyzing and visualizing the **industrial workforce distribution in India** using modern **EDA, NLP, and Machine Learning techniques**.  
Traditionally, workers have been classified into **main and marginal categories** (excluding agriculture). However, the existing datasets are outdated.  
This study updates the classification through **clustering, classification, and regression models** to provide meaningful insights for **policy-making and resource management**.

---

## 🔧 Technologies Used
- **Exploratory Data Analysis (EDA)**
- **Visualization**
- **Natural Language Processing (NLP)**
- **Machine Learning (Classification & Regression)**

---

## 🌍 Domain
**Resource Management / Human Resource Analytics**

---

## 📝 Problem Statement
- Current industrial classification of the workforce is **not updated**.  
- Difficult to capture the **real employment trends**.  
- There is a **need for accurate, up-to-date classification** by sector, division, and class.  
- **Goal**: Build an **NLP + ML pipeline** to improve workforce classification and visualization.

---

## 📂 Dataset & Preprocessing
- **Source**: Census/industrial workforce CSV files.  
- **Preprocessing Steps**:
  - Removed special characters and converted codes to integers.  
  - Encoded categorical variables (States, NIC Name).  
  - Handled missing values.  
  - Performed feature engineering (ratios: rural/urban, marginal/total).  

---

## 📊 Exploratory Data Analysis (EDA)
- Industry distribution across India.  
- **Correlation heatmaps** of workforce attributes.  
- **Outlier detection** using boxplots.  
- **Word clouds** and frequency plots for industry names.  

---

## 🤖 NLP & Clustering
- Applied **TF-IDF Vectorization** on industry descriptions (`NIC Name`).  
- Performed **KMeans Clustering (k=6)** to group industries.  
- Identified **top terms per cluster** revealing sectoral patterns.  
- Exported **cluster-labeled dataset** for visualization.  

---

## 🧠 Classification Models
Trained multiple models to classify industries:  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Naive Bayes  
- AdaBoost, ExtraTrees, XGBoost  

✅ **Best Accuracy:** *Random Forest / XGBoost*

---

## 📈 Regression Models
Target Variable: **Main Workers – Total Persons**  

Models Tested:  
- Linear Regression, Ridge, Lasso, ElasticNet  
- Decision Tree, Random Forest, Gradient Boosting, XGBoost  
- KNN Regressor  

✅ **Best Fit:** *ElasticNet* (saved as final model)

---

## 📌 Results & Insights
- **Clustering**: Segmented industries into 6 logical groups.  
- **Classification**: Achieved strong accuracy in predicting sector labels.  
- **Regression**: ElasticNet provided balanced performance for workforce prediction.  
- **Visualization**: Heatmaps and cluster terms highlighted sectoral trends.  

---

## ✅ Conclusion
- Successfully built an **Industrial HR Geo-Visualization pipeline**.  
- Integrated **EDA, NLP clustering, Classification, and Regression**.  
- Provided updated workforce classification useful for:  
  - Policy-making  
  - Employment planning  
  - Industrial analysis  

**Future Scope**:
- Deploy as a **Streamlit Dashboard** for interactive geo-visualization.  
- Extend to **state-wise workforce mapping**.  

---

## 🙏 Acknowledgment
Thank you for exploring this project! Contributions and feedback are always welcome.
