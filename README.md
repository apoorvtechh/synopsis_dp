# 📘 NYC Taxi Demand Forecasting — Project Synopsis

This repository hosts the **interactive Streamlit synopsis** for the NYC Taxi Demand Forecasting project.  
It visually explains the entire workflow — from data cleaning → region clustering → feature engineering → modeling → hybrid forecasting → evaluation.

---

## 🚀 Live Synopsis App  
Explore the full interactive explanation:

🔗 **https://apoorvtechh-synopsis-dp-app-wxbcw6.streamlit.app/**

## 🚀 Live  App  
Explore the working:

🔗 **https://apoorvtechh-dashboard-demand-prediction-app-gx2szx.streamlit.app/**

---

## 📚 What This Synopsis Covers

### 1️⃣ Introduction  
- Why predicting demand is important for ride-hailing systems  
- How forecasting improves:
  - 🚕 Driver route planning  
  - ⚡ Platform surge pricing  
  - 🗺️ City-level traffic management  
- What the project aims to solve: **predict NYC taxi demand every 15 minutes, for 30 regions**

---

### 2️⃣ Dataset Summary  
- **NYC Yellow Taxi Trip Records (Jan–Mar 2016)**  
- Over **33 million raw rows**  
- Cleaned using:
  - Latitude/longitude bounding box  
  - Trip distance limits  
  - Fare amount range  
- Final shape after cleaning: **~26 million usable records**

---

### 3️⃣ Region Clustering  
- Used **MiniBatch KMeans (K=30)**  
- Coordinates scaled using StandardScaler  
- Haversine distance used to pick the best K  
- Output: **30 compact, meaningful taxi demand zones across NYC**

---

### 4️⃣ Time-Series Construction  
For every region:
- Demand aggregated at **15-minute resolution**  
- Added:
  - Lag features  
  - Rolling mean & rolling std  
  - Day-of-week, month  
  - EWMA smoothing (region-wise α optimization)

Final structured dataset:  
📊 **262,080 rows × engineered time-series features**

---

### 5️⃣ Baseline Models  
#### 🔮 Prophet  
Captures:
- Long-term trend  
- Daily & weekly seasonality  

#### ⚡ XGBoost  
Learns:
- Short-term patterns  
- Non-linear relationships  
- Sudden spikes  

Baseline XGBoost MAPE ≈ **0.061**  

---

### 6️⃣ Hybrid Modeling (Prophet + XGBoost)  
- Prophet’s hourly **trend** is merged into 15-min data as a new feature  
- XGBoost then learns residual patterns  
- Region-wise custom model trained for each of the 30 regions  

Result:  
🎯 **MAPE improves from 0.037 → 0.0301**

---

### 7️⃣ Region-wise Evaluation  
- Each region evaluated individually  
- Visualization included in the app  
- Heatmaps, trend plots, and spike behavior highlighted visually  

---

### 8️⃣ Full Production Pipeline  
The production implementation lives in the main forecasting repo:

🔗 **https://github.com/apoorvtechh/demand_forecasting**

It includes:
- DVC data pipeline  
- MLflow experiment tracking  
- Model registration  
- Region-wise trained hybrid models  
- Deployment-ready structure  

---

### 9️⃣ Experimentation Code  
All experiments, research notebooks, and clustering scripts:

🔗 **https://github.com/apoorvtechh/demand_preidction_experimentation**

---

## 🧭 Purpose of This Synopsis  
This app exists to **explain the entire project visually**, including:
- Why each method is chosen  
- How features were engineered  
- What improvements hybrid modeling brings  
- How region-wise behavior affects forecasting  

It supports interviews, project documentation, and portfolio presentation.

---

 
