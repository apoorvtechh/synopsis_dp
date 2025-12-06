import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------
# APP CONFIG
# ---------------------------------------
st.set_page_config(page_title="Demand Prediction Dashboard", layout="wide")


# ---------------------------------------
# RADIO BUTTON MENU
# ---------------------------------------
menu = st.sidebar.radio(
    "📌 Navigate Sections",
    [
        "1️⃣ Project Overview",
        "2️⃣ Dataset Description",
        "3️⃣ Exploratory Data Analysis",
        "4️⃣ Data Preprocessing ",
        "5️⃣ Baseline Model",
        "6️⃣ Hyperparameter Tuning",
        "7️⃣ DVC Pipeline",
        "8️⃣ Conclusion"
    ]
)

# ---------------------------------------
# 1️⃣ PROJECT OVERVIEW
# ---------------------------------------
if menu == "1️⃣ Project Overview":
    st.header("📊 Project Overview")
    st.write("""
    This project predicts hourly ride demand using a **Hybrid Forecasting Model**
    combining **Prophet** and **XGBoost**.

    ### Key Features
    - Region-wise demand prediction  
    - Rolling window optimization  
    - Long-term & short-term pattern capture  
    - End-to-end Streamlit dashboard  
    - Deployed with AWS Auto Scaling  
    """)

# ---------------------------------------
# 2️⃣ DATASET DESCRIPTION
# ---------------------------------------
elif menu == "2️⃣ Dataset Description":
    st.header("🗂️ Dataset Description")

    st.write(f"""
    ### 📌 Total Rows Across All Months  
    - **Million:** ~{3.3} M  
    - **Crore:** ~{3.3} Cr  

    Together, the three monthly files contain more than **33 million (3.3crore) taxi trips**.

    The NYC Yellow Taxi Trip Records dataset for **2016 (Jan–Mar)** is extremely large, containing millions of individual ride entries across New York City.  
    To help visualize the structure and schema inside this dashboard, below is a **sample DataFrame** representing the actual fields from the original dataset.
    """)


    # ---------------------------
    # SAMPLE DATAFRAME (STATIC)
    # ---------------------------
    import pandas as pd

    sample_data = {
        "VendorID": [2, 2, 2, 2, 2, 2, 1, 1, 2],
        "tpep_pickup_datetime": [
            "2016-01-01 00:00:00",
            "2016-01-01 00:00:00",
            "2016-01-01 00:00:00",
            "2016-01-01 00:00:00",
            "2016-01-01 00:00:00",
            "2016-01-01 00:00:00",
            "2016-01-01 00:00:01",
            "2016-01-01 00:00:02",
            "2016-01-01 00:00:02",
        ],
        "tpep_dropoff_datetime": [
            "2016-01-01 00:00:00",
            "2016-01-01 00:00:00",
            "2016-01-01 00:00:00",
            "2016-01-01 00:00:00",
            "2016-01-01 00:00:00",
            "2016-01-01 00:18:30",
            "2016-01-01 00:11:55",
            "2016-01-01 00:11:14",
            "2016-01-01 00:11:08",
        ],
        "passenger_count": [2.0, 5.0, 1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0],
        "trip_distance": [1.1, 4.9, 0.5, 4.7, 1.7, 5.5, 1.2, 6.0, 3.2],
        "pickup_longitude": [
            -73.990372, -73.980782, -73.984550, -73.993469, -73.960625,
            -73.980118, -73.979424, -73.947151, -73.998344
        ],
        "pickup_latitude": [
            40.734695, 40.729912, 40.679565, 40.718990, 40.781330,
            40.743050, 40.744614, 40.791046, 40.723896
        ],
        "RatecodeID": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "store_and_fwd_flag": ["N", "N", "N", "N", "N", "N", "N", "N", "N"],
        "dropoff_longitude": [
            -73.981842, -73.944473, -73.950272, -73.962242, -73.977264,
            -73.913490, -73.992035, -73.920769, -73.995850
        ],
        "dropoff_latitude": [
            40.732407, 40.716679, 40.788925, 40.657333, 40.758514,
            40.763142, 40.753944, 40.865578, 40.688400
        ],
        "fare_amount": [7.5, 18.0, 33.0, 16.5, 8.0, 19.0, 9.0, 18.0, 11.5]
    }

    df_example = pd.DataFrame(sample_data)

    # ---------------------------
    # DISPLAY SAMPLE DATAFRAME
    # ---------------------------
    st.subheader("📘 Sample Taxi Trip DataFrame (Preview)")
    st.dataframe(df_example)


    # ---------------------------
    # COLUMN INFORMATION
    # ---------------------------
    st.subheader("📑 Column Information")
    st.write(pd.DataFrame({
        "Column Name": df_example.columns,
        "Data Type": df_example.dtypes.astype(str)
    }))

    # ---------------------------
    # DATA IS IN 3 MONTHLY PARTS
    # ---------------------------
    st.subheader("🗃️ Dataset Files (Three Parts)")
    jan_m = 14       # 14 million
    feb_m = 12       # 12 million
    mar_m = 12       # 12 million

    # Convert to crores (1 crore = 10 million)
    jan_cr = jan_m / 10
    feb_cr = feb_m / 10
    mar_cr = mar_m / 10

    st.write(f"""
    The original dataset is divided into **three monthly CSV files**:

    | Month | File Name | 
    |-------|-----------|
    | **January 2016** | `yellow_tripdata_2016-01.csv` | 
    | **February 2016** | `yellow_tripdata_2016-02.csv` | 
    | **March 2016** | `yellow_tripdata_2016-03.csv` | 

    """)



# ---------------------------------------
# 3️⃣ EDA
# ---------------------------------------
elif menu == "3️⃣ Exploratory Data Analysis":
    st.header("🔎 Exploratory Data Analysis (EDA)")

    st.write("""
    The goal of EDA was to evaluate the **quality, validity, and reliability** of the NYC Taxi dataset.
    This step ensures that the downstream forecasting model is trained only on **trustworthy, realistic, and clean data**.
    Below are the major checks performed and the reasons behind each decision.
    """)

    # ================================
    # COLUMN DEFINITIONS
    # ================================
    st.subheader("📘 Key Column Definitions")
    st.write("""
    Understanding column meanings is essential for spotting invalid entries and outliers.

    | Column | Meaning |
    |--------|---------|
    | **tpep_pickup_datetime** | Timestamp when the passenger was picked up |
    | **tpep_dropoff_datetime** | Timestamp when the passenger was dropped off |
    | **passenger_count** | Number of passengers in the vehicle (valid range: 1–5) |
    | **trip_distance** | Distance traveled in miles, recorded by the taximeter |
    | **pickup_latitude / pickup_longitude** | GPS coordinates of pickup location |
    | **dropoff_latitude / dropoff_longitude** | GPS coordinates of dropoff location |
    | **fare_amount** | Base fare charged for the trip (excluding taxes, tips, surcharges) |
    | **RatecodeID** | Category of fare: standard, JFK, Newark, negotiated fare, etc. |
    | **store_and_fwd_flag** | Whether the trip record was stored and forwarded later (Y/N) |
    """)

    st.info("These definitions were used to identify invalid or impossible values during cleaning.")

    # ================================
    # 1. Passenger Count
    # ================================
    st.subheader("1️⃣ Passenger Count Validation")
    st.write("""
    - NYC Taxi Commission regulations permit **1 to 5 passengers** per ride.  
    - Any values outside this range represent **invalid or corrupted records**.  
    - Such rows were removed because they cannot occur in real-world scenarios.
    """)
    st.info("Ensures the dataset reflects realistic passenger behavior.")
    st.markdown("---")
    # ================================
    # 2. Trip Distance Outliers
    # ================================
    st.subheader("2️⃣ Trip Distance Outliers")

    st.write("""
    The trip distance column showed some **extremely unrealistic values**.

    ### 🚨 Detected Outlier Example
    - **Maximum observed distance:** ~19,072,628 miles  
    - For reference, the **Earth’s circumference** is only 24,901 miles.  
    - This is clearly due to **GPS malfunction or corrupted input**.

    ### 🎯 Why Removed?
    - Unrealistic numbers distort averages and statistical distributions.  
    - They cause models to learn incorrect patterns and reduce accuracy.

    ### ✔ Final Cleaning Rule
    - Only values below the **99th percentile (~24 miles)** were kept.  
    - All extreme distances were removed as invalid data.
    """)
    st.info("Removes GPS errors and ensures distance data remains meaningful.")
    st.markdown("---")
    # ================================
    # 3. Fare Amount Outliers
    # ================================
    st.subheader("3️⃣ Fare Amount Outliers")

    st.write("""
    Fare amounts also included extreme invalid entries.

    ### 🚨 Detected Outlier Example
    - **Maximum observed fare:** ~$429,496.72  
    - No taxi ride in NYC can cost anywhere close to this amount.  

    ### ❗ Why Removed?
    These values are caused by:
    - Meter malfunction  
    - Data corruption  
    - Overflow errors  

    Keeping such values would:
    - Skew statistical summaries  
    - Mislead fare trend analysis  
    - Confuse machine learning models  

    ### ✔ Final Cleaning Rule
    - Only fares between **$0.50 and the 99th percentile (~$81)** were kept.
    - All excessively large fares were removed.
    """)
    st.info("Keeps only realistic pricing data for modeling.")
    st.markdown("---")
    # ================================
    # 4. RatecodeID Cleaning
    # ================================
    st.subheader("4️⃣ RatecodeID Cleaning")
    st.write("""
    - Valid RatecodeID values range from **1 to 6** according to NYC taxi specifications.  
    - Any codes outside this range were removed or corrected.  
    """)
    st.info("Ensures all trips follow official NYC taxi categorization.")

    # ================================
    # 5. store_and_fwd_flag Missing Values
    # ================================
    st.subheader("5️⃣ Handling Missing 'store_and_fwd_flag'")
    st.write("""
    - Missing values were replaced with a placeholder category (`99`).  
    - This avoids dropping a large number of rows.
    """)
    st.info("Prevents unnecessary data loss from missing categorical values.")
    st.markdown("---")
    # ================================
    # 6. NYC Bounding Box Validation
    # ================================
    st.subheader("6️⃣ Latitude & Longitude Validation")
    st.write("""
    Coordinates were checked against official NYC geography.

    ### NYC Valid Boundary
    - Latitude: **40.60 → 40.85**  
    - Longitude: **-74.05 → -73.70**

    Any coordinate outside this range indicates:
    - GPS errors  
    - Trips recorded outside NYC  
    - Impossible travel locations  

    Such rows were removed.
    """)
    st.info("Removes spatial errors and ensures all trips occurred in NYC.")
    st.markdown("---")
    # ================================
    # 7. Final Outlier Removal Strategy
    # ================================
    st.subheader("7️⃣ Final Outlier Removal Strategy")

    st.write("""
    The dataset was cleaned using:
    - **Percentile-based filtering** (distance & fare)  
    - **Regulatory rules** (passenger count, RatecodeID)  
    - **Geographical boundaries** (NYC bounding box)

    This ensures:
    - Removal of noise and corrupted entries  
    - Stabilized feature distributions  
    - Improved model reliability  
    """)
    st.success("EDA completed — dataset is now clean and ready for forecasting.")



# ---------------------------------------
# 4️⃣ ROLLING WINDOW
# --------------------------------------
elif menu == "4️⃣ Data Preprocessing ":
    st.header("🧹 Data Preprocessing & Region Clustering")

    st.write("""
    The goal of data preprocessing is to convert **raw taxi pickup coordinates** into
    meaningful geographic **regions**, and then transform them into **clean time-series**
    suitable for forecasting.

    Since the dataset contains over **33 million rows**, every step is optimized for 
    large-scale, memory-efficient processing (Dask, chunking, incremental learning).
    """)

    st.markdown("---")

    # ================================================================
    # 1️⃣ Loading Pickup Coordinates with Dask
    # ================================================================
    st.subheader("1️⃣ Loading Pickup Coordinates from Parquet using Dask")

    st.write("""
    **Why this step?**  
    Loading 33M rows directly with Pandas would easily overflow memory.  
    **Dask** reads Parquet files lazily in partitions, making it ideal for big-data preprocessing.
    """)

    st.success("✅ Parquet file loaded efficiently with Dask!")

    st.code("""
pickup_coord_dataset = dd.read_parquet("pickup_coords_parquet.parquet")
rows = pickup_coord_dataset.shape[0].compute()
cols = len(pickup_coord_dataset.columns)
print(f"Shape: ({rows}, {cols})")
    """, language="python")

    st.write("### Sample Pickup Coordinates")
    st.table({
        "pickup_latitude": [40.734695, 40.729912, 40.679565, 40.718990, 40.781330],
        "pickup_longitude": [-73.990372, -73.980782, -73.984550, -73.993469, -73.960625]
    })

    st.info("Reason: Handle very large datasets without crashing system memory.")

    st.markdown("---")

    # ================================================================
    # 2️⃣ Sampling the Dataset for Visualization
    # ================================================================
    st.subheader("2️⃣ Sampling the Dataset for Visualization")

    st.write("""
    **Why this step?**  
    Plotting tens of millions of points in the browser is not feasible.  
    A **small random sample** preserves the overall spatial pattern at a fraction of the cost.
    """)

    st.code("""
sample_df = pickup_coord_dataset.sample(frac=0.1, random_state=42)
    """, language="python")

    st.image("image2.png", caption="NYC Pickup Density (Sample)")

    st.info("Reason: Use lightweight samples to understand city-wide pickup density.")

    st.markdown("---")

    # ================================================================
    # 3️⃣ Scaling Coordinates with StandardScaler
    # ================================================================
    st.subheader("3️⃣ Scaling Coordinates with StandardScaler")

    st.write("""
    **Why this step?**  
    Clustering algorithms like K-Means / MiniBatchKMeans are **distance-based**.  
    Since latitude and longitude are on different numeric scales, they must be
    standardized to avoid bias towards one axis.
    """)

    st.code("""
scaler = StandardScaler()
scaled_coords = scaler.fit_transform(sample_df)
scaled_coords.columns = ["pickup_longitude_scaled", "pickup_latitude_scaled"]
    """, language="python")

    st.success("✅ Coordinates normalized for clustering.")

    st.info("Reason: Ensures both latitude and longitude contribute fairly to distance.")

    st.markdown("---")

    # ================================================================
    # 4️⃣ Incremental StandardScaler + MiniBatchKMeans (Full Data)
    # ================================================================
    st.subheader("4️⃣ Incremental Scaling & MiniBatchKMeans on 33M+ Rows")

    st.write("""
    **Why incremental training?**  
    - You cannot call `.fit()` or `.fit_predict()` on all 33M rows at once.  
    - Instead, both **StandardScaler** and **MiniBatchKMeans** are trained **incrementally**
      over many Parquet files using `partial_fit`.
    """)


    st.info("Reason: This pattern allows scaling and clustering over all Parquet parts without loading everything into RAM at once.")

    st.markdown("---")

    # ================================================================
    # 5️⃣ Selecting the Optimal Number of Clusters (K)
    # ================================================================
    st.subheader("5️⃣ Selecting the Optimal Number of Regions (K)")

    st.write("""
    **Why this step?**  
    The number of clusters (regions) directly controls how coarse or fine the city is partitioned.

    To pick **K**, we:
    - Trained MiniBatchKMeans for multiple values of K (10, 20, ..., 90)  
    - Converted centroids back to real coordinates  
    - Used the **Haversine distance** to measure how far each region center is from its neighbors  
    - Computed the percentage of regions whose **average neighbor distance** lies in **1–1.5 miles**  
    """)

    st.code("""
percentage = ((avg_distances >= 1.0) & (avg_distances <= 1.5)).mean() * 100
    """, language="python")

    st.info("Reason: We want regions that are geographically compact and meaningful in the real world.")

    st.markdown("---")

    # ================================================================
    # 6️⃣ Final K = 30 Regions + Cluster Visualization
    # ================================================================
    st.subheader("6️⃣ Final Region Clustering (K = 30)")

    st.write("""
    Based on the distance analysis, **30 clusters** provided a good balance between:
    - Spatial compactness  
    - City-wide coverage  
    - Interpretability for demand forecasting
    """)

    st.code("""
centroids = scaler.inverse_transform(mini_batch.cluster_centers_)
centroids_df = pd.DataFrame(centroids, columns=["pickup_latitude","pickup_longitude"])
centroids_df.to_csv("final_cluster_centroids.csv", index=False)
    """, language="python")

    st.image("image.png", caption="NYC Region Clusters (K = 30) — Pickups + Centroids", use_container_width=False)

    st.success("🎉 Final region model trained — 30 pickup regions identified across NYC.")

    st.markdown("---")

    # ================================================================
    # 7️⃣ Assigning Region IDs to Every Trip
    # ================================================================
    st.subheader("7️⃣ Assigning a Region (Cluster ID) to Each Trip")

    st.write("""
    **Why this step?**  
    Every trip must be mapped to one of the 30 regions.  
    This **region ID** becomes a key feature for downstream time-series modeling.
    """)



    st.success("✅ Region IDs assigned to every trip record.")

    st.markdown("---")

    # ================================================================
    # 8️⃣ Time-Series Preparation (Region-Level)
    # ================================================================
    st.subheader("8️⃣ Time-Series Preparation (Region-Level Pickup Counts)")

    st.write("""
    **Why this step?**  
    Forecasting models like Prophet / XGBoost need **regular time-series**, not raw trips.

    We:
    - Converted pickup timestamps to datetime  
    - Grouped by **region + 15-minute intervals**  
    - Counted trips to form `total_pickups` per region per time bin
    """)

    st.code("""
df_final["tpep_pickup_datetime"] = pd.to_datetime(df_final["tpep_pickup_datetime"], errors="coerce")
df_final = df_final.dropna(subset=["tpep_pickup_datetime"])
df_final = df_final.set_index("tpep_pickup_datetime")

final_resampled = (
    df_final
    .groupby("region")
    .resample("15min")
    .size()
    .reset_index(name="total_pickups")
)
final_resampled.rename(columns={"pickup_count": "total_pickups"}, inplace=True)
    """, language="python")

    st.success("✅ 15-minute region-level time-series dataset created.")

    st.markdown("---")

    # ================================================================
    # 9️⃣ Rolling Window Optimization (SMA Baseline)
    # ================================================================
    st.subheader("9️⃣ Rolling Window Optimization (Simple Moving Average)")

    st.write("""
    As a **baseline smoother**, we applied a Simple Moving Average (SMA) with window sizes **3 to 10** 
    and evaluated performance using **MAPE** for each region.

    Key finding:
    - For the **top-performing regions**, the best window was consistently **3 time steps**.
    """)

    st.code("""
# Example result (top 10 regions)
region  best_window  best_mape
18      3           0.1015
26      3           0.1042
8       3           0.1055
6       3           0.1056
3       3           0.1075
24      3           0.1090
13      3           0.1096
29      3           0.1158
10      3           0.1165
11      3           0.1165
    """, language="python")

    st.info("Interpretation: A small SMA window (3 steps) is enough to remove noise without oversmoothing the signal.")

    st.markdown("---")

    # ================================================================
    # 🔟 EWMA Optimization (Final Choice)
    # ================================================================
    st.subheader("🔟 EWMA Smoothing (Final Selected Smoother)")

    st.write("""
    To get a more **responsive smoothing method**, we also applied **Exponentially Weighted Moving Average (EWMA)** 
    and tuned the smoothing factor **α** in the range **0.05 → 0.55**.

    For each region, we selected the α that minimized MAPE.
    """)

    st.code("""
# Best EWMA α per region (top 10)
region  best_alpha  best_mape
18      0.55        0.0730
6       0.55        0.0767
8       0.55        0.0770
3       0.55        0.0791
26      0.55        0.0794
13      0.55        0.0809
24      0.55        0.0815
29      0.55        0.0834
10      0.55        0.0845
11      0.55        0.0854
    """, language="python")

    st.write("""
    **Key insight:**  
    - For the **best-performing regions**, α = **0.55** consistently gave the lowest MAPE  
    - EWMA with α = 0.55 performed **better than SMA (window = 3)** in terms of MAPE  
    """)

    st.image("image3.png", caption="EWMA Smoothing vs Actual (Example Region)", use_container_width=False)

    st.success("✅ Final decision: **EWMA (α = 0.55)** selected as the primary smoothing method for the time-series before feeding into the hybrid model.")



# 5️⃣ PROPHET MODEL
# ---------------------------------------
# 5️⃣ BASELINE MODELING
# -------------------------------------------------------
elif menu == "5️⃣ Baseline Model":

    st.header("📈 Baseline Modeling — From Features to Prophet & XGBoost")

    st.write("""
    Once the raw NYC taxi trips were converted into **15-minute region-level time series**, 
    the next step was to build **strong baseline models** that could:

    - Understand temporal patterns (daily/weekly cycles)
    - React to short-term demand spikes
    - Scale across **30 different regions**
    - Provide a robust benchmark before designing a hybrid solution

    This section walks step-by-step through:
    1. **Feature Engineering** — how raw time-series was enriched  
    2. **XGBoost Baseline** — a strong machine learning baseline  
    3. **Region-wise modeling strategy** — why separate models per region  
    4. **Prophet Baseline** — classical trend + seasonality model  
    5. **Why Prophet alone is not enough**  
    6. **Why a Hybrid approach (Prophet + XGBoost) was needed**  
    7. **XGBoost vs Hybrid comparison**
    """)

    st.markdown("---")

    # ============================================================
    # 1️⃣ FEATURE ENGINEERING (CORE OF THE PIPELINE)
    # ============================================================
    st.subheader("1️⃣ Feature Engineering — Turning Time Into Predictive Signals")

    st.write("""
    After:
    - Assigning each trip to a **region (cluster)**  
    - Aggregating trips into **15-minute time bins**  

    the base time-series looked like:

    ```text
    tpep_pickup_datetime | region | total_pickups
    ------------------------------------------------
    2016-01-01 00:00     |   0    |   138
    2016-01-01 00:00     |   1    |   156
    ...
    2016-03-31 23:45     |  29    |   155
    ```

    This is good, but **not enough** for a machine learning model.
    ML models like XGBoost don’t “understand time” by themselves — they only see numbers.
    So, we built **time-aware features** on top of this.
    """)

    st.write("### ✅ Final Structured Dataset")
    st.markdown("""
    - **Shape after preprocessing:** `262,080 rows × 9+ columns`  
    - Each row represents: **one region** × **one 15-minute interval**
    """)

    st.write("### 🔧 Feature Set and Why Each Was Created")

    st.markdown("#### ⏳ Lag Features — Recent Demand Memory")

    st.write("""
    **Features:** `lag_1`, `lag_2`, `lag_3`, `lag_4`  
    - `lag_1` → demand 15 minutes ago  
    - `lag_2` → 30 minutes ago  
    - `lag_3` → 45 minutes ago  
    - `lag_4` → 60 minutes ago  

    **Why?**  
    Real-world demand is **highly autocorrelated**.  
    If pickups were high in the last hour, they are likely to stay high in the next few intervals.

    **Why not more lags (e.g., lag_24, lag_48)?**  
    - Too many lags increase dimensionality and risk of overfitting.  
    - Short lags (1–4) are enough to capture local momentum without exploding feature space.
    """)

    st.markdown("#### 📉 Rolling Mean — Local Trend Smoothing")

    st.write("""
    **Features:** `rolling_mean_3`, `rolling_mean_6`  

    - `rolling_mean_3` → average pickups over last **45 minutes**  
    - `rolling_mean_6` → average pickups over last **90 minutes**

    **Why?**
    - Raw 15-min demand is noisy (rain, events, randomness)  
    - Rolling mean gives XGBoost a **smoothed view** of demand
    - Helps the model understand short-term trend without manual decomposition

    **Why window sizes 3 and 6? Why not 12 or 24?**  
    - Very large windows over-smooth and hide real spikes.  
    - During experimentation, **window=3** was consistently the best for smoothing vs responsiveness.
    """)

    st.markdown("#### 📊 Rolling Standard Deviation — Local Volatility")

    st.write("""
    **Features:** `rolling_std_3`, `rolling_std_6`  

    These measure **how unstable** demand has been recently:

    - High std → noisy or peak periods  
    - Low std → stable demand  

    **Why?**  
    XGBoost can use this to:
    - Treat stable periods differently from volatile ones  
    - Avoid overreacting to a single spike when volatility is normally low
    """)

    st.markdown("#### 📆 Calendar & Region Features")

    st.write("""
    **Features:** `day_of_week`, `month`, `avg_pickups`, `region`  

    - `day_of_week` → captures weekday vs weekend patterns  
    - `month` → difference between January, February, March behavior  
    - `avg_pickups` → baseline demand level of that region  
    - `region` → numerical region ID (acts as a categorical indicator)

    **Why not one global model without region?**  
    - Different regions (airport vs Manhattan vs residential) have **very different** demand levels.  
    - Including `region` allows the model to **learn different behaviours per zone**, even if trained jointly.
    """)

    st.success("Feature engineering converts a simple time-series into a rich supervised learning dataset.")

    st.markdown("---")

    # ============================================================
    # 2️⃣ XGBOOST BASELINE MODEL (FEATURE-BASED)
    # ============================================================
    st.subheader("2️⃣ XGBoost Baseline — Learning Short-Term Demand Patterns")

    st.write("""
    With all these engineered features, **XGBoost** was used as the first strong baseline model.

    **Why XGBoost?**
    - Handles **non-linear relationships** very well  
    - Naturally supports **tabular feature sets** (lags, rolling stats, calendar)  
    - Efficient and fast to train even on large datasets  
    - Works well with **imbalanced and noisy** data  

    XGBoost’s job here:
    - Learn how recent history (`lag_*`) and short-term trend (`rolling_mean_*`) influence next 15-min demand  
    - Learn weekly patterns through `day_of_week`  
    - Learn how different regions behave via `region` and `avg_pickups`
    """)

    st.image("image4.png", caption="Effect of Lag & Rolling Features on XGBoost Predictions", use_container_width=False)

    st.write("""
    ### 📊 XGBoost Results

    - ✅ **XGBoost Baseline Overall MAPE:** `0.061`  


    """)

    st.info("""
    Interpretation:  
    - For most regions, XGBoost predicts with **2–4% MAPE**, which is very strong.  
    - A few regions (like Region 1) are highly volatile and harder to predict.
    """)

    st.markdown("---")

    # ============================================================
    # 3️⃣ REGION-WISE MODELING STRATEGY
    # ============================================================
    st.subheader("3️⃣ Region-Wise Modeling — One Model Per Region")

    st.write("""
    Instead of relying on **one global model for the entire NYC**,  
    we adopted a **region-wise modeling strategy**.

    ### Why?

    - Demand behaviour near **JFK/LaGuardia** is not the same as in **Midtown Manhattan**  
    - Residential regions may have strong morning and evening peaks  
    - Commercial regions show different weekday vs weekend activity

    So, for many experiments, models were trained **per region**, for example:
    - Separate XGBoost for Regions
    - Separate Prophet runs per region for analysis  
    """)

    st.write("""
    **Advantages:**
    - Better modeling of local behaviour  
    - Interpretability per region  
    - Ability to inspect **which regions are easy or hard to predict**

    **Why not only a single global model?**
    - Global models may underfit small but important regions  
    - Region-wise modeling ensures **fair attention** to each area
    """)

    st.info("Region-wise modeling gave deeper insight into city behaviour and error distribution.")

    st.markdown("---")

    # ============================================================
    # 4️⃣ PROPHET BASELINE — TREND & SEASONALITY MODEL
    # ============================================================
    st.subheader("4️⃣ Prophet Baseline — Understanding Trend and Seasonality")

    st.write("""
    Alongside XGBoost, a **classical forecasting model — Prophet** — was also used.

    **What Prophet is good at:**
    - Capturing **smooth long-term trend**  
    - Modeling **daily cycles** (office hours, nightlife, commuting)  
    - Modeling **weekly patterns** (weekdays vs weekends)  

    Prophet was run region-wise on:
    - **15-min resampled data** for fine granularity  
    - **1-hour resampled data** to get a cleaner trend signal
    """)

    st.write("""
    ### Why Prophet was included:

    - It provides **interpretable components**: trend, daily seasonality, weekly seasonality  
    - Helps understand: _“Is the issue noise or trend?”_  
    - Generates a **smooth trend estimate**, which later becomes useful for the **hybrid model**
    """)

    st.warning("""
    However, Prophet **struggles** with:
    - Sudden spikes (events, heavy rain, concert nights)  
    - Region-specific micro-patterns  
    - High-frequency 15-min fluctuations  

    Prophet baseline MAPE typically ranged from **0.093 to 0.12**,  
    which is **worse than XGBoost**.
    """)

    st.info("Conclusion: Prophet alone is not accurate enough, but it is very useful as a trend extractor.")

    st.markdown("---")

    # ============================================================
    # 5️⃣ WHY PROPHET ALONE WAS NOT SELECTED
    # ============================================================
    st.subheader("5️⃣ Why Prophet Alone Was Not Enough")

    st.write("""
    During experiments:
    - Prophet provided excellent **trend + seasonality plots**, but **higher MAPE** than XGBoost  
    - XGBoost captured **local dynamics**, but lacked **global trend awareness**

    So, Prophet was **not chosen** as the final standalone model because:
    - It could not react fast enough to short 15-min demand spikes  
    - It focused more on smooth cycles than sharp variations  
    - XGBoost was clearly stronger on pure accuracy
    """)

    st.info("Decision: Use Prophet for what it is best at — providing a clean trend signal — and combine it with XGBoost.")

    st.markdown("---")

    # ============================================================
    # 6️⃣ HYBRID IDEA — ADD PROPHET TREND INTO XGBOOST
    # ============================================================
    st.subheader("6️⃣ Hybrid Idea — XGBoost + Prophet Trend")

    st.write("""
    Based on all observations:

    - ✅ XGBoost is very good at **short-term, local, non-linear patterns**  
    - ✅ Prophet is very good at **smooth trend and seasonal structure**  
    - ❌ Neither alone perfectly balances **trend + short-term spikes**  

    👉 **Solution:** Use Prophet to generate an **hourly trend signal**,  
    then use that as an extra feature in XGBoost.

    This new feature is called: **`prophet_trend`**
    """)

    st.write("""
    ### How `prophet_trend` was created:

    1. For each region:
        - Resample to **1-hour** pickups  
        - Train Prophet on the hourly series  
        - Forecast trend values for the full train+test period  

    2. Map each 15-min timestamp to its corresponding **hourly Prophet prediction**  
    3. Merge that value into the 15-min dataset as a new column: `prophet_trend`  
    4. Retrain XGBoost with the feature set:
    ```text
    ['lag_1','lag_2','lag_3','lag_4',
     'region','avg_pickups','day_of_week',
     'rolling_mean_3','rolling_mean_6',
     'rolling_std_3','rolling_std_6',
     'prophet_trend']
    ```
    """)

    st.success("Now the model sees both **local behaviour (lags/rolling)** and **global trend (Prophet)**.")

    st.markdown("---")

    # ============================================================
    # 7️⃣ XGBOOST VS HYBRID (PROPHET + XGBOOST) — COMPARISON
    # ============================================================
    st.subheader("7️⃣ XGBoost vs Hybrid Model — Region-wise Performance Comparison")

    st.write("""
    The hybrid model was trained **region-wise** for all regions:

    - For each region:
      - Prophet produced an hourly trend → `prophet_trend`  
      - XGBoost was trained with all engineered features + trend  

    ### 🔍 Key Observations:
    - MAPE improved in **almost every region**  
    - Some regions improved significantly (e.g., 20, 10, 29, 13)  
    - Average error came down from **0.037 → 0.0336**
    """)

    st.image("image5.png", caption="Region-wise MAPE Comparison: XGBoost vs Hybrid (Prophet + XGBoost)", use_container_width=False)

    st.write("""
    ### 🧾 Example Comparison

    | Region | XGBoost MAPE | Hybrid (Prophet+XGB) MAPE |
    |--------|--------------|---------------------------|
    | 20     | 0.0207       | **0.0182** |
    | 10     | 0.0247       | **0.0213** |
    | 29     | 0.0281       | **0.0242** |
    | 14     | 0.0279       | **0.0249** |
    | 0      | 0.0317       | **0.0265** |
    | 22     | 0.0243       | **0.0213** |
    | 13     | 0.0228       | **0.0206** |

    **Overall:**
    - ✅ XGBoost Baseline Avg MAPE ≈ **0.0370**  
    - ✅ Hybrid Prophet + XGBoost Avg MAPE ≈ **0.0336**  
    - ➜ **Consistent improvement across city-wide regions**
    """)

    st.success("Conclusion: Hybrid modeling gives more stable, trend-aware, and accurate demand forecasts than any single model alone.")


# ---------------------------------------
# 7️⃣ HYBRID MODEL
# ---------------------------------------
elif menu == "6️⃣ Hyperparameter Tuning":

    st.header("🔀 Hyperparameter Tuning — Optimizing the Hybrid XGBoost Model")

    st.write("""
    After building the Hybrid Model (Prophet Trend + XGBoost), the next step was to **optimize performance**.
    Instead of manually guessing the values of hyperparameters like:

    - learning rate  
    - max_depth  
    - n_estimators  
    - subsample  
    - colsample_bytree  

    We used a **systematic and automated tuning framework (Optuna)** to find the best model
    configuration **for every region independently**.

    ---
    """)

    # ================================================================
    # 1️⃣ Why Hyperparameter Tuning Was Required
    # ================================================================
    st.subheader("1️⃣ Why Hyperparameter Tuning Was Necessary?")

    st.write("""
    The hybrid model uses **XGBoost**, which is extremely sensitive to the choice of hyperparameters.

    - Some regions are **stable** (office areas, airports) → shallow trees work well  
    - Some regions are **highly volatile** (nightlife zones, downtown areas) → deeper trees & lower LR  
    - Some regions have **periodic drop-offs** → more estimators needed  

    A *single global XGBoost configuration* cannot work for all 30 regions.

    **Therefore → tuning must be done region-wise.**
    """)

    st.info("Each NYC region has unique demand behaviour. One model configuration cannot fit all.")

    st.markdown("---")

    # ================================================================
    # 2️⃣ Why Region-wise Tuning Instead of Global Tuning?
    # ================================================================
    st.subheader("2️⃣ Why Tuning Per Region?")

    st.write("""
    **Reason 1 — Different demand patterns:**  
    Midtown ≠ Residential ≠ Airports.  
    Optimal parameters vary drastically between regions.

    **Reason 2 — Error patterns vary:**  
    - Regions with sharp spikes need **low learning_rate**  
    - Smooth regions benefit from **higher subsample**  
    - Crowded regions need **more estimators**  

    **Reason 3 — Better accuracy:**  
    Region-wise tuning improved it to **0.0301 MAPE**.

    → This is a **~19% improvement**.
    """)

    st.success("Region-wise tuning significantly boosts accuracy and stability.")

    st.markdown("---")

    # ================================================================
    # 3️⃣ What Hyperparameters Were Tuned?
    # ================================================================
    st.subheader("3️⃣ Hyperparameters Optimized Using Optuna")

    st.write("""
    Optuna searched the best value combinations from these ranges:

    - **learning_rate:** 0.01 → 0.3  
    - **max_depth:** 3 → 12  
    - **n_estimators:** 200 → 800  
    - **subsample:** 0.5 → 1.0  
    - **colsample_bytree:** 0.5 → 1.0  
    - **min_child_weight:** 1 → 10  
    
    **Objective:** minimize `MAPE`.

    Optuna automatically:
    - samples hyperparameter combinations  
    - trains the model  
    - evaluates MAPE  
    - chooses the best configuration  
    """)

    st.info("Optuna finds optimal parameters faster than manual grid/random search.")

    st.markdown("---")

    # ================================================================
    # 4️⃣ Region-wise Optuna Tuning Results
    # ================================================================
    st.subheader("4️⃣ Region-wise Optuna Best MAPE Results")

    st.write("""
    Below are the **top-performing regions** after tuning.
    """)

    st.code("""
📊 Region-wise Best XGBoost (Optuna) Results:

    region  best_mape
    ----------------------
    20      0.016934
    3       0.018372
    22      0.018635
    13      0.018788
    10      0.019366
    29      0.020965
    27      0.021225
    0       0.021356
    18      0.021545
    11      0.021646
    6       0.022140
    23      0.022854
    26      0.023066
    9       0.023581
    15      0.024303
    14      0.024818
    ...
""")

    st.success("🏁 Average MAPE across all regions after tuning: **0.0301**")

    st.write("""
    ✔ This tuning reduced errors by **19%**  
    ✔ Variability across regions reduced significantly  
    ✔ The model became more stable for low-demand regions  
    """)

    st.info("Optuna results were saved to: `optuna_xgb_regionwise_results.csv`")

    st.markdown("---")

    # ================================================================
    # 5️⃣ Interpretation — What Did Tuning Teach Us?
    # ================================================================
    st.subheader("5️⃣ What Insights Did Hyperparameter Tuning Provide?")

    st.write("""
    **Insight 1 — Business districts need shallow trees**  
    Midtown (Region 3, 22, 13) behaves smoothly → best depth = 3–4  

    **Insight 2 — Airport regions need deeper models**  
    JFK/LGA zones (Region 20, 10, 29) have big spikes → depth = 8–10  

    **Insight 3 — Learning rate is highly region dependent**  
    - Volatile regions → LR = 0.01–0.05  
    - Stable regions → LR = 0.1–0.2  

    **Insight 4 — Hybrid models benefit more from tuning**  
    Because `prophet_trend` adds new information, parameter tuning helps XGBoost:
    - adjust tree complexity  
    - reduce overfitting  
    - balance trend vs noise  
    """)

    st.success("Hyperparameter tuning improved both performance and interpretability.")

    st.markdown("---")

    # ================================================================
    # 6️⃣ Summary of the Hyperparameter Tuning Stage
    # ================================================================
    st.subheader("6️⃣ Final Summary — Tuning Stage Impact")

    st.write("""
    - Performed **region-wise Optuna tuning**  
    - Evaluated hundreds of models per region  
    - Found the best XGBoost configuration for **every** demand zone  
    - Reduced MAPE from **0.037 → 0.0301**  
    - Increased accuracy, stability, and robustness  

    **This tuning stage prepares the model for final deployment in the Hybrid Forecasting step.**
    """)

    st.success("Hyperparameter tuning complete — ready for final Hybrid model deployment ✔")


elif menu == "7️⃣ DVC Pipeline":
    st.header("🔀 End-to-End DVC Pipeline Overview")

    st.write("""
    The entire forecasting system is automated and reproducible using **DVC (Data Version Control)**.
    Each stage converts raw NYC taxi trip records → region-wise hybrid forecasting models.
    Below is the exact structure and purpose of every stage.
    """)

    st.markdown("---")

    # ---------------------------------------------------------
    # 1️⃣ INGEST STAGE
    # ---------------------------------------------------------
    st.subheader("1️⃣ Data Ingestion & Cleaning (`ingest` stage)")

    st.write("""
    **Purpose:**  
    Safely load ~38M rows of raw taxi CSV files and clean them.

    **Key Operations:**  
    - Loads data using **Dask** to avoid memory overflow.  
    - Removes invalid coordinates, distance/fare outliers, and unused fields.  
    - Produces a clean, standardized dataset.

    **Output:**  
    `data/interim/df_cleaned.parquet`
    """)

    st.info("Why needed? → Raw NYC taxi data is extremely large & noisy; must be cleaned once in a reproducible step.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 2️⃣ EXTRACT FEATURES
    # ---------------------------------------------------------
    st.subheader("2️⃣ Region Clustering & 15-Min Time-Series (`extract_features`)")

    st.write("""
    **Purpose:**  
    Convert cleaned coordinates into meaningful pickup regions & 15-min demand signals.

    **Key Operations:**  
    - Normalizes coordinates using **StandardScaler**.  
    - Applies **MiniBatchKMeans** to create **30 pickup regions**.  
    - Aggregates all trips into **15-minute pickup counts per region**.  
    - Optimizes **EWMA smoothing (α)** per region for stable baseline signals.

    **Outputs:**  
    - `data/processed/resampled_data.csv`  
    - `models/scaler.joblib`  
    - `models/mb_kmeans.joblib`
    """)

    st.info("Why needed? → Forecasting works only after data is converted into region-wise time-series.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 3️⃣ FEATURE PROCESSING
    # ---------------------------------------------------------
    st.subheader("3️⃣ Feature Engineering for Forecasting (`feature_processing`)")

    st.write("""
    **Purpose:**  
    Build ML-ready features for XGBoost.

    **Key Features Created:**  
    - Lag features → `lag_1`, `lag_2`, `lag_3`, `lag_4`  
    - Rolling statistics → `rolling_mean_3`, `rolling_mean_6`, `rolling_std_3`, `rolling_std_6`  
    - Date features → `day_of_week`, `month`  
    - Merges smoothing signal → `avg_pickups`

    **Train/Test Split:**  
    - Train → January & February  
    - Test → March

    **Outputs:**  
    - `train.csv`  
    - `test.csv`
    """)

    st.info("Why needed? → XGBoost cannot learn time patterns unless we explicitly create lag & rolling features.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 4️⃣ TRAIN STAGE
    # ---------------------------------------------------------
    st.subheader("4️⃣ Hybrid Model Training per Region (`train`)")

    st.write("""
    **Purpose:**  
    Train a **hybrid forecasting model** for each of the 30 pickup regions.

    **What Happens:**  
    - Prophet models hourly trend & seasonality.  
    - Trend is merged back into 15-min dataset as `prophet_trend`.  
    - XGBoost learns short-term, non-linear behavior using full feature set.

    **Outputs:**  
    - `prophet_region_*.joblib`  
    - `xgb_region_*.joblib`
    """)

    st.info("Why hybrid? → Prophet handles long-term patterns, XGBoost handles short-term spikes.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 5️⃣ EVALUATE STAGE
    # ---------------------------------------------------------
    st.subheader("5️⃣ Region-Wise Evaluation (`evaluate`)")

    st.write("""
    **Purpose:**  
    Validate hybrid performance for each region independently.

    **What Happens:**  
    - Prophet produces hourly predictions for March.  
    - Predictions are mapped back to 15-min timestamps.  
    - XGBoost corrects the short-term variations.  
    - Final predictions are compared with actual data using **MAPE**.

    **Output:**  
    `regionwise_mape.csv`
    """)

    st.success("Average Hybrid MAPE across all 30 regions: **0.0301**")

    st.markdown("---")

    # ---------------------------------------------------------
    # 6️⃣ REGISTER (SKIPPED)
    # ---------------------------------------------------------
    st.subheader("6️⃣ Model Registry")

    st.write("""
    Two stages are intentionally frozen:

    - **`register_model`**  
      Would register Prophet + XGBoost models to MLflow Model Registry.

    - **`register_scaler_kmeans`**  
      Would upload scaler + KMeans preprocessing artifacts.

    """)

    st.markdown("---")

elif menu == "8️⃣ Conclusion":
    st.header("📌 8️⃣ Conclusion — End-to-End Demand Forecasting System")

    st.write("""
    This project demonstrates a complete **big-data to forecasting pipeline** for NYC taxi demand,
    converting over **33 million raw pickup records** into accurate **15-minute interval forecasts**
    across 30 geographic regions.

    The final system is optimized, scalable, and production-ready, powered by clustering,
    time-series feature engineering, Prophet trend modeling, and XGBoost learning.
    """)

    st.markdown("---")

    # -----------------------------------------------------------
    # 🎯 PROJECT OUTCOME
    # -----------------------------------------------------------
    st.subheader("🎯 Key Outcomes")

    st.write("""
    - Built a **scalable preprocessing pipeline** using Dask + chunk-based processing.  
    - Created **30 geographic regions** using MiniBatchKMeans for better spatial forecasting.  
    - Generated strong **time-series features** (lags, rolling stats, EWMA).  
    - Developed **Prophet**, **XGBoost**, and **Hybrid Prophet+XGBoost** models.  
    - Performed **region-wise model training** and **hyperparameter tuning**.  
    - Obtained a highly accurate final model with:

      ### ⭐ **Final Average Hybrid MAPE ≈ 3.21%**
    """)

    st.success("🚀 The hybrid approach significantly improved forecasting accuracy across most regions.")

    st.markdown("---")

    # -----------------------------------------------------------
    # 🧠 WHY THE HYBRID MODEL WORKED BEST
    # -----------------------------------------------------------
    st.subheader("🧠 Why the Hybrid Model Worked Best")

    st.write("""
    - Prophet captured **trend + daily + weekly seasonality** very accurately.  
    - XGBoost captured **short-term spikes**, **non-linearities**, and **region-specific patterns**.  
    - Combining them allowed XGBoost to learn local deviations from Prophet's trend.

    **Final Formula:**  
    ### `Final Forecast = Prophet Trend + XGBoost Residual Corrections`
    """)

    st.info("Hybrid model = global pattern understanding + local short-term intelligence.")

    st.markdown("---")


    # -----------------------------------------------------------
    # 🧩 TECH STACK
    # -----------------------------------------------------------
    st.subheader("🧩 Tech Stack Used")

    st.write("""
    **Data Processing:** Dask, Pandas 
    **Clustering:** MiniBatchKMeans  
    **Feature Engineering:** NumPy, Pandas  
    **Forecasting Models:** Prophet, XGBoost
    **Optimization:** Optuna (region-wise tuning)  
    **Model Tracking:** MLflow  
    **Automation:** DVC Pipeline  
    **Deployment:** Streamlit Dashboard  
    """)

    st.markdown("---")

    # -----------------------------------------------------------
    # 🔗 REPOSITORIES
    # -----------------------------------------------------------
    st.subheader("🔗 Project Repositories")

    st.write("""
    - 📦 **Experimentation Repo:**  
      https://github.com/apoorvtechh/demand_preidction_experimentation  

    - 🚀 **Final Production Pipeline Repo:**  
      https://github.com/apoorvtechh/demand_forecasting  
    """)

    st.markdown("---")

    # -----------------------------------------------------------
    # 🏁 FINAL MESSAGE
    # -----------------------------------------------------------
    st.subheader("🏁 Final Note")

    st.write("""
    This project proves how combining **big data engineering**, **time-series modeling**, and 
    **machine learning** can produce highly accurate demand forecasting systems.

    The pipeline is modular, reusable, and ready for scaling to other cities or industries.
    """)

    st.success("🎉 Project Completed — End-to-End Hybrid Forecasting System Ready!")
 
