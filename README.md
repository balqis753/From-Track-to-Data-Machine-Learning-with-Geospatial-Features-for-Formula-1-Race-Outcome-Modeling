# From-Track-to-Data-Machine-Learning-with-Geospatial-Features-for-Formula-1-Race-Outcome-Modeling

A comprehensive spatial data science project that predicts Formula 1 race outcomes by integrating geospatial track features, driver performance analytics, and machine learning techniques.


## Overview

This project bridges geodetic engineering principles with data science to analyze Formula 1 racing through a unique spatial lens. By integrating geospatial circuit characteristics (latitude, longitude, altitude) with traditional racing metrics, this work demonstrates how physical geography influences competitive outcomes in motorsport.

**Key Objectives:**
- Predict race winners using geospatial and performance features
- Analyze how circuit geography affects racing outcomes
- Identify driver performance archetypes through clustering
- Map Formula 1's global expansion and competitive evolution

## Dataset

**Source:** Comprehensive Formula 1 historical data (1950-2024) = https://www.kaggle.com/datasets/muhammadehsan02/formula-1-world-championship-history-1950-2024?resource=download

**Components:**
- 26,519 race records
- 14 CSV files covering circuits, drivers, constructors, races, lap times, pit stops, and qualifying results
- 60+ engineered features combining spatial and temporal dimensions

**Key Data Files:**
- `Track_Information.csv` - Circuit locations and characteristics
- `Driver_Details.csv` - Driver profiles and career statistics
- `Race_Results.csv` - Historical race outcomes
- `Constructor_Performance.csv` - Team performance metrics
- `Lap_Timings.csv` - Detailed lap-by-lap data

## Methodology

### 1. Data Preparation
- **Integration**: Consolidated 6 relational F1 datasets into unified analytical framework
- **Cleaning**: Standardized units, synchronized temporal references, handled null values
- **Feature Engineering**: Created racing-specific metrics (win rates, podium frequencies, era classifications)

### 2. Exploratory Data Analysis
- Statistical profiling with correlation mapping
- Era-based dominance assessment
- Performance trend analysis across F1 history
- Geospatial distribution patterns

### 3. Predictive Modeling
- **Classification Models**: XGBoost and Random Forest for race winner and podium prediction
- **Regression Models**: OLS for lap time forecasting
- **Time Series**: Championship points evolution analysis
- **Spatial Models**: Geographic performance pattern recognition

### 4. Advanced Analytics
- **Clustering**: K-Means driver segmentation into 4 performance profiles
- **Anomaly Detection**: Isolation Forest for outlier identification
- **Statistical Testing**: Hypothesis testing across technological eras
- **Geospatial Analysis**: Altitude, latitude, and longitude impact on performance

## Technologies Used

### Programming & Libraries
- **Python 3.8+** - Core programming language
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting framework
- **Statsmodels** - Statistical modeling and hypothesis testing

### Visualization & Geospatial
- **Plotly** - Interactive visualizations and geospatial mapping
- **Seaborn** - Statistical data visualization
- **GeoPandas** - Geospatial data operations
- **Folium** - Interactive maps

### Development Tools
- **Jupyter Notebook** - Interactive development environment
- **GitHub** - Version control

## Results

### Model Performance

**Championship Forecasting:**
- Accuracy: 97.7%
- Successfully predicted dominant drivers (Verstappen, Hamilton, Schumacher)

**Lap Time Prediction:**
- R² Score: 0.34
- Moderate predictive capability, revealing circuit-specific variance

### Findings

**Geospatial Insights:**
- **Altitude Impact**: Negative correlation with fastest lap speeds (high-altitude circuits show reduced velocities)
- **Latitude Patterns**: Tropical circuits exhibit higher lap time variability due to temperature/humidity
- **Geographic Clusters**: European circuits show tighter performance clustering vs. equatorial venues

**Statistical Discoveries:**
- Grid position coefficient: -0.16 (p < 0.001) - qualifying advantage is crucial
- Progressive performance improvement from 1950s to present
- Four distinct driver archetypes: Dominant Champions, Midfield Specialists, Opportunistic Scorers, and Outliers

**Constructor Dominance Eras:**
- Ferrari dominance (2000s)
- Red Bull Racing supremacy (2010s-2020s)
- Mercedes hybrid era (2014-2020)

## Visualizations

### Interactive Maps
- **Global Circuit Distribution**: Worldwide F1 venue mapping with performance overlays
- **Constructor Championship Evolution**: Animated timeline (1980-2024)
- **Circuit Performance Heatmaps**: Geographic performance patterns

### Dashboards
- Championship timeline visualization
- Constructor dominance analysis
- Driver head-to-head comparison tool
- Circuit-specific performance breakdowns
- Real-time 2024 season tracking

### Static Visualizations
- Correlation heatmaps
- Era-based performance trends
- Driver clustering pairplots
- Statistical distribution analyses


##  Acknowledgments

- Formula 1 data sourced from [Kaggle F1 Dataset]
- Inspired by the intersection of geospatial engineering and data science
- Built with passion for motorsport analytics and spatial intelligence

---

A data enthusiast and F1 fan who saw racing data as the perfect playground for spatial analytics.
**Keywords:** #DataScience #MachineLearning #GeospatialAnalysis #Formula1 #Python #PredictiveModeling #SportsAnalytics #FeatureEngineering #Visualization
