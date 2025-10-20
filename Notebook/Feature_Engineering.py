## Feature Engineering

# Load integrated data 
integrated_path = "data/processed/Integrated_Data.csv"
df = pd.read_csv(integrated_path)

print(f" Integrated data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

df.columns = df.columns.str.lower()

# 1. Add winner and podium columns 
def classify_finish(pos):
    try:
        pos = int(pos)
        if pos == 1:
            return "win"
        elif pos in [2, 3]:
            return "podium"
        else:
            return "finish"
    except:
        return np.nan

df["finish_category"] = df["position"].apply(classify_finish)

# Data cleaning for numeric columns 
numeric_cols = ["grid", "position", "points"]

# Replace non-numeric values like '\N' or empty strings with NaN, then convert
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Check if conversion worked
print(df[numeric_cols].dtypes)
print(df[numeric_cols].isna().sum())

# 2. Calculate driver performance metrics
driver_perf = (
    df.groupby("driverid")
    .agg(
        total_races=("raceid", "count"),
        total_wins=("finish_category", lambda x: (x == "win").sum()),
        total_podiums=("finish_category", lambda x: (x.isin(["win", "podium"])).sum()),
        total_points=("points", "sum"),
        avg_grid=("grid", "mean"),
        avg_position=("position", "mean"),
    )
    .reset_index()
)

driver_perf["win_rate"] = driver_perf["total_wins"] / driver_perf["total_races"]
driver_perf["podium_rate"] = driver_perf["total_podiums"] / driver_perf["total_races"]

print(driver_perf.head())

# 3. Combine metrics with main data 
df = df.merge(driver_perf[["driverid", "win_rate", "podium_rate", "avg_position"]], on="driverid", how="left")

# 4 Era Classification (year) 
def classify_era(year):
    if year < 1990:
        return "Pre-Modern"
    elif 1990 <= year < 2014:
        return "V8 Era"
    elif 2014 <= year < 2022:
        return "Hybrid Turbo Era"
    else:
        return "Ground Effect Era"

df["f1_era"] = df["year"].apply(classify_era)

# 5. Seasonal Aggregation (Driver x Season) 
seasonal_driver = (
    df.groupby(["year", "driverid"])
    .agg(
        total_races=("raceid", "nunique"),
        total_points=("points", "sum"),
        avg_position=("position", "mean"),
        wins=("finish_category", lambda x: (x == "win").sum()),
        podiums=("finish_category", lambda x: (x.isin(["win", "podium"])).sum()),
    )
    .reset_index()
)
seasonal_driver["win_rate"] = seasonal_driver["wins"] / seasonal_driver["total_races"]
seasonal_driver["podium_rate"] = seasonal_driver["podiums"] / seasonal_driver["total_races"]

# 6. Performance Trend (moving average) 
seasonal_driver["trend_points_ma3"] = (
    seasonal_driver.groupby("driverid")["total_points"].transform(lambda x: x.rolling(3, min_periods=1).mean())
)

# 7. save feature engineering 
output_path = "data/processed"
os.makedirs(output_path, exist_ok=True)
df.to_csv(os.path.join(output_path, "F1_Integrated_Features.csv"), index=False)
seasonal_driver.to_csv(os.path.join(output_path, "Driver_Seasonal_Features.csv"), index=False)

print(f"""
 Feature engineering selesai!
 - {output_path}/F1_Integrated_Features.csv
 - {output_path}/Driver_Seasonal_Features.csv
""")