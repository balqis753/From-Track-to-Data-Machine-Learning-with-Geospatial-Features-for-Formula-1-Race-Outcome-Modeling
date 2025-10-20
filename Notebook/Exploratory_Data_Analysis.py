## Exploratory Data Analysis (EDA)
### Descriptive Statistics
# • Summary statistics untuk key metrics
# • Distribution analysis
# • Correlation matrices
# • Time series patterns


df = pd.read_csv("data/processed/F1_Integrated_Features.csv")

print(f"Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head()

summary_stats = df.describe().T
display(summary_stats)

metrics = ["grid", "position", "points", "fastestlapspeed"]
display(df[metrics].describe())

# Plot distributions
plt.figure(figsize=(12, 5))
for i, col in enumerate(["points", "grid", "position"], 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

top_drivers = df.groupby("surname")["points"].sum().nlargest(10)
sns.barplot(x=top_drivers.values, y=top_drivers.index)
plt.title("Top 10 Drivers by Total Points")
plt.show()

numeric_cols = df.select_dtypes(include=[np.number])
corr = numeric_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Matrix of Numeric Features")
plt.show()

corr_target = corr["points"].sort_values(ascending=False)
print("Correlation with 'points':")
print(corr_target.head(10))

# Convert date to datetime (if exists)
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Points trend per season
if "year" in df.columns:
    season_points = df.groupby("year")["points"].sum().reset_index()
    sns.lineplot(data=season_points, x="year", y="points", marker="o")
    plt.title("Total Points per Season")
    plt.show()

driver_yearly = (
    df.groupby(["year", "surname"])["points"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(30, 10))  # Lebar 12 inch, tinggi 6 inch
sns.lineplot(
    data=driver_yearly[driver_yearly["surname"].isin(top_names)],
    x="year", y="points", hue="surname", marker="o"
)
plt.title("Points Trend by Top 5 Drivers")
plt.show()

### Business things
# • Driver analysis
# • Constructor analysis
# • Race & circuit analysis
# • Strategic analysis

#### Driver analysis

df = pd.read_csv("data/processed/F1_Integrated_Features.csv")
driver_details = pd.read_csv("data/processed/Driver_Details_clean.csv")

df["win"] = (df["position"] == 1).astype(int)
df["podium"] = (df["position"] <= 3).astype(int)

# Calculate performance metrics per driver
driver_perf = (
    df.groupby("driverid")
    .agg(
        total_points=("points", "sum"),
        total_wins=("win", "sum"),
        total_podiums=("podium", "sum"),
        total_races=("raceid", "count"),
    )
    .reset_index()
)

# Calculate the average points per race
driver_perf["avg_points_per_race"] = driver_perf["total_points"] / driver_perf["total_races"]

# Combine with the driver's name
driver_perf = driver_perf.merge(
    driver_details[["driverid", "forename", "surname", "nationality"]],
    on="driverid",
    how="left"
)
driver_perf["Driver"] = driver_perf["forename"] + " " + driver_perf["surname"]

driver_perf["Score"] = (
    driver_perf["total_points"].rank(ascending=False, method="dense") * 0.4
    + driver_perf["total_wins"].rank(ascending=False, method="dense") * 0.3
    + driver_perf["avg_points_per_race"].rank(ascending=False, method="dense") * 0.3
)
driver_perf["Overall_Rank"] = driver_perf["Score"].rank(ascending=True, method="dense")

# Select the 15 best drivers
top_drivers = driver_perf.sort_values("Overall_Rank").head(15)

# visualisation
plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_drivers,
    y="Driver",
    x="avg_points_per_race",
    palette="viridis"
)
plt.title("Top 15 F1 Drivers of All Time (Weighted Performance Ranking)", fontsize=14)
plt.xlabel("Average Points per Race")
plt.ylabel("")
plt.tight_layout()
plt.show()

# preview
display(top_drivers[["Overall_Rank", "Driver", "nationality", "total_points", "total_wins", "avg_points_per_race"]])

df["win"] = (df["position"] == 1).astype(int)
df["podium"] = (df["position"] <= 3).astype(int)

# Calculate performance per year per driver 
driver_yearly = (
    df.groupby(["year", "driverid"])
    .agg(
        total_points=("points", "sum"),
        avg_points_per_race=("points", "mean"),
        total_wins=("win", "sum"),
        total_podiums=("podium", "sum"),
        races=("raceid", "count"),
    )
    .reset_index()
)

# merge with drivers
driver_yearly = driver_yearly.merge(
    driver_details[["driverid", "forename", "surname"]],
    on="driverid",
    how="left"
)
driver_yearly["Driver"] = driver_yearly["forename"] + " " + driver_yearly["surname"]

# Select top driver
top_drivers = (
    df.groupby("driverid")["points"].sum().nlargest(6).index.tolist()
)
selected_drivers = driver_yearly[driver_yearly["driverid"].isin(top_drivers)]

# plot trend
plt.figure(figsize=(16, 6))
sns.lineplot(
    data=selected_drivers,
    x="year",
    y="avg_points_per_race",
    hue="Driver",
    linewidth=2.5
)
plt.title("Driver Performance Evolution Over Time", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Average Points per Race")
plt.legend(title="Driver", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# preview
display(selected_drivers.head(10))

df["position"] = pd.to_numeric(df["position"], errors="coerce")

# Calculate consistency metrics per driver
driver_consistency = (
    df.groupby("driverid")
    .agg(
        races=("raceid", "count"),
        avg_finish=("position", "mean"),
        std_finish=("position", "std"),
        avg_points=("points", "mean"),
        total_points=("points", "sum")
    )
    .reset_index()
)

# merge with drivers
driver_consistency = driver_consistency.merge(
    driver_details[["driverid", "forename", "surname"]],
    on="driverid",
    how="left"
)
driver_consistency["Driver"] = driver_consistency["forename"] + " " + driver_consistency["surname"]

# Filter drivers with a minimum of 30 races
driver_consistency = driver_consistency[driver_consistency["races"] >= 30]

driver_consistency["consistency_score"] = (
    (1 / (1 + driver_consistency["std_finish"])) * 0.5 +
    (1 / (1 + driver_consistency["avg_finish"])) * 0.5
)

# top 10 driver
top_consistent = driver_consistency.sort_values("consistency_score", ascending=False).head(10)

# Plot Top 10 Most Consistent Drivers
plt.figure(figsize=(10, 5))
sns.barplot(
    data=top_consistent,
    x="Driver",
    y="consistency_score",
    palette="viridis"
)
plt.title("Top 10 Most Consistent F1 Drivers", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Consistency Score")
plt.xlabel("")
plt.tight_layout()
plt.show()

# Preview
display(top_consistent[["Driver", "races", "avg_finish", "std_finish", "avg_points", "consistency_score"]])

df["position"] = pd.to_numeric(df["position"], errors="coerce")
df["points"] = pd.to_numeric(df["points"], errors="coerce")

# Count the number of races per driver per year
driver_year_stats = (
    df.groupby(["driverid", "year"])
    .agg(
        races=("raceid", "count"),
        avg_position=("position", "mean"),
        total_points=("points", "sum"),
        avg_points=("points", "mean")
    )
    .reset_index()
)

# rookie year per driver
first_year = driver_year_stats.groupby("driverid")["year"].min().reset_index()
first_year.columns = ["driverid", "rookie_year"]

driver_year_stats = driver_year_stats.merge(first_year, on="driverid", how="left")

driver_year_stats["experience_level"] = driver_year_stats.apply(
    lambda row: "Rookie" if row["year"] == row["rookie_year"] else "Veteran",
    axis=1
)

driver_year_stats = driver_year_stats.merge(
    driver_details[["driverid", "forename", "surname"]],
    on="driverid",
    how="left"
)
driver_year_stats["Driver"] = driver_year_stats["forename"] + " " + driver_year_stats["surname"]

# Rookie vs Veteran
rookie_vs_veteran = (
    driver_year_stats.groupby("experience_level")
    .agg(
        avg_finish=("avg_position", "mean"),
        avg_points=("avg_points", "mean"),
        races=("races", "sum")
    )
    .reset_index()
)

# Visualisation 
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(
    data=rookie_vs_veteran,
    x="experience_level",
    y="avg_points",
    palette="coolwarm",
    ax=axes[0]
)
axes[0].set_title("Avg Points per Race")
axes[0].set_ylabel("Average Points")

sns.barplot(
    data=rookie_vs_veteran,
    x="experience_level",
    y="avg_finish",
    palette="viridis",
    ax=axes[1]
)
axes[1].invert_yaxis()  
axes[1].set_title("Avg Finishing Position (Lower = Better)")
axes[1].set_ylabel("Average Finish")

plt.suptitle("Rookie vs Veteran Performance Comparison", fontsize=14)
plt.tight_layout()
plt.show()

display(rookie_vs_veteran)

#### Constructor analysis
df = pd.read_csv("data/processed/F1_Integrated_Features.csv")
team_details = pd.read_csv("data/processed/Team_Details_clean.csv")

df["points"] = pd.to_numeric(df["points"], errors="coerce")

# classification
def classify_era(year):
    if 1950 <= year <= 1959: return "1950s"
    elif 1960 <= year <= 1969: return "1960s"
    elif 1970 <= year <= 1979: return "1970s"
    elif 1980 <= year <= 1989: return "1980s"
    elif 1990 <= year <= 1999: return "1990s"
    elif 2000 <= year <= 2009: return "2000s"
    elif 2010 <= year <= 2019: return "2010s"
    else: return "2020s"

df["era"] = df["year"].apply(classify_era)

# Total points per constructor per era
era_dominance = (
    df.groupby(["era", "constructorid"])["points"]
    .sum()
    .reset_index()
)

# Combine with the constructor name
era_dominance = era_dominance.merge(
    team_details[["constructorid", "name"]],
    left_on="constructorid",
    right_on="constructorid",
    how="left"
)

# Pick the top constructor of each era
color_palette = {
    "Ferrari": "#ED1C24",
    "McLaren": "#FF8000",
    "Mercedes": "#00A19B",
    "Red Bull": "#223971",
    "Williams": "#00A0DD"
}


top_per_era = era_dominance.loc[era_dominance.groupby("era")["points"].idxmax()]

plt.figure(figsize=(10, 5))
sns.barplot(data=top_per_era, x="era", y="points", hue="name", dodge=False, palette=color_palette)
plt.title("Dominant Constructors per Era (1950s–2020s)", fontsize=14)
plt.ylabel("Total Points (Era Sum)")
plt.xlabel("Era")
plt.legend(title="Constructor", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

display(top_per_era[["era", "name", "points"]])

# Average points per constructor per year
constructor_trend = (
    df.groupby(["year", "constructorid"])["points"]
    .sum()
    .reset_index()
)

constructor_trend = constructor_trend.merge(
    team_details[["constructorid", "name"]],
    left_on="constructorid",
    right_on="constructorid",
    how="left"
)

# Top 5 constructor based on points
top_5 = constructor_trend.groupby("name")["points"].sum().nlargest(5).index
trend_top5 = constructor_trend[constructor_trend["name"].isin(top_5)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=trend_top5, x="year", y="points", hue="name", linewidth=2)
plt.title("Evolution of Top 5 Constructors Over Time", fontsize=14)
plt.ylabel("Season Total Points")
plt.xlabel("Year")
plt.legend(title="Constructor", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# Calculate the total points based on the driver–constructor combination
combo_success = (
    df.groupby(["constructorid", "driverid"])["points"]
    .sum()
    .reset_index()
)

driver_details = pd.read_csv("data/processed/Driver_Details_clean.csv")

combo_success = (
    combo_success.merge(
        team_details[["constructorid", "name"]],
        left_on="constructorid",
        right_on="constructorid",
        how="left"
    )
    .merge(
        driver_details[["driverid", "forename", "surname"]],
        on="driverid",
        how="left"
    )
)

combo_success["Driver"] = combo_success["forename"] + " " + combo_success["surname"]

color_palette = {
    "Ferrari": "#ED1C24",
    "McLaren": "#FF8000",
    "Mercedes": "#00A19B",
    "Red Bull": "#223971",
    "Williams": "#00A0DD"
}

top_combos = combo_success.nlargest(10, "points")

plt.figure(figsize=(10, 5))
sns.barplot(
    data=top_combos,
    y="Driver",
    x="points",
    hue="name",
    palette=color_palette,
    dodge=False
)
plt.title("Top 10 Most Successful Constructor–Driver Combinations", fontsize=14)
plt.xlabel("Total Points Together")
plt.ylabel("Driver")
plt.legend(title="Constructor", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

display(top_combos[["Driver", "name", "points"]])


#### Race & circuit analysis
# Average points per winner per circuit
circuit_perf = (
    df.groupby("name")["points"]
    .mean()
    .reset_index()
    .sort_values("points", ascending=False)
)

plt.figure(figsize=(12, 5))
sns.barplot(data=circuit_perf.head(15), x="points", y="name", palette="viridis")
plt.title("Top Circuits by Average Points per Race", fontsize=14)
plt.xlabel("Average Points Scored")
plt.ylabel("Circuit")
plt.tight_layout()
plt.show()

# Average points per country (proxy: geographical conditions & weather)
country_weather_proxy = (
    df.groupby("country")["points"]
    .mean()
    .reset_index()
    .sort_values("points", ascending=False)
)

plt.figure(figsize=(10, 5))
sns.barplot(data=country_weather_proxy.head(10), x="points", y="country", palette="coolwarm")
plt.title("Average Race Performance by Country (Weather Proxy)", fontsize=14)
plt.xlabel("Average Points")
plt.ylabel("Country")
plt.tight_layout()
plt.show()


#### Strategic analysis

# PIT STOP STRATEGY EFFECTIVENESS 

df["position_change"] = df["grid"] - df["position"]

avg_improvement = df.groupby("constructorref")["position_change"].mean().sort_values(ascending=False).reset_index()

plt.figure(figsize=(25, 9))
sns.barplot(x="constructorref", y="position_change", data=avg_improvement, palette="crest")
plt.xticks(rotation=90, ha='right')
plt.title("Average Position Improvement (Proxy for Strategy Effectiveness)")
plt.ylabel("Average Grid-to-Finish Gain")
plt.xlabel("Constructor")
plt.tight_layout()
plt.show()

# QUALIFYING POSITION vs RACE RESULT CORRELATION

plt.figure(figsize=(7, 5))
sns.scatterplot(x="grid", y="position", alpha=0.5, data=df, color="teal")
sns.regplot(x="grid", y="position", data=df, scatter=False, color="black")
plt.title("Correlation between Qualifying and Race Result")
plt.xlabel("Grid Position (Start)")
plt.ylabel("Finish Position")
plt.gca().invert_xaxis()  # grid 1 = pole/grid position
plt.gca().invert_yaxis()  # position 1 = race/finish winner
plt.tight_layout()
plt.show()

corr = df["grid"].corr(df["position"])
print(f"Correlation between grid and finish position: {corr:.3f}")
if corr > 0.7:
    print("Strong correlation: starting position greatly influences finishing result.")
elif corr > 0.4:
    print("Moderate correlation: grid position has noticeable impact.")
else:
    print("Weak correlation: races are more unpredictable.")

# SPRINT RACE IMPACT ON MAIN RACE

if "sprint_date" in df.columns:
    sprint_summary = df.groupby("year")["round"].count().reset_index(name="sprint_count")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sprint_summary, x="year", y="sprint_count", palette="viridis")
    plt.title("Sprint Races per Year")
    plt.ylabel("Count of Sprint Races")
    plt.xlabel("Year")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
else:
    print("Jika Kolom 'sprint_date' belum berisi nilai valid, maka sprint analysis dilewati.")

# CHAMPIONSHIP BATTLE ANALYSIS per DECADE

f1 = pd.read_csv("data/processed/F1_Integrated_Features.csv")

f1 = f1.dropna(subset=["year", "points"])
f1["year"] = f1["year"].astype(int)

def get_decade(year):
    return f"{year//10*10}s"

f1["decade"] = f1["year"].apply(get_decade)

driver_year_points = (
    f1.groupby(["year", "driverref"])["points"]
    .sum()
    .reset_index()
)

driver_year_points["decade"] = driver_year_points["year"].apply(get_decade)

top5_per_decade = (
    driver_year_points.groupby(["decade", "driverref"])["points"]
    .sum()
    .reset_index()
    .sort_values(["decade", "points"], ascending=[True, False])
    .groupby("decade")
    .head(5)
)

# Filter data 
filtered_data = driver_year_points.merge(top5_per_decade[["decade", "driverref"]], on=["decade", "driverref"])

max_points = filtered_data["points"].max()

sns.set(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(len(decades), 1, figsize=(14, 5*len(decades)), sharex=False)

for i, decade in enumerate(decades):
    ax = axes[i]
    data = filtered_data[filtered_data["decade"] == decade]

    
    decade_start = int(decade[:4])
    decade_end = decade_start + 9

    for driver in data["driverref"].unique():
        driver_data = data[data["driverref"] == driver]
        ax.plot(driver_data["year"], driver_data["points"], marker="o", label=driver)

    ax.set_title(f"Championship Battle in {decade} (Top 5 Drivers)", fontsize=14, weight="bold")
    ax.set_ylabel("Points per Year")
    ax.set_ylim(0, max_points)
    ax.set_xlim(decade_start, decade_end)
    ax.legend(title="Driver", loc="upper left")

axes[-1].set_xlabel("Year")
plt.tight_layout()
plt.show()


