## Data Cleaning

data_path = "data/raw"   

files = [
    "Constructor_Performance.csv",
    "Constructor_Rankings.csv",
    "Driver_Details.csv",
    "Driver_Rankings.csv",
    "Lap_Timings.csv",
    "Pit_Stop_Records.csv",
    "Qualifying_Results.csv",
    "Race_Results.csv",
    "Race_Schedule.csv",
    "Race_Status.csv",
    "Season_Summaries.csv",
    "Sprint_Race_Results.csv",
    "Team_Details.csv",
    "Track_Information.csv"
]

dfs = {f.replace(".csv", ""): pd.read_csv(os.path.join(data_path, f), low_memory=False) for f in files}

print(f" {len(dfs)} dataset berhasil dimuat.")

def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.lower()
    )
    return df

dfs = {name: clean_columns(df) for name, df in dfs.items()}
print(" Nama kolom telah dinormalisasi.")

qual = dfs["Qualifying_Results"]

# missing value check
print("\nSebelum imputasi missing values:")
print(qual.isna().sum())

# karena q2 dan q3 kosong untuk pembalap yang tidak lolos ke sesi tersebut, maka isi dengan "DNQ" (Did Not Qualify)
qual["q2"] = qual["q2"].fillna("DNQ")
qual["q3"] = qual["q3"].fillna("DNQ")

dfs["Qualifying_Results"] = qual

print("\nSetelah imputasi missing values:")
print(qual.isna().sum())

schedule = dfs["Race_Schedule"]
date_cols = ["date", "fp1_date", "fp2_date", "fp3_date", "quali_date", "sprint_date"]

for col in date_cols:
    if col in schedule.columns:
        schedule[col] = pd.to_datetime(schedule[col], errors="coerce")

dfs["Race_Schedule"] = schedule
print("\n Kolom tanggal di Race_Schedule telah dikonversi ke datetime.")

for name, df in dfs.items():
    n_before = len(df)
    df = df.drop_duplicates()
    n_after = len(df)
    dfs[name] = df
    if n_before != n_after:
        print(f"⚠️ {name}: {n_before - n_after} duplikat dihapus.")
print(" Pemeriksaan duplikasi selesai.")

if "driver_details" in dfs:
    dfs["Driver_Details"]["forename"] = dfs["Driver_Details"]["forename"].str.lower()
    dfs["Driver_Details"]["surname"] = dfs["Driver_Details"]["surname"].str.lower()

if "team_details" in dfs:
    dfs["Team_Details"]["name"] = dfs["Team_Details"]["name"].str.lower()

print(" Normalisasi teks selesai.")

print("\nRingkasan setelah cleaning:")
for name, df in dfs.items():
    print(f"{name:25s} | rows: {len(df):7d} | cols: {df.shape[1]:2d} | missing: {df.isna().sum().sum():4d}")


# save

output_path = "data/processed"
os.makedirs(output_path, exist_ok=True)

for name, df in dfs.items():
    df.to_csv(os.path.join(output_path, f"{name}_clean.csv"), index=False)

print(f"\n Semua file hasil cleaning disimpan di folder: {output_path}")

race_schedule = pd.read_csv("data/raw/Race_Schedule.csv")

missing_summary = race_schedule.isnull().sum().to_frame('missing_count')
missing_summary['missing_pct'] = 100 * missing_summary['missing_count'] / len(race_schedule)
display(missing_summary)

display(race_schedule[race_schedule.isnull().any(axis=1)].head(10))

os.makedirs("data/processed", exist_ok=True)

for name, df in dfs.items():
    df.to_csv(os.path.join(output_path, f"{name}_clean.csv"), index=False)
print(dfs.keys())

## Data Integration


processed_path = "data/processed"

dfs = {
    "Race_Results": pd.read_csv(os.path.join(processed_path, "Race_Results_clean.csv")),
    "Driver_Details": pd.read_csv(os.path.join(processed_path, "Driver_Details_clean.csv")),
    "Team_Details": pd.read_csv(os.path.join(processed_path, "Team_Details_clean.csv")),
    "Race_Schedule": pd.read_csv(os.path.join(processed_path, "Race_Schedule_clean.csv")),
    "Track_Information": pd.read_csv(os.path.join(processed_path, "Track_Information_clean.csv"))
}

print(" Dataset berhasil dimuat:")
for name, df in dfs.items():
    print(f"{name:20s} | rows: {len(df):6d} | cols: {df.shape[1]:2d}")

for name, df in dfs.items():
    print(f"\n📄 {name} columns:")
    print(df.columns.tolist())


processed_path = "data/processed"

dfs = {
    "Race_Results": pd.read_csv(os.path.join(processed_path, "Race_Results_clean.csv")),
    "Driver_Details": pd.read_csv(os.path.join(processed_path, "Driver_Details_clean.csv")),
    "Team_Details": pd.read_csv(os.path.join(processed_path, "Team_Details_clean.csv")),
    "Race_Schedule": pd.read_csv(os.path.join(processed_path, "Race_Schedule_clean.csv")),
    "Track_Information": pd.read_csv(os.path.join(processed_path, "Track_Information_clean.csv"))
}

print(" Dataset berhasil dimuat:")
for name, df in dfs.items():
    print(f"{name:20s} | rows: {len(df):6d} | cols: {df.shape[1]:2d}")

for name in ["Driver_Details", "Team_Details", "Race_Schedule", "Track_Information"]:
    if "url" in dfs[name].columns:
        dfs[name] = dfs[name].rename(columns={"url": f"url_{name.lower()}"})

# data integration

# Race_Results + Driver_Details
merged = pd.merge(
    dfs["Race_Results"],
    dfs["Driver_Details"],
    on="driverid",
    how="left"
)

# + Team_Details
merged = pd.merge(
    merged,
    dfs["Team_Details"],
    on="constructorid",
    how="left"
)

# + Race_Schedule
merged = pd.merge(
    merged,
    dfs["Race_Schedule"],
    on="raceid",
    how="left"
)

# + Track_Information 
if "Track_Information" in dfs:
    merged = pd.merge(
        merged,
        dfs["Track_Information"],
        on="circuitid",
        how="left"
    )

print(f"\n✅ Hasil integrasi data: {merged.shape[0]} baris x {merged.shape[1]} kolom")

# save
output_path = "data/processed"
os.makedirs(output_path, exist_ok=True)
merged.to_csv(os.path.join(output_path, "Integrated_Data.csv"), index=False)

print(f"📂 Data terintegrasi disimpan ke: {output_path}/Integrated_Data.csv")
