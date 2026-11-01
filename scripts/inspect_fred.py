import urllib.request
import csv
import io
import pandas as pd

series = ["SP500", "DJIA", "NASDAQCOM"]
dfs = []
for s in series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        df = pd.read_csv(r)
        # rename date column if needed
        df.columns = ["date", s]
        # replace '.' with NaN
        df[s] = pd.to_numeric(df[s], errors="coerce")
        dfs.append(df)

merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on="date", how="outer")

# filter date range: 2021-01-01 to 2023-01-19
merged["date"] = pd.to_datetime(merged["date"])
sub = merged[(merged["date"] >= "2021-01-01") & (merged["date"] <= "2023-01-19")]
# drop any missing series
sub_clean = sub.dropna().sort_values("date").reset_index(drop=True)
print("Row count:", len(sub_clean))
print("Expected:", 515)
print("Start date:", sub_clean["date"].min().strftime("%Y-%m-%d"))
print("End date:", sub_clean["date"].max().strftime("%Y-%m-%d"))
print("Sample head:")
print(sub_clean.head())
print("Sample tail:")
print(sub_clean.tail())

