import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "historical_freight_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "multi_horizon_freight_data.csv"


# ============================================================
# 2. LOAD AND SORT DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 3. LAG FEATURES
# ============================================================

lag_columns = [
    "Freight_Rate_USD",
    "BDI_Index",
    "Bunker_Fuel_USD",
    "Coal_Price_USD",
    "Port_Wait_Days",
]

lag_periods = [1, 3, 7, 14, 30]

for column in lag_columns:
    for lag in lag_periods:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)


# ============================================================
# 4. ROLLING FEATURES
# ============================================================

rolling_windows = [7, 14, 30]

for window in rolling_windows:

    df[f"Freight_MA_{window}"] = (
        df["Freight_Rate_USD"]
        .rolling(window)
        .mean()
    )

    df[f"Freight_STD_{window}"] = (
        df["Freight_Rate_USD"]
        .rolling(window)
        .std()
    )

    df[f"BDI_MA_{window}"] = (
        df["BDI_Index"]
        .rolling(window)
        .mean()
    )

    df[f"Bunker_MA_{window}"] = (
        df["Bunker_Fuel_USD"]
        .rolling(window)
        .mean()
    )

    df[f"Coal_MA_{window}"] = (
        df["Coal_Price_USD"]
        .rolling(window)
        .mean()
    )


# ============================================================
# 5. MOMENTUM FEATURES
# ============================================================

df["Freight_Change_1D"] = (
    df["Freight_Rate_USD"].pct_change(1)
)

df["Freight_Change_7D"] = (
    df["Freight_Rate_USD"].pct_change(7)
)

df["Freight_Change_14D"] = (
    df["Freight_Rate_USD"].pct_change(14)
)

df["Freight_Change_30D"] = (
    df["Freight_Rate_USD"].pct_change(30)
)

df["BDI_Change_7D"] = (
    df["BDI_Index"].pct_change(7)
)

df["Bunker_Change_7D"] = (
    df["Bunker_Fuel_USD"].pct_change(7)
)

df["Coal_Change_7D"] = (
    df["Coal_Price_USD"].pct_change(7)
)


# ============================================================
# 6. SEASONAL FEATURES
# ============================================================

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter
df["DayOfYear"] = df["Date"].dt.dayofyear

df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)


# ============================================================
# 7. MULTI-HORIZON TARGETS
# ============================================================

# 15-day forecast
df["Target_15D"] = (
    df["Freight_Rate_USD"].shift(-15)
)

# 30-day forecast
df["Target_30D"] = (
    df["Freight_Rate_USD"].shift(-30)
)


# ============================================================
# 8. DIRECTION TARGETS
# ============================================================

df["Direction_15D"] = np.where(
    df["Target_15D"] > df["Freight_Rate_USD"],
    1,
    0
)

df["Direction_30D"] = np.where(
    df["Target_30D"] > df["Freight_Rate_USD"],
    1,
    0
)


# ============================================================
# 9. DROP INVALID ROWS
# ============================================================

df = df.dropna().reset_index(drop=True)


# ============================================================
# 10. SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 11. SUMMARY
# ============================================================

print("=" * 65)
print("MULTI-HORIZON FEATURE ENGINEERING COMPLETE")
print("=" * 65)

print(f"Original records : 2069")
print(f"Usable records   : {len(df)}")
print(f"Total columns    : {len(df.columns)}")

print("\nForecast targets:")
print(" - Target_15D")
print(" - Target_30D")

print("\nDirection targets:")
print(" - Direction_15D")
print(" - Direction_30D")

print(f"\nOutput file:")
print(OUTPUT_FILE)

print("\nMULTI-HORIZON DATASET READY.")