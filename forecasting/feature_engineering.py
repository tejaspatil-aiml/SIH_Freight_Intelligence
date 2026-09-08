import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "data" / "historical_freight_data.csv"
OUTPUT_FILE = BASE_DIR / "data" / "engineered_freight_data.csv"

df = pd.read_csv(INPUT_FILE)

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Sort chronologically
df = df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------
# 2. Historical Lag Features
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 3. Rolling Market Features
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 4. Momentum Features
# ---------------------------------------------------------

df["Freight_Change_1D"] = (
    df["Freight_Rate_USD"]
    .pct_change(1)
)

df["Freight_Change_7D"] = (
    df["Freight_Rate_USD"]
    .pct_change(7)
)

df["BDI_Change_7D"] = (
    df["BDI_Index"]
    .pct_change(7)
)

df["Bunker_Change_7D"] = (
    df["Bunker_Fuel_USD"]
    .pct_change(7)
)

df["Coal_Change_7D"] = (
    df["Coal_Price_USD"]
    .pct_change(7)
)


# ---------------------------------------------------------
# 5. Calendar / Seasonality Features
# ---------------------------------------------------------

df["DayOfWeek"] = df["Date"].dt.dayofweek
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter
df["DayOfYear"] = df["Date"].dt.dayofyear


# Cyclic representation of seasonality
df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)


# ---------------------------------------------------------
# 6. Future Forecast Targets
# ---------------------------------------------------------

# Forecast freight rate 7 days ahead
df["Target_7D"] = df["Freight_Rate_USD"].shift(-7)

# Direction of future freight movement
df["Direction_7D"] = np.where(
    df["Target_7D"] > df["Freight_Rate_USD"],
    1,
    0
)


# ---------------------------------------------------------
# 7. Remove rows created by lag/rolling operations
# ---------------------------------------------------------

df = df.dropna().reset_index(drop=True)


# ---------------------------------------------------------
# 8. Save engineered dataset
# ---------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)


# ---------------------------------------------------------
# 9. Summary
# ---------------------------------------------------------

print("=" * 60)
print("FREIGHT FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print(f"Original records : 2069")
print(f"Engineered records: {len(df)}")
print(f"Total features   : {len(df.columns)}")
print(f"Output file      : {OUTPUT_FILE}")

print("\nTarget columns:")
print(" - Target_7D")
print(" - Direction_7D")

print("\nFeature engineering completed successfully.")