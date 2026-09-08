from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

HISTORICAL_DATA_FILE = DATA_DIR / "historical_freight_data.csv"
OUTPUT_FILE = OUTPUT_DIR / "forecast_output.json"

MODEL_FILES = {
    "7D": {
        "P10": MODEL_DIR / "freight_model_p10.pkl",
        "P50": MODEL_DIR / "freight_model_p50.pkl",
        "P90": MODEL_DIR / "freight_model_p90.pkl",
    },
    "15D": {
        "P10": MODEL_DIR / "freight_15d_p10.pkl",
        "P50": MODEL_DIR / "freight_15d_p50.pkl",
        "P90": MODEL_DIR / "freight_15d_p90.pkl",
    },
    "30D": {
        "P10": MODEL_DIR / "freight_30d_p10.pkl",
        "P50": MODEL_DIR / "freight_30d_p50.pkl",
        "P90": MODEL_DIR / "freight_30d_p90.pkl",
    },
}

FEATURE_FILES = {
    "7D": MODEL_DIR / "forecast_features.pkl",
    "MULTI": MODEL_DIR / "multi_horizon_features.pkl",
}


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_latest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuilds ONLY historical features required by the trained models.

    Important:
    We use the latest RAW observation instead of taking the last
    row from an engineered dataset whose future targets are truncated.
    """

    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    lag_columns = [
        "Freight_Rate_USD",
        "BDI_Index",
        "Bunker_Fuel_USD",
        "Coal_Price_USD",
        "Port_Wait_Days",
    ]

    lag_periods = [1, 3, 7, 14, 30]

    # ----------------------------
    # Lag features
    # ----------------------------
    for column in lag_columns:
        for lag in lag_periods:
            data[f"{column}_lag_{lag}"] = data[column].shift(lag)

    # ----------------------------
    # Rolling features
    # ----------------------------
    data["Freight_MA_7"] = (
        data["Freight_Rate_USD"].rolling(7).mean()
    )

    data["Freight_STD_7"] = (
        data["Freight_Rate_USD"].rolling(7).std()
    )

    data["BDI_MA_7"] = data["BDI_Index"].rolling(7).mean()
    data["Bunker_MA_7"] = data["Bunker_Fuel_USD"].rolling(7).mean()
    data["Coal_MA_7"] = data["Coal_Price_USD"].rolling(7).mean()

    data["Freight_MA_14"] = (
        data["Freight_Rate_USD"].rolling(14).mean()
    )

    data["Freight_STD_14"] = (
        data["Freight_Rate_USD"].rolling(14).std()
    )

    data["BDI_MA_14"] = data["BDI_Index"].rolling(14).mean()
    data["Bunker_MA_14"] = data["Bunker_Fuel_USD"].rolling(14).mean()
    data["Coal_MA_14"] = data["Coal_Price_USD"].rolling(14).mean()

    data["Freight_MA_30"] = (
        data["Freight_Rate_USD"].rolling(30).mean()
    )

    data["Freight_STD_30"] = (
        data["Freight_Rate_USD"].rolling(30).std()
    )

    data["BDI_MA_30"] = data["BDI_Index"].rolling(30).mean()
    data["Bunker_MA_30"] = data["Bunker_Fuel_USD"].rolling(30).mean()
    data["Coal_MA_30"] = data["Coal_Price_USD"].rolling(30).mean()

    # ----------------------------
    # Momentum / percentage change
    # ----------------------------
    data["Freight_Change_1D"] = (
        data["Freight_Rate_USD"].pct_change(1)
    )

    data["Freight_Change_7D"] = (
        data["Freight_Rate_USD"].pct_change(7)
    )

    data["Freight_Change_14D"] = (
        data["Freight_Rate_USD"].pct_change(14)
    )

    data["Freight_Change_30D"] = (
        data["Freight_Rate_USD"].pct_change(30)
    )

    data["BDI_Change_7D"] = data["BDI_Index"].pct_change(7)

    data["Bunker_Change_7D"] = (
        data["Bunker_Fuel_USD"].pct_change(7)
    )

    data["Coal_Change_7D"] = (
        data["Coal_Price_USD"].pct_change(7)
    )

    # ----------------------------
    # Calendar features
    # ----------------------------
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    data["Quarter"] = data["Date"].dt.quarter
    data["DayOfYear"] = data["Date"].dt.dayofyear

    data["Month_Sin"] = np.sin(
        2 * np.pi * data["Month"] / 12
    )

    data["Month_Cos"] = np.cos(
        2 * np.pi * data["Month"] / 12
    )

    return data


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def load_features(path):
    with open(path, "rb") as file:
        return pickle.load(file)


# ============================================================
# SAFE QUANTILE ORDERING
# ============================================================

def enforce_quantile_order(p10, p50, p90):
    """
    Ensures:
        P10 <= P50 <= P90
    """

    values = np.sort(
        np.array([p10, p50, p90], dtype=float)
    )

    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_horizon(
    latest_row: pd.DataFrame,
    horizon: str
):
    """
    Generates P10 / P50 / P90 forecast for one horizon.
    """

    if horizon == "7D":
        feature_file = FEATURE_FILES["7D"]

    else:
        feature_file = FEATURE_FILES["MULTI"]

    features = load_features(feature_file)

    if isinstance(features, dict):
        # Handle possible dictionary-based feature storage
        if horizon in features:
            feature_list = features[horizon]
        elif horizon.lower() in features:
            feature_list = features[horizon.lower()]
        elif "features" in features:
            feature_list = features["features"]
        else:
            feature_list = list(features.values())[0]
    else:
        feature_list = features

    # Make sure every trained feature exists
    missing_features = [
        feature
        for feature in feature_list
        if feature not in latest_row.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features for {horizon}: "
            f"{missing_features}"
        )

    X = latest_row[feature_list]

    predictions = {}

    for quantile in ["P10", "P50", "P90"]:

        model = load_model(
            MODEL_FILES[horizon][quantile]
        )

        prediction = model.predict(X)

        predictions[quantile] = float(
            prediction[0]
        )

    p10, p50, p90 = enforce_quantile_order(
        predictions["P10"],
        predictions["P50"],
        predictions["P90"],
    )

    return {
        "P10": round(p10, 2),
        "P50": round(p50, 2),
        "P90": round(p90, 2),
    }


# ============================================================
# MAIN FORECAST ENGINE
# ============================================================

def generate_forecast():

    print("=" * 60)
    print("FREIGHT FORECAST ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load RAW historical data
    # --------------------------------------------------------

    if not HISTORICAL_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Historical data not found:\n"
            f"{HISTORICAL_DATA_FILE}"
        )

    df = pd.read_csv(HISTORICAL_DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Latest RAW observation
    # --------------------------------------------------------

    latest_raw = df.iloc[-1]

    latest_date = latest_raw["Date"]
    current_freight = float(
        latest_raw["Freight_Rate_USD"]
    )

    print(f"Latest historical date : {latest_date.date()}")
    print(
        f"Current freight rate   : "
        f"${current_freight:.2f}/ton"
    )

    # --------------------------------------------------------
    # Build historical features
    # --------------------------------------------------------

    feature_data = build_latest_features(df)

    # Only latest row is used for prediction
    latest_row = feature_data.iloc[[-1]].copy()

    # Remove target columns if they somehow exist
    target_columns = [
        "Target_7D",
        "Target_15D",
        "Target_30D",
        "Direction_7D",
        "Direction_15D",
        "Direction_30D",
    ]

    latest_row = latest_row.drop(
        columns=[
            column
            for column in target_columns
            if column in latest_row.columns
        ],
        errors="ignore",
    )

    # --------------------------------------------------------
    # Generate forecasts
    # --------------------------------------------------------

    forecasts = {}

    for horizon in ["7D", "15D", "30D"]:

        result = predict_horizon(
            latest_row,
            horizon
        )

        p50 = result["P50"]

        if p50 > current_freight:
            direction = "UP"

        elif p50 < current_freight:
            direction = "DOWN"

        else:
            direction = "FLAT"

        forecasts[horizon] = {
            "forecast_date": (
                latest_date +
                pd.Timedelta(
                    days=int(
                        horizon.replace("D", "")
                    )
                )
            ).strftime("%Y-%m-%d"),

            "P10": result["P10"],
            "P50": result["P50"],
            "P90": result["P90"],

            "direction": direction,
        }

        print(
            f"{horizon:>3} | "
            f"P10 ${result['P10']:.2f} | "
            f"P50 ${result['P50']:.2f} | "
            f"P90 ${result['P90']:.2f} | "
            f"{direction}"
        )

    # --------------------------------------------------------
    # Final JSON
    # --------------------------------------------------------

    output = {
        "forecast_date": latest_date.strftime(
            "%Y-%m-%d"
        ),

        "current_freight_rate": round(
            current_freight,
            2
        ),

        "unit": "USD/ton",

        "horizons": forecasts,

        "metadata": {
            "source": "historical_freight_data.csv",
            "latest_observation_used": latest_date.strftime(
                "%Y-%m-%d"
            ),
            "model_type": "LightGBM Quantile Regression",
            "quantiles": [
                "P10",
                "P50",
                "P90",
            ],
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print("=" * 60)
    print(
        f"Forecast saved to:\n{OUTPUT_FILE}"
    )
    print("=" * 60)

    return output


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    generate_forecast()