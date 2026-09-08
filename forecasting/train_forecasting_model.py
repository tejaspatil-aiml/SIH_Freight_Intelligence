# ================================================================
# SIH FREIGHT INTELLIGENCE
# LIGHTGBM 7-DAY FREIGHT FORECASTING MODEL
# ================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ================================================================
# 1. PROJECT PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2. LOAD ENGINEERED DATA
# ================================================================

DATA_FILE = DATA_DIR / "engineered_freight_data.csv"

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


print("=" * 70)
print("LIGHTGBM 7-DAY FREIGHT FORECASTING MODEL")
print("=" * 70)


# ================================================================
# 3. TARGET AND FEATURES
# ================================================================

TARGET = "Target_7D"

EXCLUDED_COLUMNS = [
    "Date",
    "Target_7D",
    "Direction_7D"
]


feature_columns = [
    column
    for column in df.columns
    if column not in EXCLUDED_COLUMNS
]


X = df[feature_columns]
y = df[TARGET]


print(f"Total observations : {len(df)}")
print(f"Total features     : {len(feature_columns)}")


# ================================================================
# 4. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ================================================================

total_records = len(df)

train_end = int(total_records * 0.70)
validation_end = int(total_records * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_validation = X.iloc[train_end:validation_end]
y_validation = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]


print(f"Training records   : {len(X_train)}")
print(f"Validation records : {len(X_validation)}")
print(f"Testing records    : {len(X_test)}")

print()

print("Date ranges:")

print(
    f"Train      : "
    f"{df['Date'].iloc[0].date()} → "
    f"{df['Date'].iloc[train_end - 1].date()}"
)

print(
    f"Validation : "
    f"{df['Date'].iloc[train_end].date()} → "
    f"{df['Date'].iloc[validation_end - 1].date()}"
)

print(
    f"Test       : "
    f"{df['Date'].iloc[validation_end].date()} → "
    f"{df['Date'].iloc[-1].date()}"
)

print()

print(f"Total features     : {len(feature_columns)}")


# ================================================================
# 5. LIGHTGBM QUANTILE MODEL FUNCTION
# ================================================================

def train_quantile_model(alpha):

    model = lgb.LGBMRegressor(

        objective="quantile",

        alpha=alpha,

        n_estimators=500,

        learning_rate=0.03,

        num_leaves=31,

        max_depth=-1,

        min_child_samples=20,

        subsample=0.85,

        colsample_bytree=0.85,

        reg_alpha=0.1,

        reg_lambda=0.2,

        random_state=42,

        verbosity=-1
    )

    model.fit(

        X_train,

        y_train,

        eval_set=[
            (X_validation, y_validation)
        ],

        eval_names=[
            "validation"
        ],

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False
            )
        ]
    )

    return model


# ================================================================
# 6. TRAIN P10 MODEL
# ================================================================

print("Training P10 model...")

model_p10 = train_quantile_model(0.10)


# ================================================================
# 7. TRAIN P50 MODEL
# ================================================================

print("Training P50 model...")

model_p50 = train_quantile_model(0.50)


# ================================================================
# 8. TRAIN P90 MODEL
# ================================================================

print("Training P90 model...")

model_p90 = train_quantile_model(0.90)


# ================================================================
# 9. PREDICTIONS ON TEST DATA
# ================================================================

pred_p10 = model_p10.predict(X_test)

pred_p50 = model_p50.predict(X_test)

pred_p90 = model_p90.predict(X_test)


# ================================================================
# 10. QUANTILE ORDERING
# ================================================================
# Ensures:
#
# P10 <= P50 <= P90
#
# This prevents impossible prediction intervals.


pred_p10 = np.minimum(pred_p10, pred_p50)

pred_p90 = np.maximum(pred_p90, pred_p50)


# ================================================================
# 11. MODEL PERFORMANCE
# ================================================================

mae = mean_absolute_error(
    y_test,
    pred_p50
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        pred_p50
    )
)


# ================================================================
# 12. DIRECTIONAL ACCURACY
# ================================================================

current_freight = df["Freight_Rate_USD"].iloc[
    validation_end:
].values


actual_direction = np.sign(
    y_test.values - current_freight
)

predicted_direction = np.sign(
    pred_p50 - current_freight
)


directional_accuracy = (
    actual_direction == predicted_direction
).mean() * 100


# ================================================================
# 13. P10-P90 COVERAGE
# ================================================================

coverage = (
    (y_test.values >= pred_p10) &
    (y_test.values <= pred_p90)
).mean() * 100


# ================================================================
# 14. PRINT RESULTS
# ================================================================

print()

print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"P50 MAE              : {mae:.4f}"
)

print(
    f"P50 RMSE             : {rmse:.4f}"
)

print(
    f"Directional Accuracy : {directional_accuracy:.2f}%"
)

print(
    f"P10-P90 Coverage     : {coverage:.2f}%"
)

print("=" * 70)


# ================================================================
# 15. SAMPLE FORECASTS
# ================================================================

print()
print("=" * 70)
print("SAMPLE FORECASTS")
print("=" * 70)

sample_count = min(10, len(X_test))

sample_dates = df["Date"].iloc[
    validation_end:
].iloc[:sample_count]


for i in range(sample_count):

    print(
        f"{sample_dates.iloc[i].date()} | "
        f"P10: ${pred_p10[i]:.2f} | "
        f"P50: ${pred_p50[i]:.2f} | "
        f"P90: ${pred_p90[i]:.2f}"
    )


# ================================================================
# 16. SAVE MODELS
# ================================================================

joblib.dump(
    model_p10,
    MODEL_DIR / "freight_model_p10.pkl"
)

joblib.dump(
    model_p50,
    MODEL_DIR / "freight_model_p50.pkl"
)

joblib.dump(
    model_p90,
    MODEL_DIR / "freight_model_p90.pkl"
)


# ================================================================
# 17. SAVE FEATURE LIST
# ================================================================

joblib.dump(
    feature_columns,
    MODEL_DIR / "forecast_features.pkl"
)


# ================================================================
# 18. SAVE TEST RESULTS
# ================================================================

test_results = pd.DataFrame({

    "Date": df["Date"].iloc[
        validation_end:
    ].values,

    "Actual_Freight": y_test.values,

    "P10_Forecast": pred_p10,

    "P50_Forecast": pred_p50,

    "P90_Forecast": pred_p90,

    "Current_Freight": current_freight

})


test_results["Actual_Direction"] = np.where(
    test_results["Actual_Freight"]
    > test_results["Current_Freight"],
    "UP",
    "DOWN"
)


test_results["Predicted_Direction"] = np.where(
    test_results["P50_Forecast"]
    > test_results["Current_Freight"],
    "UP",
    "DOWN"
)


test_results.to_csv(
    DATA_DIR / "forecast_test_results.csv",
    index=False
)


# ================================================================
# 19. FINAL STATUS
# ================================================================

print()

print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    "✓ models/freight_model_p10.pkl"
)

print(
    "✓ models/freight_model_p50.pkl"
)

print(
    "✓ models/freight_model_p90.pkl"
)

print(
    "✓ models/forecast_features.pkl"
)

print(
    "✓ data/forecast_test_results.csv"
)

print()

print("=" * 70)
print("7-DAY FREIGHT FORECASTING MODEL COMPLETED")
print("=" * 70)