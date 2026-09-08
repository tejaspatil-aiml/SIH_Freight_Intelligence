import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "multi_horizon_freight_data.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 3. FEATURES
# ============================================================

excluded_columns = [
    "Date",
    "Target_15D",
    "Target_30D",
    "Direction_15D",
    "Direction_30D"
]

feature_columns = [
    column for column in df.columns
    if column not in excluded_columns
]

X = df[feature_columns]


# ============================================================
# 4. CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end]
X_validation = X.iloc[train_end:validation_end]
X_test = X.iloc[validation_end:]


print("=" * 70)
print("MULTI-HORIZON LIGHTGBM FORECASTING")
print("=" * 70)

print(f"Total observations : {n}")
print(f"Training records   : {len(X_train)}")
print(f"Validation records : {len(X_validation)}")
print(f"Testing records    : {len(X_test)}")

print("\nDate ranges:")

print(
    f"Train      : {df['Date'].iloc[0].date()} "
    f"→ {df['Date'].iloc[train_end - 1].date()}"
)

print(
    f"Validation : {df['Date'].iloc[train_end].date()} "
    f"→ {df['Date'].iloc[validation_end - 1].date()}"
)

print(
    f"Test       : {df['Date'].iloc[validation_end].date()} "
    f"→ {df['Date'].iloc[-1].date()}"
)


# ============================================================
# 5. MODEL TRAINING FUNCTION
# ============================================================

def train_quantile_model(X_train, y_train, X_validation, y_validation, alpha):

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
        eval_set=[(X_validation, y_validation)],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50,
                verbose=False
            )
        ]
    )

    return model


# ============================================================
# 6. TRAIN ONE HORIZON
# ============================================================

def train_horizon(horizon):

    target = f"Target_{horizon}D"

    print("\n" + "-" * 70)
    print(f"TRAINING {horizon}-DAY FORECAST")
    print("-" * 70)

    y = df[target]

    y_train = y.iloc[:train_end]
    y_validation = y.iloc[train_end:validation_end]
    y_test = y.iloc[validation_end:]

    print("Training P10...")
    model_p10 = train_quantile_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        0.10
    )

    print("Training P50...")
    model_p50 = train_quantile_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        0.50
    )

    print("Training P90...")
    model_p90 = train_quantile_model(
        X_train,
        y_train,
        X_validation,
        y_validation,
        0.90
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    pred_p10 = model_p10.predict(X_test)
    pred_p50 = model_p50.predict(X_test)
    pred_p90 = model_p90.predict(X_test)

    # Guarantee logical quantile ordering
    pred_p10 = np.minimum(pred_p10, pred_p50)
    pred_p90 = np.maximum(pred_p90, pred_p50)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    current_freight = X_test["Freight_Rate_USD"].values

    actual_direction = np.sign(
        y_test.values - current_freight
    )

    predicted_direction = np.sign(
        pred_p50 - current_freight
    )

    directional_accuracy = (
        np.mean(
            actual_direction == predicted_direction
        ) * 100
    )

    coverage = (
        np.mean(
            (y_test.values >= pred_p10) &
            (y_test.values <= pred_p90)
        ) * 100
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nPERFORMANCE")

    print(f"P50 MAE              : {mae:.4f}")
    print(f"P50 RMSE             : {rmse:.4f}")
    print(
        f"Directional Accuracy : "
        f"{directional_accuracy:.2f}%"
    )
    print(
        f"P10-P90 Coverage     : "
        f"{coverage:.2f}%"
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    joblib.dump(
        model_p10,
        MODEL_DIR / f"freight_{horizon}d_p10.pkl"
    )

    joblib.dump(
        model_p50,
        MODEL_DIR / f"freight_{horizon}d_p50.pkl"
    )

    joblib.dump(
        model_p90,
        MODEL_DIR / f"freight_{horizon}d_p90.pkl"
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    results = pd.DataFrame({
        "Date": df["Date"].iloc[validation_end:].values,
        "Current_Freight": current_freight,
        "Actual": y_test.values,
        "P10": pred_p10,
        "P50": pred_p50,
        "P90": pred_p90
    })

    results["Signal"] = np.where(
        results["P50"] > results["Current_Freight"],
        "UP",
        "DOWN"
    )

    results.to_csv(
        BASE_DIR /
        "data" /
        f"forecast_{horizon}d_test_results.csv",
        index=False
    )

    return {
        "Horizon": f"{horizon}D",
        "MAE": mae,
        "RMSE": rmse,
        "Directional_Accuracy": directional_accuracy,
        "Coverage": coverage
    }


# ============================================================
# 7. TRAIN BOTH HORIZONS
# ============================================================

results_15d = train_horizon(15)

results_30d = train_horizon(30)


# ============================================================
# 8. COMPARISON
# ============================================================

comparison = pd.DataFrame([
    results_15d,
    results_30d
])

print("\n" + "=" * 70)
print("15D vs 30D MODEL COMPARISON")
print("=" * 70)

print(
    comparison.round(2).to_string(index=False)
)


# ============================================================
# 9. SAVE FEATURE LIST
# ============================================================

joblib.dump(
    feature_columns,
    MODEL_DIR / "multi_horizon_features.pkl"
)


print("\n" + "=" * 70)
print("MULTI-HORIZON TRAINING COMPLETE")
print("=" * 70)

print("Models saved successfully.")
print(f"Model directory: {MODEL_DIR}")