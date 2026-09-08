import json
from pathlib import Path

import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Freight Intelligence Command Center",
    page_icon="🚢",
    layout="wide"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

FORECAST_FILE = BASE_DIR / "outputs" / "forecast_output.json"
OPTIMIZATION_FILE = BASE_DIR / "outputs" / "vessel_optimization_output.json"


# =========================================================
# OPTIMIZER IMPORT
# =========================================================

try:
    from optimizer.vessel_optimizer import run_vessel_optimizer

    OPTIMIZER_AVAILABLE = True
    OPTIMIZER_ERROR = None

except Exception as error:
    OPTIMIZER_AVAILABLE = False
    OPTIMIZER_ERROR = str(error)


# =========================================================
# JSON LOADER
# =========================================================

def load_json(file_path):
    if not file_path.exists():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception as error:
        st.error(f"Unable to read {file_path.name}: {error}")
        return None


# =========================================================
# SAFE VALUE HELPERS
# =========================================================

def get_value(data, *keys, default=0):
    if not isinstance(data, dict):
        return default

    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return default


def get_float(data, *keys, default=0):
    try:
        return float(get_value(data, *keys, default=default))
    except (TypeError, ValueError):
        return float(default)


def get_int(data, *keys, default=0):
    try:
        return int(float(get_value(data, *keys, default=default)))
    except (TypeError, ValueError):
        return int(default)


# =========================================================
# LOAD OUTPUTS
# =========================================================

forecast = load_json(FORECAST_FILE)
default_optimization = load_json(OPTIMIZATION_FILE)


# =========================================================
# SESSION STATE
# =========================================================

if "optimization_result" not in st.session_state:
    st.session_state.optimization_result = default_optimization

if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = None


# =========================================================
# HEADER
# =========================================================

st.title("🚢 Freight Intelligence Command Center")

st.caption(
    "AI-powered Freight Forecasting & Vessel Chartering "
    "Decision Support"
)

st.divider()


# =========================================================
# FORECAST VALIDATION
# =========================================================

if forecast is None:
    st.error(
        "Forecast output not found. "
        "Run the forecasting engine first."
    )
    st.stop()


# =========================================================
# CURRENT MARKET INTELLIGENCE
# =========================================================

st.subheader("📊 Current Market Intelligence")

# Supports the corrected forecast engine output:
# "current_freight_rate"
current_freight = get_float(
    forecast,
    "current_freight_rate",
    "current_freight_usd_per_ton",
    "current_freight"
)

forecast_date = forecast.get("forecast_date", "N/A")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Freight",
        f"${current_freight:.2f}/ton"
    )

with col2:
    st.metric(
        "Forecast Date",
        forecast_date
    )

with col3:
    st.metric(
        "Forecast Horizons",
        "7D • 15D • 30D"
    )

st.divider()


# =========================================================
# FORECAST SECTION
# =========================================================

st.subheader("🔮 Probabilistic Freight Forecast")

horizons = forecast.get("horizons", {})

forecast_columns = st.columns(3)

for column, horizon_name, title in zip(
    forecast_columns,
    ["7D", "15D", "30D"],
    ["📅 7 Days", "📅 15 Days", "📅 30 Days"]
):

    with column:

        data = horizons.get(horizon_name, {})

        st.markdown(f"### {title}")

        p50 = get_float(data, "P50", "p50")
        p10 = get_float(data, "P10", "p10")
        p90 = get_float(data, "P90", "p90")

        direction = str(
            get_value(
                data,
                "Direction",
                "direction",
                default="FLAT"
            )
        ).upper()

        st.metric(
            "P50 Forecast",
            f"${p50:.2f}/ton",
            delta=f"{p50 - current_freight:+.2f}"
        )

        st.write(f"**P10:** ${p10:.2f}/ton")
        st.write(f"**P90:** ${p90:.2f}/ton")

        if direction == "UP":
            st.warning("📈 Expected Direction: UP")
        elif direction == "DOWN":
            st.success("📉 Expected Direction: DOWN")
        else:
            st.info("➡️ Expected Direction: FLAT")


st.divider()


# =========================================================
# DECISION CONTROL CENTER
# =========================================================

st.subheader("🎯 Decision Control Center")

st.caption(
    "Run the MILP optimizer under different operational "
    "and market conditions."
)


# =========================================================
# CONTROLS
# =========================================================

control1, control2 = st.columns(2)

with control1:

    st.markdown("### 📦 Cargo Requirement")

    cargo_demand = st.number_input(
        "Cargo Demand (tons)",
        min_value=10000,
        max_value=180000,
        value=150000,
        step=10000
    )

    st.markdown("### 📈 Freight Market Shock")

    freight_shock = st.slider(
        "Freight Shock (%)",
        min_value=-30,
        max_value=50,
        value=0,
        step=5
    )


with control2:

    st.markdown("### ⏱️ Port Delay")

    port_delay = st.slider(
        "Additional Port Delay (days)",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    st.markdown("### ⚓ Vessel Availability")

    vessel_availability = st.slider(
        "Available Fleet Capacity (%)",
        min_value=40,
        max_value=100,
        value=100,
        step=10
    )


st.markdown("")

run_decision = st.button(
    "🚀 RUN LIVE MILP DECISION",
    use_container_width=True,
    type="primary"
)


# =========================================================
# RUN LIVE OPTIMIZATION
# =========================================================

if run_decision:

    if not OPTIMIZER_AVAILABLE:

        st.error("MILP optimizer could not be imported.")
        st.code(OPTIMIZER_ERROR)

    else:

        try:

            with st.spinner(
                "Running forecasting-aware MILP optimization..."
            ):

                result = run_vessel_optimizer(
                    cargo_demand=float(cargo_demand),
                    freight_shock_pct=float(freight_shock),
                    port_delay_days=float(port_delay),
                    vessel_availability_pct=float(
                        vessel_availability
                    )
                )

            st.session_state.optimization_result = result
            st.session_state.scenario_result = result

            st.success(
                "✅ Decision successfully generated "
                "by the MILP optimizer."
            )

        except Exception as error:

            st.error(f"❌ Optimization failed: {error}")


# =========================================================
# CURRENT RESULT
# =========================================================

optimization = st.session_state.optimization_result

if optimization is None:

    st.warning("Run the optimizer to generate a decision.")
    st.stop()


# =========================================================
# RESULT VALUES
# =========================================================

recommendation = str(
    get_value(
        optimization,
        "recommendation",
        default="NO DECISION"
    )
)

selected_window = str(
    get_value(
        optimization,
        "selected_procurement_window",
        "selected_window",
        default="N/A"
    )
)

selected_freight = get_float(
    optimization,
    "forecast_freight_usd_per_ton",
    "selected_freight_usd_per_ton"
)

base_selected_freight = get_float(
    optimization,
    "base_freight_usd_per_ton",
    default=selected_freight
)

result_shock = get_float(
    optimization,
    "freight_shock_pct",
    default=0
)

result_delay = get_float(
    optimization,
    "port_delay_days",
    default=0
)

result_availability = get_float(
    optimization,
    "vessel_availability_pct",
    default=100
)

total_cargo = get_float(
    optimization,
    "total_cargo_tons",
    "cargo_demand"
)

vessel_count = get_int(
    optimization,
    "vessel_count",
    default=0
)

charter_cost = get_float(
    optimization,
    "total_charter_cost_usd",
    "charter_cost_usd"
)

average_landed = get_float(
    optimization,
    "average_landed_cost_usd_per_ton",
    "average_landed_cost"
)

total_landed = get_float(
    optimization,
    "total_landed_cost_usd",
    "total_landed_cost"
)


# =========================================================
# PROCUREMENT COMPARISON DATA
# =========================================================

comparison_data = optimization.get("procurement_window_comparison", [])

now_data = (
    comparison_data.get("NOW", {})
    if isinstance(comparison_data, dict)
    else {}
)

now_total_landed = get_float(
    now_data,
    "total_landed_cost_usd",
    "total_landed_cost",
    "Total_Landed_Cost",
    "total_cost",
    default=0
)

modeled_advantage_vs_now = (
    now_total_landed - total_landed
)


st.divider()


# =========================================================
# RECOMMENDATION
# =========================================================

st.subheader("⚓ Vessel Chartering Recommendation")

if "BOOK" in recommendation.upper():

    st.success(
        f"🟢 **OPTIMIZER RECOMMENDATION: {recommendation}**"
    )

elif "WAIT" in recommendation.upper():

    st.warning(
        f"🟡 **OPTIMIZER RECOMMENDATION: {recommendation}**"
    )

else:

    st.info(
        f"🔵 **OPTIMIZER RECOMMENDATION: {recommendation}**"
    )


# =========================================================
# DECISION METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Selected Window", selected_window)

with col2:
    st.metric(
        "Scenario Freight",
        f"${selected_freight:.2f}/ton"
    )

with col3:
    st.metric(
        "Cargo Volume",
        f"{total_cargo:,.0f} tons"
    )

with col4:
    st.metric(
        "Vessels Selected",
        vessel_count
    )


# =========================================================
# SCENARIO CONDITIONS
# =========================================================

st.markdown("### 🧪 Active Scenario Conditions")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Freight Shock",
        f"{result_shock:+.0f}%"
    )

with col2:
    st.metric(
        "Port Delay",
        f"{result_delay:.0f} days"
    )

with col3:
    st.metric(
        "Fleet Availability",
        f"{result_availability:.0f}%"
    )


# =========================================================
# LANDED COST
# =========================================================

st.markdown("### 💰 Landed Cost Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Charter Cost",
        f"${charter_cost:,.0f}"
    )

with col2:
    st.metric(
        "Average Landed Cost",
        f"${average_landed:.2f}/ton"
    )

with col3:
    st.metric(
        "Total Landed Cost",
        f"${total_landed:,.0f}"
    )


# =========================================================
# PROCUREMENT WINDOW COMPARISON
# =========================================================

st.markdown("### 📅 Procurement Window Comparison")

if isinstance(comparison_data, list) and comparison_data:

    rows = []

    for row in comparison_data:

        if not isinstance(row, dict):
            continue

        window = str(get_value(row, "window", "Window", default="N/A"))

        freight = get_float(
            row,
            "freight_usd_per_ton",
            "forecast_freight_usd_per_ton",
            "base_freight_usd_per_ton",
            "freight",
            default=0
        )

        landed = get_float(
            row,
            "average_landed_cost_usd_per_ton",
            "average_landed_cost",
            "landed_cost_usd_per_ton",
            "landed_cost",
            default=0
        )

        total = get_float(
            row,
            "total_landed_cost_usd",
            "total_landed_cost",
            "Total_Landed_Cost",
            "total_cost",
            default=0
        )

        rows.append({
            "Window": window,
            "Freight ($/t)": f"${freight:.2f}",
            "Landed Cost ($/t)": f"${landed:.2f}",
            "Total Landed Cost": f"${total:,.0f}"
        })

    if rows:
        st.table(rows)

else:

    st.info(
        "Procurement comparison is unavailable in the saved optimizer output."
    )


# =========================================================
# VESSEL ALLOCATION
# =========================================================

st.markdown("### 🚢 Recommended Vessel Allocation")

allocations = optimization.get("vessel_allocations", [])

if not allocations:

    st.info(
        "No vessel allocation returned by the optimizer."
    )

else:

    for vessel in allocations:

        vessel_id = str(
            get_value(
                vessel,
                "vessel_id",
                "Vessel_ID",
                "id",
                default="UNKNOWN"
            )
        )

        vessel_type = str(
            get_value(
                vessel,
                "vessel_type",
                "Vessel_Type",
                "type",
                default="Unknown"
            )
        )

        origin = str(
            get_value(
                vessel,
                "origin",
                "Origin",
                default="Unknown"
            )
        )

        destination = str(
            get_value(
                vessel,
                "destination",
                "Destination",
                default="Unknown"
            )
        )

        quantity_tons = get_float(
            vessel,
            "quantity_tons",
            "quantity",
            "cargo_tons",
            "allocated_tons",
            default=0
        )

        capacity_tons = get_float(
            vessel,
            "capacity_tons",
            "Capacity_Tons",
            "capacity",
            default=0
        )

        utilization_percent = get_float(
            vessel,
            "utilization_percent",
            "utilization",
            "Utilization",
            default=(
                quantity_tons / capacity_tons * 100
                if capacity_tons > 0
                else 0
            )
        )

        landed_cost_per_ton = get_float(
            vessel,
            "landed_cost_usd_per_ton",
            "landed_cost",
            "cost_per_ton",
            default=0
        )

        with st.container():

            st.markdown(
                f"#### 🚢 {vessel_id} — {vessel_type}"
            )

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.write("**Route**")
                st.write(f"{origin} → {destination}")

            with col2:
                st.write("**Cargo**")
                st.write(f"{quantity_tons:,.0f} tons")

            with col3:
                st.write("**Capacity**")
                st.write(f"{capacity_tons:,.0f} tons")

            with col4:
                st.write("**Utilization**")
                st.write(f"{utilization_percent:.2f}%")

            with col5:
                st.write("**Landed Cost**")
                st.write(f"${landed_cost_per_ton:.2f}/ton")

            st.divider()


# =========================================================
# DECISION EXPLANATION
# =========================================================

st.subheader("💡 Why This Decision?")

st.caption(
    "The optimizer does not choose a window from freight rate alone. "
    "It compares complete modeled landed cost, including cargo cost, "
    "freight, chartering and operational delay assumptions."
)

if "WAIT" in recommendation.upper():

    if modeled_advantage_vs_now > 0:

        st.success(
            f"### 🟢 {recommendation} — "
            f"the selected window has a lower modeled total landed "
            f"cost than booking NOW."
        )

    else:

        st.info(
            f"### 🔵 {recommendation} — "
            f"the optimizer selected the lowest feasible "
            f"landed-cost window under the active scenario."
        )

elif "BOOK" in recommendation.upper():

    st.warning(
        f"### 🟠 {recommendation} — "
        f"booking NOW minimizes the modeled total landed cost."
    )

else:

    st.info(f"### 🔵 {recommendation}")


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Freight",
        f"${current_freight:.2f}/ton"
    )

with col2:
    st.metric(
        "Selected Freight",
        f"${selected_freight:.2f}/ton",
        delta=f"{selected_freight - current_freight:+.2f}"
    )

with col3:

    if modeled_advantage_vs_now > 0:

        st.metric(
            "Modeled Cost Advantage vs NOW",
            f"${modeled_advantage_vs_now:,.0f}"
        )

    elif modeled_advantage_vs_now < 0:

        st.metric(
            "Additional Cost vs NOW",
            f"${abs(modeled_advantage_vs_now):,.0f}"
        )

    else:

        st.metric(
            "Cost Difference vs NOW",
            "$0"
        )


# =========================================================
# TOTAL COST ADVANTAGE
# =========================================================

difference_vs_now = get_float(
    optimization,
    "difference_vs_now_usd",
    "cost_difference_vs_now_usd",
    default=0
)

if difference_vs_now != 0:

    if difference_vs_now < 0:

        st.success(
            f"💰 **Estimated Optimization Advantage vs NOW:** "
            f"${abs(difference_vs_now):,.0f}"
        )

    else:

        st.warning(
            f"⚠️ **Additional modeled cost vs NOW:** "
            f"${difference_vs_now:,.0f}"
        )


# =========================================================
# WHAT-IF EXPLANATION
# =========================================================

if (
    result_shock != 0
    or result_delay != 0
    or result_availability != 100
):

    st.info(
        f"**Scenario reasoning:** The optimizer evaluated a "
        f"{result_shock:+.0f}% freight shock, "
        f"{result_delay:.0f}-day additional port delay, and "
        f"{result_availability:.0f}% available fleet capacity. "
        f"The resulting decision is **{recommendation}** "
        f"for the {selected_window} procurement window."
    )


# =========================================================
# DECISION LOGIC
# =========================================================

with st.expander("🧠 View Decision Logic"):

    st.write(
        "The system combines probabilistic freight forecasting "
        "with a Mixed Integer Linear Programming optimizer."
    )

    st.write("1. Forecast future freight using P10/P50/P90.")
    st.write("2. Apply market shock assumptions.")
    st.write("3. Add operational delay costs.")
    st.write("4. Restrict available vessel capacity.")
    st.write("5. Evaluate NOW, 7D, 15D and 30D procurement windows.")
    st.write("6. Allocate cargo to compatible vessels.")
    st.write("7. Minimize total landed cost.")
    st.write("8. Return an actionable Book/Wait decision.")


st.divider()


# =========================================================
# DECISION INTELLIGENCE PIPELINE
# =========================================================

st.subheader("🧠 Decision Intelligence Pipeline")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("**1. DATA**\n\nMarket • Fuel • Coal • Port")

with col2:
    st.info("**2. FORECAST**\n\nLightGBM • P10/P50/P90")

with col3:
    st.info("**3. OPTIMIZE**\n\nMILP • Vessel Allocation")

with col4:
    st.info("**4. DECIDE**\n\nBook • Wait • Compare")


st.divider()


# =========================================================
# INTELLIGENCE LAYER
# =========================================================

st.subheader("🤖 Intelligence Layer")

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "**Forecasting Model**\n\n"
        "LightGBM Quantile Models"
    )

with col2:
    st.info(
        "**Forecast Type**\n\n"
        "P10 / P50 / P90"
    )

with col3:
    st.info(
        "**Optimization Engine**\n\n"
        "Mixed Integer Linear Programming"
    )


st.divider()


# =========================================================
# SYSTEM STATUS
# =========================================================

st.subheader("🟢 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("Forecast Engine: ONLINE")

with col2:

    if OPTIMIZER_AVAILABLE:
        st.success("MILP Optimizer: ONLINE")
    else:
        st.error("MILP Optimizer: ERROR")

with col3:
    st.success("Decision Dashboard: ONLINE")


# =========================================================
# DISCLAIMER
# =========================================================

st.caption(
    "Prototype note: vessel, charter and operational inputs are "
    "prototype/simulated inputs for demonstration. P10/P50/P90 "
    "are forecast quantiles and are not guaranteed confidence "
    "intervals. Port-delay cost uses a prototype linear delay-cost "
    "assumption. Production deployment requires validated commercial "
    "charter rates, port constraints, contracts and operational data."
)
