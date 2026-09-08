from pathlib import Path
import json
import pandas as pd
import pulp


# ================================================================
# PATH CONFIGURATION
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

CARGO_FILE = DATA_DIR / "optimizer_data.csv"
VESSEL_FILE = DATA_DIR / "vessel_data.csv"
FORECAST_FILE = OUTPUT_DIR / "forecast_output.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def load_forecast():
    """
    Load forecast_output.json.
    """

    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            f"Forecast output not found: {FORECAST_FILE}"
        )

    with open(
        FORECAST_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def get_forecast_value(forecast_data, horizon):
    """
    Extract P50 freight forecast.

    Supports the current structure:

    {
        "horizons": {
            "7D": {
                "P10": ...,
                "P50": ...,
                "P90": ...
            }
        }
    }

    Also supports a few alternative structures for robustness.
    """

    horizon = str(horizon).upper()

    # ------------------------------------------------------------
    # CURRENT PROJECT STRUCTURE
    # ------------------------------------------------------------

    if "horizons" in forecast_data:

        horizons = forecast_data["horizons"]

        if horizon in horizons:

            value = horizons[horizon]

            if isinstance(value, dict):

                for key in [
                    "P50",
                    "p50",
                    "p50_forecast"
                ]:

                    if key in value:
                        return float(value[key])

    # ------------------------------------------------------------
    # DIRECT STRUCTURE
    # ------------------------------------------------------------

    if horizon in forecast_data:

        value = forecast_data[horizon]

        if isinstance(value, dict):

            for key in [
                "P50",
                "p50",
                "p50_forecast"
            ]:

                if key in value:
                    return float(value[key])

    # ------------------------------------------------------------
    # ALTERNATIVE "forecasts" STRUCTURE
    # ------------------------------------------------------------

    if "forecasts" in forecast_data:

        forecasts = forecast_data["forecasts"]

        if horizon in forecasts:

            value = forecasts[horizon]

            if isinstance(value, dict):

                for key in [
                    "P50",
                    "p50",
                    "p50_forecast"
                ]:

                    if key in value:
                        return float(value[key])

    raise KeyError(
        f"Could not find P50 forecast for {horizon}"
    )


def get_current_freight(forecast_data):
    """
    Extract current freight rate.
    """

    possible_keys = [
        "current_freight_usd_per_ton",
        "current_freight_rate",
        "current_freight",
        "Current_Freight",
        "currentFreight"
    ]

    # ------------------------------------------------------------
    # ROOT LEVEL
    # ------------------------------------------------------------

    for key in possible_keys:

        if key in forecast_data:

            return float(
                forecast_data[key]
            )

    # ------------------------------------------------------------
    # MARKET LEVEL
    # ------------------------------------------------------------

    if "market" in forecast_data:

        market = forecast_data["market"]

        if isinstance(market, dict):

            for key in possible_keys:

                if key in market:

                    return float(
                        market[key]
                    )

    raise KeyError(
        "Current freight value not found "
        "in forecast_output.json"
    )


# ================================================================
# SINGLE WINDOW MILP OPTIMIZER
# ================================================================

def optimize_single_window(
    cargo_df,
    vessel_df,
    window_name,
    freight_rate,
    port_delay_days=0.0,
    vessel_availability_pct=100.0
):
    """
    Solve the MILP for one procurement window.

    Objective:

        FOB Cost
      + Port Cost
      + Base Demurrage
      + Additional Delay Cost
      + Freight Cost
      + Charter Cost

    Constraints:

      1. Total cargo demand satisfaction
      2. Vessel capacity
      3. Route compatibility
      4. Fleet availability
    """

    # ------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------

    if freight_rate < 0:
        raise ValueError(
            "freight_rate cannot be negative."
        )

    if port_delay_days < 0:
        raise ValueError(
            "port_delay_days cannot be negative."
        )

    if not 0 <= vessel_availability_pct <= 100:
        raise ValueError(
            "vessel_availability_pct must be between 0 and 100."
        )

    # ------------------------------------------------------------
    # Total fleet capacity
    # ------------------------------------------------------------

    total_fleet_capacity = float(
        vessel_df["Capacity_Tons"].sum()
    )

    available_capacity = (
        total_fleet_capacity
        * vessel_availability_pct
        / 100.0
    )

    # ------------------------------------------------------------
    # MILP MODEL
    # ------------------------------------------------------------

    model = pulp.LpProblem(
        f"Freight_Optimizer_{window_name}",
        pulp.LpMinimize
    )

    # ------------------------------------------------------------
    # Decision Variables
    # ------------------------------------------------------------

    vessel_selected = {

        row.Vessel_ID:
        pulp.LpVariable(
            f"select_{row.Vessel_ID}",
            cat="Binary"
        )

        for row in vessel_df.itertuples()
    }

    cargo_quantity = {

        row.Vessel_ID:
        pulp.LpVariable(
            f"cargo_{row.Vessel_ID}",
            lowBound=0,
            cat="Continuous"
        )

        for row in vessel_df.itertuples()
    }

    # ------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------

    objective_terms = []

    for vessel in vessel_df.itertuples():

        vessel_id = vessel.Vessel_ID

        # --------------------------------------------------------
        # Prototype charter cost
        # --------------------------------------------------------

        charter_cost = (
            float(vessel.Daily_Charter_USD)
            * float(vessel.Available_Days)
        )

        # --------------------------------------------------------
        # Find matching cargo route
        # --------------------------------------------------------

        matching_cargo = cargo_df[
            (cargo_df["Origin"] == vessel.Origin)
            &
            (cargo_df["Destination"] == vessel.Destination)
        ]

        if matching_cargo.empty:
            continue

        cargo_row = matching_cargo.iloc[0]

        fob_cost = float(
            cargo_row["FOB_USD_per_Ton"]
        )

        port_cost = float(
            cargo_row["Port_Cost_USD_per_Ton"]
        )

        base_demurrage = float(
            cargo_row["Demurrage_USD_per_Ton"]
        )

        # --------------------------------------------------------
        # Prototype additional delay cost
        # --------------------------------------------------------

        additional_delay_cost = (
            base_demurrage
            * port_delay_days
        )

        landed_cost_per_ton = (
            fob_cost
            + port_cost
            + base_demurrage
            + additional_delay_cost
            + freight_rate
        )

        objective_terms.append(

            charter_cost
            * vessel_selected[vessel_id]

            +

            landed_cost_per_ton
            * cargo_quantity[vessel_id]
        )

    model += pulp.lpSum(objective_terms)

    # ------------------------------------------------------------
    # Total Demand Constraint
    # ------------------------------------------------------------

    total_demand = float(
        cargo_df["Demand_Tons"].sum()
    )

    model += (

        pulp.lpSum(
            cargo_quantity.values()
        )

        == total_demand

    ), "Total_Cargo_Demand"

    # ------------------------------------------------------------
    # Vessel Capacity Constraints
    # ------------------------------------------------------------

    for vessel in vessel_df.itertuples():

        vessel_id = vessel.Vessel_ID

        model += (

            cargo_quantity[vessel_id]

            <=

            float(vessel.Capacity_Tons)
            * vessel_selected[vessel_id]

        ), f"Capacity_{vessel_id}"

    # ------------------------------------------------------------
    # Fleet Availability Constraint
    # ------------------------------------------------------------

    model += (

        pulp.lpSum(

            float(vessel.Capacity_Tons)
            * vessel_selected[vessel.Vessel_ID]

            for vessel in vessel_df.itertuples()

        )

        <= available_capacity

    ), "Fleet_Availability"

    # ------------------------------------------------------------
    # Route Demand Constraints
    # ------------------------------------------------------------

    for cargo_row in cargo_df.itertuples():

        matching_vessels = vessel_df[
            (vessel_df["Origin"] == cargo_row.Origin)
            &
            (vessel_df["Destination"] == cargo_row.Destination)
        ]

        if matching_vessels.empty:
            continue

        vessel_ids = (
            matching_vessels["Vessel_ID"]
            .tolist()
        )

        model += (

            pulp.lpSum(
                cargo_quantity[v]
                for v in vessel_ids
            )

            == float(cargo_row.Demand_Tons)

        ), (
            f"Demand_"
            f"{cargo_row.Origin}_"
            f"{cargo_row.Destination}"
        )

    # ------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------

    solver = pulp.PULP_CBC_CMD(
        msg=False
    )

    model.solve(solver)

    status = pulp.LpStatus[
        model.status
    ]

    # ------------------------------------------------------------
    # Infeasible / non-optimal
    # ------------------------------------------------------------

    if status != "Optimal":

        return {

            "status": status,

            "feasible": False,

            "window": window_name,

            "freight_rate": round(
                freight_rate,
                2
            ),

            "total_cargo_tons": 0,

            "total_landed_cost": None,

            "average_landed_cost": None,

            "vessel_count": 0,

            "charter_cost": None,

            "allocations": []
        }

    # ------------------------------------------------------------
    # Extract solution
    # ------------------------------------------------------------

    allocations = []

    total_charter_cost = 0.0

    total_cargo = 0.0

    for vessel in vessel_df.itertuples():

        vessel_id = vessel.Vessel_ID

        selected = pulp.value(
            vessel_selected[vessel_id]
        )

        quantity = pulp.value(
            cargo_quantity[vessel_id]
        )

        if selected is None:
            selected = 0

        if quantity is None:
            quantity = 0

        if selected > 0.5 and quantity > 0:

            matching_cargo = cargo_df[
                (cargo_df["Origin"] == vessel.Origin)
                &
                (cargo_df["Destination"] == vessel.Destination)
            ]

            if matching_cargo.empty:
                continue

            cargo_row = (
                matching_cargo.iloc[0]
            )

            charter_cost = (
                float(vessel.Daily_Charter_USD)
                * float(vessel.Available_Days)
            )

            additional_delay_cost = (

                float(
                    cargo_row[
                        "Demurrage_USD_per_Ton"
                    ]
                )

                * port_delay_days
            )

            vessel_landed_cost = (

                float(
                    cargo_row[
                        "FOB_USD_per_Ton"
                    ]
                )

                +

                float(
                    cargo_row[
                        "Port_Cost_USD_per_Ton"
                    ]
                )

                +

                float(
                    cargo_row[
                        "Demurrage_USD_per_Ton"
                    ]
                )

                +

                additional_delay_cost

                +

                freight_rate
            )

            utilization = (

                quantity
                / float(vessel.Capacity_Tons)
                * 100
            )

            allocations.append({

                "vessel_id":
                    vessel_id,

                "vessel_type":
                    vessel.Vessel_Type,

                "origin":
                    vessel.Origin,

                "destination":
                    vessel.Destination,

                "cargo_tons":
                    round(
                        quantity,
                        2
                    ),

                "capacity_tons":
                    float(
                        vessel.Capacity_Tons
                    ),

                "utilization_pct":
                    round(
                        utilization,
                        2
                    ),

                "landed_cost_usd_per_ton":
                    round(
                        vessel_landed_cost,
                        2
                    )
            })

            total_charter_cost += (
                charter_cost
            )

            total_cargo += (
                quantity
            )

    # ------------------------------------------------------------
    # Total Landed Cost
    # ------------------------------------------------------------

    total_landed_cost = pulp.value(
        model.objective
    )

    if total_landed_cost is None:
        total_landed_cost = 0.0

    average_landed_cost = (

        total_landed_cost
        / total_cargo

        if total_cargo > 0

        else 0.0
    )

    return {

        "status": "Optimal",

        "feasible": True,

        "window": window_name,

        "freight_rate":
            round(
                freight_rate,
                2
            ),

        "total_cargo_tons":
            round(
                total_cargo,
                2
            ),

        "vessel_count":
            len(allocations),

        "charter_cost":
            round(
                total_charter_cost,
                2
            ),

        "average_landed_cost":
            round(
                average_landed_cost,
                2
            ),

        "total_landed_cost":
            round(
                total_landed_cost,
                2
            ),

        "allocations":
            allocations
    }


# ================================================================
# MAIN OPTIMIZER
# ================================================================

def run_vessel_optimizer(
    cargo_demand=None,
    freight_shock_pct=0.0,
    port_delay_days=0.0,
    vessel_availability_pct=100.0
):
    """
    Run complete freight procurement optimization.

    Parameters
    ----------
    cargo_demand:
        Optional total coking coal demand override.

    freight_shock_pct:
        Scenario freight shock.
        Example:
            +25 = freight increases by 25%
            -10 = freight decreases by 10%

    port_delay_days:
        Additional simulated port delay.

    vessel_availability_pct:
        Percentage of prototype fleet capacity available.
    """

    print()
    print("=" * 65)
    print(
        "        FREIGHT INTELLIGENCE MILP OPTIMIZER"
    )
    print("=" * 65)
    print()

    # ============================================================
    # LOAD DATA
    # ============================================================

    if not CARGO_FILE.exists():

        raise FileNotFoundError(
            f"Cargo data not found: {CARGO_FILE}"
        )

    if not VESSEL_FILE.exists():

        raise FileNotFoundError(
            f"Vessel data not found: {VESSEL_FILE}"
        )

    cargo_df = pd.read_csv(
        CARGO_FILE
    )

    vessel_df = pd.read_csv(
        VESSEL_FILE
    )

    forecast_data = load_forecast()

    # ============================================================
    # VALIDATE REQUIRED COLUMNS
    # ============================================================

    required_cargo_columns = [

        "Cargo",
        "Origin",
        "Destination",
        "Demand_Tons",
        "FOB_USD_per_Ton",
        "Port_Cost_USD_per_Ton",
        "Demurrage_USD_per_Ton",
        "Available_Days"
    ]

    required_vessel_columns = [

        "Vessel_ID",
        "Vessel_Type",
        "Capacity_Tons",
        "Daily_Charter_USD",
        "Speed_Knots",
        "Origin",
        "Destination",
        "Available_Days"
    ]

    missing_cargo = [

        col for col in required_cargo_columns

        if col not in cargo_df.columns
    ]

    missing_vessel = [

        col for col in required_vessel_columns

        if col not in vessel_df.columns
    ]

    if missing_cargo:

        raise ValueError(
            "Missing cargo columns: "
            + ", ".join(missing_cargo)
        )

    if missing_vessel:

        raise ValueError(
            "Missing vessel columns: "
            + ", ".join(missing_vessel)
        )

    # ============================================================
    # PROJECT SCOPE
    # ============================================================

    cargo_df = cargo_df[
        (cargo_df["Cargo"] == "Coking Coal")
        &
        (cargo_df["Origin"] == "Australia")
    ].copy()

    vessel_df = vessel_df[
        vessel_df["Origin"] == "Australia"
    ].copy()

    if cargo_df.empty:

        raise ValueError(
            "No Australia-origin Coking Coal "
            "cargo found in optimizer_data.csv."
        )

    if vessel_df.empty:

        raise ValueError(
            "No Australia-origin vessels "
            "found in vessel_data.csv."
        )

    # ============================================================
    # ORIGINAL DEMAND
    # ============================================================

    original_total_demand = float(
        cargo_df["Demand_Tons"].sum()
    )

    # ============================================================
    # DEMAND OVERRIDE
    # ============================================================

    if cargo_demand is not None:

        cargo_demand = float(
            cargo_demand
        )

        if cargo_demand <= 0:

            raise ValueError(
                "cargo_demand must be greater than zero."
            )

        scale = (

            cargo_demand
            / original_total_demand
        )

        cargo_df["Demand_Tons"] = (

            cargo_df["Demand_Tons"]
            * scale
        )

    total_cargo = float(
        cargo_df["Demand_Tons"].sum()
    )

    # ============================================================
    # CURRENT FREIGHT
    # ============================================================

    current_freight = get_current_freight(
        forecast_data
    )

    # ============================================================
    # FUTURE P50 FORECASTS
    # ============================================================

    forecast_7d = get_forecast_value(
        forecast_data,
        "7D"
    )

    forecast_15d = get_forecast_value(
        forecast_data,
        "15D"
    )

    forecast_30d = get_forecast_value(
        forecast_data,
        "30D"
    )

    # ============================================================
    # PROCUREMENT WINDOWS
    # ============================================================

    windows = {

        "NOW":
            current_freight,

        "7D":
            forecast_7d,

        "15D":
            forecast_15d,

        "30D":
            forecast_30d
    }

    # ============================================================
    # FREIGHT SHOCK
    # ============================================================

    shock_multiplier = (

        1
        + freight_shock_pct / 100.0
    )

    if shock_multiplier < 0:

        raise ValueError(
            "freight_shock_pct results in "
            "a negative freight multiplier."
        )

    scenario_windows = {

        window:
            base_rate
            * shock_multiplier

        for window, base_rate
        in windows.items()
    }

    # ============================================================
    # RUN MILP FOR EACH WINDOW
    # ============================================================

    results = []

    for window in [

        "NOW",
        "7D",
        "15D",
        "30D"

    ]:

        result = optimize_single_window(

            cargo_df=cargo_df,

            vessel_df=vessel_df,

            window_name=window,

            freight_rate=
                scenario_windows[window],

            port_delay_days=
                port_delay_days,

            vessel_availability_pct=
                vessel_availability_pct
        )

        result[
            "base_freight_usd_per_ton"
        ] = round(
            windows[window],
            2
        )

        result[
            "freight_shock_pct"
        ] = round(
            freight_shock_pct,
            2
        )

        result[
            "port_delay_days"
        ] = round(
            port_delay_days,
            2
        )

        result[
            "vessel_availability_pct"
        ] = round(
            vessel_availability_pct,
            2
        )

        results.append(
            result
        )

    # ============================================================
    # FEASIBLE WINDOWS
    # ============================================================

    feasible_results = [

        result

        for result in results

        if result["feasible"]
    ]

    if not feasible_results:

        raise RuntimeError(

            "No feasible procurement window "
            "found under the current constraints."
        )

    # ============================================================
    # BEST WINDOW
    # ============================================================

    best_result = min(

        feasible_results,

        key=lambda x:
            x["total_landed_cost"]
    )

    selected_window = (
        best_result["window"]
    )

    # ============================================================
    # RECOMMENDATION
    # ============================================================

    if selected_window == "NOW":

        recommendation = (
            "BOOK NOW"
        )

    else:

        recommendation = (
            f"WAIT {selected_window}"
        )

    # ============================================================
    # WINDOW COMPARISON
    # ============================================================

    comparison = []

    for result in results:

        comparison.append({

            "window":
                result["window"],

            "base_freight_usd_per_ton":
                result[
                    "base_freight_usd_per_ton"
                ],

            "scenario_freight_usd_per_ton":
                result[
                    "freight_rate"
                ],

            "average_landed_cost_usd_per_ton":
                result[
                    "average_landed_cost"
                ],

            "total_landed_cost_usd":
                result[
                    "total_landed_cost"
                ],

            "vessels_selected":
                result[
                    "vessel_count"
                ],

            "charter_cost_usd":
                result[
                    "charter_cost"
                ],

            "status":
                result[
                    "status"
                ],

            "feasible":
                result[
                    "feasible"
                ]
        })

    # ============================================================
    # NOW BASELINE
    # ============================================================

    now_result = next(

        result

        for result in results

        if result["window"] == "NOW"
    )

    # ============================================================
    # COST DIFFERENCE VS NOW
    # ============================================================

    cost_difference_vs_now = (

        best_result[
            "total_landed_cost"
        ]

        -

        now_result[
            "total_landed_cost"
        ]
    )

    # ============================================================
    # PERCENT DIFFERENCE VS NOW
    # ============================================================

    if (
        now_result["total_landed_cost"]
        and
        now_result["total_landed_cost"] != 0
    ):

        cost_difference_pct_vs_now = (

            cost_difference_vs_now
            /
            now_result[
                "total_landed_cost"
            ]
            * 100
        )

    else:

        cost_difference_pct_vs_now = 0.0

    # ============================================================
    # SELECTED FORECAST
    # ============================================================

    if selected_window == "NOW":

        selected_base_freight = (
            current_freight
        )

    else:

        selected_base_freight = (
            windows[selected_window]
        )

    # ============================================================
    # FINAL OUTPUT
    # ============================================================

    output = {

        "status":
            "OPTIMAL",

        "recommendation":
            recommendation,

        "selected_procurement_window":
            selected_window,

        "current_freight_usd_per_ton":
            round(
                current_freight,
                2
            ),

        "base_freight_usd_per_ton":
            round(
                selected_base_freight,
                2
            ),

        "forecast_freight_usd_per_ton":
            round(
                best_result[
                    "freight_rate"
                ],
                2
            ),

        "scenario_freight_usd_per_ton":
            round(
                best_result[
                    "freight_rate"
                ],
                2
            ),

        "freight_shock_pct":
            round(
                freight_shock_pct,
                2
            ),

        "port_delay_days":
            round(
                port_delay_days,
                2
            ),

        "vessel_availability_pct":
            round(
                vessel_availability_pct,
                2
            ),

        "total_cargo_tons":
            round(
                total_cargo,
                2
            ),

        "total_charter_cost_usd":
            best_result[
                "charter_cost"
            ],

        "average_landed_cost_usd_per_ton":
            best_result[
                "average_landed_cost"
            ],

        "total_landed_cost_usd":
            best_result[
                "total_landed_cost"
            ],

        "cost_difference_vs_now_usd":
            round(
                cost_difference_vs_now,
                2
            ),

        "cost_difference_pct_vs_now":
            round(
                cost_difference_pct_vs_now,
                2
            ),

        "vessel_count":
            best_result[
                "vessel_count"
            ],

        "vessel_allocations":
            best_result[
                "allocations"
            ],

        "procurement_window_comparison":
            comparison,

        "optimization_method":
            "Mixed Integer Linear Programming (MILP)",

        "forecast_type":
            "P50 scenario",

        "model_scope":
            "Australia to East Coast India Coking Coal",

        "data_status":
            "Prototype / simulated vessel inputs",

        "notes": [

            "NOW uses current freight.",

            "7D, 15D and 30D use their "
            "corresponding P50 forecasts.",

            "Freight shock is applied "
            "consistently to every window.",

            "The recommendation is the "
            "lowest total landed-cost "
            "feasible window.",

            "Port-delay cost uses a "
            "prototype linear demurrage "
            "assumption.",

            "Charter rates and vessel "
            "availability are prototype "
            "inputs.",

            "Forecast P50 is a model "
            "quantile and not a guaranteed "
            "confidence interval."
        ]
    }

    # ============================================================
    # SAVE JSON
    # ============================================================

    output_file = (
        OUTPUT_DIR
        /
        "vessel_optimization_output.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    # ============================================================
    # CONSOLE OUTPUT
    # ============================================================

    print(
        f"Optimization Status : "
        f"{output['status']}"
    )

    print(
        f"Recommendation      : "
        f"{recommendation}"
    )

    print(
        f"Selected Window     : "
        f"{selected_window}"
    )

    print(
        f"Current Freight     : "
        f"${current_freight:.2f}/ton"
    )

    print(
        f"Selected Base Rate  : "
        f"${selected_base_freight:.2f}/ton"
    )

    print(
        f"Scenario Freight    : "
        f"${best_result['freight_rate']:.2f}/ton"
    )

    print(
        f"Freight Shock       : "
        f"{freight_shock_pct:+.1f}%"
    )

    print(
        f"Port Delay          : "
        f"{port_delay_days:.1f} days"
    )

    print(
        f"Vessel Availability : "
        f"{vessel_availability_pct:.1f}%"
    )

    print(
        f"Total Cargo         : "
        f"{total_cargo:,.0f} tons"
    )

    print(
        f"Vessels Selected    : "
        f"{best_result['vessel_count']}"
    )

    print(
        f"Charter Cost        : "
        f"${best_result['charter_cost']:,.2f}"
    )

    print(
        f"Average Landed Cost : "
        f"${best_result['average_landed_cost']:,.2f}/ton"
    )

    print(
        f"Total Landed Cost   : "
        f"${best_result['total_landed_cost']:,.2f}"
    )

    print(
        f"Difference vs NOW   : "
        f"${cost_difference_vs_now:,.2f}"
    )

    print()
    print(
        "PROCUREMENT WINDOW COMPARISON"
    )
    print("-" * 65)

    for item in comparison:

        if item["feasible"]:

            print(

                f"{item['window']:>4} | "

                f"Freight "
                f"${item['scenario_freight_usd_per_ton']:.2f}/t | "

                f"Landed "
                f"${item['average_landed_cost_usd_per_ton']:.2f}/t | "

                f"Total "
                f"${item['total_landed_cost_usd']:,.0f}"
            )

        else:

            print(

                f"{item['window']:>4} | "
                f"INFEASIBLE"
            )

    print("-" * 65)

    print()
    print(
        "VESSEL ALLOCATION"
    )

    for allocation in (
        best_result["allocations"]
    ):

        print(

            f"{allocation['vessel_id']} | "

            f"{allocation['vessel_type']} | "

            f"{allocation['destination']} | "

            f"{allocation['cargo_tons']:,.0f} tons | "

            f"{allocation['utilization_pct']:.2f}% utilized"
        )

    print()

    print(
        "JSON OUTPUT         : SUCCESS"
    )

    print(
        f"Saved to            : "
        f"{output_file}"
    )

    print(
        "VESSEL MILP STATUS  : SUCCESS"
    )

    print("=" * 65)

    return output


# ================================================================
# DIRECT EXECUTION
# ================================================================

if __name__ == "__main__":

    run_vessel_optimizer()