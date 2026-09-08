# ================================================================
# SIH FREIGHT INTELLIGENCE
# MILP OPTIMIZATION ENGINE
# Vessel Chartering + Bulk Cargo Procurement
# ================================================================

import pandas as pd
import json

from pathlib import Path
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpStatus,
    value
)


# ================================================================
# PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "optimizer" / "optimizer_data.csv"
FORECAST_FILE = BASE_DIR / "outputs" / "forecast_output.json"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# LOAD DATA
# ================================================================

cargo_data = pd.read_csv(DATA_FILE)

with open(FORECAST_FILE, "r", encoding="utf-8") as file:
    forecast_data = json.load(file)


# ================================================================
# FORECAST SCENARIOS
# ================================================================

scenarios = {
    "NOW": {
        "freight": forecast_data["current_freight_usd_per_ton"],
        "horizon": 0
    },
    "7D": {
        "freight": forecast_data["horizons"]["7D"]["P50"],
        "horizon": 7
    },
    "15D": {
        "freight": forecast_data["horizons"]["15D"]["P50"],
        "horizon": 15
    },
    "30D": {
        "freight": forecast_data["horizons"]["30D"]["P50"],
        "horizon": 30
    }
}


# ================================================================
# MILP OPTIMIZER
# ================================================================

def optimize_procurement():

    problem = LpProblem(
        "Freight_Procurement_Optimization",
        LpMinimize
    )


    # ============================================================
    # DECISION VARIABLES
    # ============================================================

    # Select exactly one procurement window.

    select = {
        scenario: LpVariable(
            f"Select_{scenario}",
            cat="Binary"
        )
        for scenario in scenarios
    }


    # Cargo quantity for each cargo/route option.

    quantities = {}

    for index, row in cargo_data.iterrows():

        quantities[index] = LpVariable(
            f"Cargo_{index}",
            lowBound=0,
            upBound=float(row["Demand_Tons"])
        )


    # ============================================================
    # AUXILIARY VARIABLES
    # ============================================================

    # freight_quantity[index, scenario]
    #
    # This represents:
    #
    # quantity of cargo assigned to a particular
    # procurement scenario.
    #
    # It prevents the illegal multiplication:
    #
    # select * quantity
    #
    # and keeps the model linear.

    freight_quantity = {}

    for index, row in cargo_data.iterrows():

        demand = float(row["Demand_Tons"])

        for scenario in scenarios:

            freight_quantity[index, scenario] = LpVariable(
                f"FreightQty_{index}_{scenario}",
                lowBound=0,
                upBound=demand
            )


    # ============================================================
    # CONSTRAINT: SCENARIO LINKING
    # ============================================================

    for index, row in cargo_data.iterrows():

        demand = float(row["Demand_Tons"])

        for scenario in scenarios:

            problem += (
                freight_quantity[index, scenario]
                <= demand * select[scenario]
            ), f"ScenarioLink_{index}_{scenario}"


    # ============================================================
    # CONSTRAINT: TOTAL CARGO
    # ============================================================

    for index, row in cargo_data.iterrows():

        problem += (
            quantities[index]
            ==
            lpSum(
                freight_quantity[index, scenario]
                for scenario in scenarios
            )
        ), f"QuantityLink_{index}"


    # ============================================================
    # CONSTRAINT: EXACTLY ONE PROCUREMENT WINDOW
    # ============================================================

    problem += (
        lpSum(
            select[scenario]
            for scenario in scenarios
        ) == 1
    ), "One_Procurement_Timing"


    # ============================================================
    # CONSTRAINT: DEMAND SATISFACTION
    # ============================================================

    for index, row in cargo_data.iterrows():

        demand = float(row["Demand_Tons"])

        problem += (
            quantities[index] == demand
        ), f"Demand_{index}"


    # ============================================================
    # OBJECTIVE FUNCTION
    # ============================================================

    total_cost = 0

    for index, row in cargo_data.iterrows():

        fob_cost = float(
            row["FOB_USD_per_Ton"]
        )

        port_cost = float(
            row["Port_Cost_USD_per_Ton"]
        )

        demurrage_cost = float(
            row["Demurrage_USD_per_Ton"]
        )


        base_cost_per_ton = (
            fob_cost
            + port_cost
            + demurrage_cost
        )


        # Base cargo cost.

        total_cost += (
            base_cost_per_ton
            * quantities[index]
        )


        # Scenario-specific freight cost.

        for scenario in scenarios:

            freight_rate = float(
                scenarios[scenario]["freight"]
            )

            total_cost += (
                freight_rate
                * freight_quantity[index, scenario]
            )


    problem += total_cost


    # ============================================================
    # SOLVE
    # ============================================================

    problem.solve()


    # ============================================================
    # STATUS
    # ============================================================

    status = LpStatus[problem.status]


    if status != "Optimal":

        return {
            "status": status,
            "message": "No optimal solution found."
        }


    # ============================================================
    # SELECTED SCENARIO
    # ============================================================

    selected_scenario = None

    for scenario in scenarios:

        if value(select[scenario]) > 0.5:

            selected_scenario = scenario

            break


    # ============================================================
    # CARGO ALLOCATION
    # ============================================================

    allocation = []

    total_cargo_tons = 0
    total_landed_cost = 0


    for index, row in cargo_data.iterrows():

        quantity = value(
            quantities[index]
        )

        freight_rate = scenarios[
            selected_scenario
        ]["freight"]


        landed_cost_per_ton = (

            float(row["FOB_USD_per_Ton"])

            + freight_rate

            + float(row["Port_Cost_USD_per_Ton"])

            + float(row["Demurrage_USD_per_Ton"])
        )


        cargo_total_cost = (
            quantity
            * landed_cost_per_ton
        )


        total_cargo_tons += quantity
        total_landed_cost += cargo_total_cost


        allocation.append({

            "cargo":
                row["Cargo"],

            "origin":
                row["Origin"],

            "destination":
                row["Destination"],

            "quantity_tons":
                round(quantity, 2),

            "freight_usd_per_ton":
                round(freight_rate, 2),

            "landed_cost_usd_per_ton":
                round(
                    landed_cost_per_ton,
                    2
                ),

            "total_cost_usd":
                round(
                    cargo_total_cost,
                    2
                )
        })


    # ============================================================
    # RECOMMENDATION
    # ============================================================

    if selected_scenario == "NOW":

        action = "BOOK TODAY"

    else:

        action = f"WAIT {selected_scenario}"


    # ============================================================
    # RESULT
    # ============================================================

    result = {

        "status":
            "OPTIMAL",

        "recommendation":
            action,

        "selected_window":
            selected_scenario,

        "forecast_freight_usd_per_ton":
            round(
                scenarios[selected_scenario]["freight"],
                2
            ),

        "total_cargo_tons":
            round(
                total_cargo_tons,
                2
            ),

        "total_landed_cost_usd":
            round(
                total_landed_cost,
                2
            ),

        "average_landed_cost_usd_per_ton":
            round(
                total_landed_cost
                / total_cargo_tons,
                2
            ),

        "cargo_allocation":
            allocation,

        "optimization_method":
            "Mixed Integer Linear Programming (MILP)"
    }


    return result


# ================================================================
# SAVE RESULT
# ================================================================

def save_result(result):

    output_file = (
        OUTPUT_DIR
        / "optimization_output.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    return output_file


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":

    print()

    print("=" * 70)

    print(
        "SIH FREIGHT INTELLIGENCE"
    )

    print(
        "MILP PROCUREMENT & CHARTERING OPTIMIZER"
    )

    print("=" * 70)


    try:

        result = optimize_procurement()


        print()

        print(
            f"Optimization Status : "
            f"{result['status']}"
        )


        if result["status"] == "OPTIMAL":

            print()

            print(
                f"Recommendation      : "
                f"{result['recommendation']}"
            )

            print(
                f"Selected Window     : "
                f"{result['selected_window']}"
            )

            print(
                f"Forecast Freight    : "
                f"${result['forecast_freight_usd_per_ton']:.2f}/ton"
            )

            print(
                f"Total Cargo         : "
                f"{result['total_cargo_tons']:,.0f} tons"
            )

            print(
                f"Average Landed Cost : "
                f"${result['average_landed_cost_usd_per_ton']:.2f}/ton"
            )

            print(
                f"Total Landed Cost   : "
                f"${result['total_landed_cost_usd']:,.2f}"
            )


            output_file = save_result(result)


            print()

            print(
                "JSON OUTPUT         : SUCCESS"
            )

            print(
                f"Saved to            : "
                f"{output_file}"
            )

            print()

            print(
                "MILP OPTIMIZER STATUS: SUCCESS"
            )


        else:

            print(
                result["message"]
            )


        print("=" * 70)


    except Exception as error:

        print()

        print("=" * 70)

        print(
            "MILP OPTIMIZER ERROR"
        )

        print("=" * 70)

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        print("=" * 70)

        raise