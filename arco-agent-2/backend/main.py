
"""ARCO Agent 2.0 — Retail campaign optimization API."""

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.strategy_engine import generate_strategy

from backend.campaign_store import save_campaign, get_campaigns


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"
MODEL_PATH = ROOT / "models" / "demand_model.joblib"


# --------------------------------------------------
# APP
# --------------------------------------------------

app = FastAPI(
    title="ARCO Agent 2.0 API",
    version="0.4.0",
    description=(
        "Retail marketing strategy prototype using synthetic data, "
        "trained demand predictions, financial evaluation, and "
        "an optional language model."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class OptimizeRequest(BaseModel):
    product_id: str = "P0001"
    stock: int = Field(default=100, ge=0)
    campaign_cost: float = Field(default=500.0, ge=0)
    budget_limit: float = Field(default=5000.0, ge=0)
    max_discount_pct: float = Field(default=30.0, ge=0, le=100)
    min_margin_pct: float = Field(default=20.0, ge=0, le=100)
    base_demand: float = Field(default=30.0, ge=0)
    campaign_days: int = Field(default=7, ge=1, le=90)


class StrategyRequest(OptimizeRequest):
    objective: str = "balanced"


# --------------------------------------------------
# DATA ACCESS
# --------------------------------------------------

def read_csv(name: str) -> pd.DataFrame:
    path = DATA / name

    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Required synthetic data file not found: {name}. "
                "Run: python -m scripts.generate_synthetic_data"
            ),
        )

    return pd.read_csv(path)


def load_demand_model():
    """Load the trained demand model."""
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# HEALTH AND SUMMARY
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "project": "ARCO Agent 2.0",
        "data_mode": "synthetic",
        "demand_model_loaded": MODEL_PATH.exists(),
    }


@app.get("/api/summary")
def summary():
    products = read_csv("products.csv")
    customers = read_csv("customers.csv")
    transactions = read_csv("transactions.csv")
    inventory = read_csv("inventory.csv")

    revenue = float(transactions["revenue"].sum())

    gross_profit = float(
        (
            transactions["revenue"]
            - transactions["unit_cost"] * transactions["units"]
        ).sum()
    )

    return {
        "products": int(len(products)),
        "customers": int(len(customers)),
        "transactions": int(len(transactions)),
        "inventory_units": int(inventory["stock_quantity"].sum()),
        "historical_revenue": round(revenue, 2),
        "historical_gross_profit": round(gross_profit, 2),
        "data_mode": "Synthetic demonstration data",
    }


# --------------------------------------------------
# DEMAND PREDICTION
# --------------------------------------------------

def predict_demand(
    model,
    product_id: str,
    store_id: str,
    price: float,
    unit_cost: float,
    discount_pct: float,
    stock: float,
    month: int,
    day_of_week: int,
) -> float:
    """Predict demand for one product, store, and day."""

    features = pd.DataFrame(
        [
            {
                "unit_price": price * (1.0 - discount_pct / 100.0),
                "discount_pct": discount_pct,
                "unit_cost": unit_cost,
                "stock_available": stock,
                "month": month,
                "day_of_week": day_of_week,
                "product_id": product_id,
                "store_id": store_id,
            }
        ]
    )

    prediction = float(model.predict(features)[0])

    return min(max(prediction, 0.0), float(stock))


# --------------------------------------------------
# CAMPAIGN OPTIMIZER
# --------------------------------------------------

@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    products = read_csv("products.csv")

    match = products.loc[
        products["product_id"].astype(str) == req.product_id
    ]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown product_id: {req.product_id}",
        )

    product = match.iloc[0]
    model = load_demand_model()

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Demand model not found. Train the model first.",
        )

    rules = read_csv("brand_rules.csv")

    brand_rule = rules.loc[
        rules["brand"].astype(str) == str(product["brand"])
    ]

    # Start with the user's requested constraints.
    max_discount = float(req.max_discount_pct)
    min_margin = float(req.min_margin_pct)

    # IMPORTANT:
    # The budget entered by the user is the optimizer's budget.
    # Do not silently reduce it using the synthetic brand-rule budget.
    budget = float(req.budget_limit)

    # Apply brand discount and margin restrictions where available.
    if not brand_rule.empty:
        rule = brand_rule.iloc[0]

        if pd.notna(rule.get("max_discount_pct")):
            max_discount = min(
                max_discount,
                float(rule["max_discount_pct"]),
            )

        if pd.notna(rule.get("min_margin_pct")):
            min_margin = max(
                min_margin,
                float(rule["min_margin_pct"]),
            )

    price = float(product["base_price"])
    unit_cost = float(product["unit_cost"])
    campaign_days = int(req.campaign_days)

    # Fixed reference date keeps comparisons reproducible.
    start_date = date(2026, 11, 2)
    store_id = "S01"

    # Evaluate supported discount levels up to the effective maximum.
    discount_options = [
        discount
        for discount in [0, 5, 10, 15, 20, 25, 30]
        if discount <= max_discount
    ]

    if 0 not in discount_options:
        discount_options.insert(0, 0)

    candidates = []

    for discount in discount_options:
        total_units = 0.0
        total_revenue = 0.0
        total_gross_profit = 0.0
        remaining_stock = float(req.stock)
        daily_predictions = []

        for day_index in range(campaign_days):
            current_date = start_date + timedelta(days=day_index)

            if remaining_stock <= 0:
                daily_predictions.append(0.0)
                continue

            demand = predict_demand(
                model=model,
                product_id=str(product["product_id"]),
                store_id=store_id,
                price=price,
                unit_cost=unit_cost,
                discount_pct=discount,
                stock=remaining_stock,
                month=current_date.month,
                day_of_week=current_date.weekday(),
            )

            units = min(
                max(float(demand), 0.0),
                remaining_stock,
            )

            selling_price = price * (1.0 - discount / 100.0)
            daily_revenue = units * selling_price
            daily_gross_profit = units * (selling_price - unit_cost)

            total_units += units
            total_revenue += daily_revenue
            total_gross_profit += daily_gross_profit

            remaining_stock = max(0.0, remaining_stock - units)
            daily_predictions.append(round(units, 2))

        # Charge the entered campaign cost exactly once for a paid offer.
        # A 0% offer represents the no-paid-campaign baseline.
        applied_campaign_cost = (
            0.0 if discount == 0 else float(req.campaign_cost)
        )

        net_profit = total_gross_profit - applied_campaign_cost

        margin_pct = (
            total_gross_profit / total_revenue * 100.0
            if total_revenue > 0
            else 0.0
        )

        reason = None

        if discount > max_discount:
            reason = "Discount limit"
        elif margin_pct < min_margin:
            reason = "Minimum margin"
        elif applied_campaign_cost > budget:
            reason = "Campaign budget"

        candidates.append(
            {
                "discount_pct": float(discount),
                "feasible": reason is None,
                "reason": reason,
                "predicted_units": round(total_units, 2),
                "daily_predictions": daily_predictions,
                "revenue": round(total_revenue, 2),
                "gross_profit": round(total_gross_profit, 2),
                "margin_pct": round(margin_pct, 2),
                "campaign_cost": round(applied_campaign_cost, 2),
                "net_profit": round(net_profit, 2),
                "incremental_profit": None,
            }
        )

    # Find the feasible 0% baseline.
    baseline = next(
        (
            candidate
            for candidate in candidates
            if candidate["discount_pct"] == 0
            and candidate["feasible"]
        ),
        None,
    )

    # Compare each feasible offer with the baseline.
    for candidate in candidates:
        if baseline is not None and candidate["feasible"]:
            candidate["incremental_profit"] = round(
                candidate["net_profit"] - baseline["net_profit"],
                2,
            )

    feasible_candidates = [
        candidate
        for candidate in candidates
        if candidate["feasible"]
    ]

    # Select the feasible candidate with the highest net profit.
    best = (
        max(
            feasible_candidates,
            key=lambda candidate: candidate["net_profit"],
        )
        if feasible_candidates
        else None
    )

    return {
        "product": {
            "product_id": str(product["product_id"]),
            "product_name": str(product["product_name"]),
            "brand": str(product["brand"]),
            "category": str(product["category"]),
            "base_price": price,
            "unit_cost": unit_cost,
        },
        "campaign_days": campaign_days,
        "store_id": store_id,
        "baseline": baseline,
        "best_strategy": best,
        "effective_constraints": {
            "max_discount_pct": max_discount,
            "min_margin_pct": min_margin,
            "budget_limit": budget,
            "requested_campaign_cost": float(req.campaign_cost),
        },
        "candidates": candidates,
        "note": (
            "Predictions use the trained demand model and synthetic data. "
            "Campaign cost is charged once for discounted campaigns; "
            "the 0% baseline has no campaign cost. Stock is shared across "
            "days and no replenishment is assumed. Results are estimates, "
            "not guaranteed sales."
        ),
    }


# --------------------------------------------------
# HYBRID AI MARKETING STRATEGY
# --------------------------------------------------

@app.post("/api/strategy")
def create_strategy(req: StrategyRequest):
    objective = req.objective.strip().lower()

    allowed_objectives = {
        "balanced",
        "profit",
        "sales",
        "inventory",
    }

    if objective not in allowed_objectives:
        raise HTTPException(
            status_code=422,
            detail=(
                "Objective must be balanced, profit, sales, or inventory."
            ),
        )

    # Use the same optimizer and financial constraints.
    optimization = optimize(req)

    try:
        strategy = generate_strategy(
            optimization=optimization,
            objective=objective,
        )
        return strategy

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )



@app.post("/api/campaigns/generate")
def generate_and_save_campaign(req: StrategyRequest):
    """
    Generate a strategy and save it to campaign history.
    """
    optimization = optimize(req)
    strategy = generate_strategy(optimization, req.objective)

    campaign_id = save_campaign(strategy)

    return {
        **strategy,
        "campaign_id": campaign_id,
        "saved": True
    }


@app.get("/api/campaigns")
def list_campaigns():
    """
    Return saved campaigns, newest first.
    """
    campaigns = get_campaigns()

    return {
        "status": "success",
        "count": len(campaigns),
        "campaigns": campaigns
    }


@app.get("/api/campaigns/{campaign_id}")
def get_campaign_details(campaign_id: int):
    """
    Return the complete saved strategy for one campaign.
    """
    import json
    import sqlite3
    from backend.campaign_store import get_connection

    with get_connection() as connection:
        row = connection.execute(
            "SELECT strategy_json FROM campaigns WHERE id = ?",
            (campaign_id,)
        ).fetchone()

    if row is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail="Campaign not found"
        )

    return {
        "campaign_id": campaign_id,
        "strategy": json.loads(row["strategy_json"])
    }
