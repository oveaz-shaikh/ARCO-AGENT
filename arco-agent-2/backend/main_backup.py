import joblib
from datetime import date
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.financial_engine import evaluate_offer, illustrative_demand

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "synthetic"

MODEL_PATH = ROOT / "models" / "demand_model.joblib"

def load_demand_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

app = FastAPI(title="ARCO Agent 2.0 API", version="0.1.0",
              description="Synthetic-data prototype; not connected to Alshaya systems.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    product_id: str = "P0001"
    stock: int = Field(default=100, ge=0)
    campaign_cost: float = Field(default=500.0, ge=0)
    budget_limit: float = Field(default=5000.0, ge=0)
    max_discount_pct: float = Field(default=30.0, ge=0, le=100)
    min_margin_pct: float = Field(default=20.0, ge=0, le=100)
    base_demand: float = Field(default=30.0, ge=0)
    campaign_days: int = Field(default=7, ge=1, le=90)

def read_csv(name: str):
    path = DATA / name
    if not path.exists():
        raise HTTPException(status_code=400, detail="Synthetic data not generated. Run: python -m scripts.generate_synthetic_data")
    return pd.read_csv(path)

@app.get("/api/health")
def health():
    return {"status": "ok", "project": "ARCO Agent 2.0", "data_mode": "synthetic"}

@app.get("/api/summary")
def summary():
    products = read_csv("products.csv")
    customers = read_csv("customers.csv")
    transactions = read_csv("transactions.csv")
    inventory = read_csv("inventory.csv")
    revenue = float(transactions["revenue"].sum())
    gross_profit = float((transactions["revenue"] - transactions["unit_cost"] * transactions["units"]).sum())
    return {
        "products": int(len(products)),
        "customers": int(len(customers)),
        "transactions": int(len(transactions)),
        "inventory_units": int(inventory["stock_quantity"].sum()),
        "historical_revenue": round(revenue, 2),
        "historical_gross_profit": round(gross_profit, 2),
        "data_mode": "Synthetic demonstration data",
    }


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    products = read_csv("products.csv")
    match = products.loc[products["product_id"] == req.product_id]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown product_id: {req.product_id}",
        )

    product = match.iloc[0]

    # Load the trained demand model.
    model = load_demand_model()
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Demand model not found. Train it first.",
        )

    rules = read_csv("brand_rules.csv")
    brand_rule = rules.loc[rules["brand"] == product["brand"]]

    max_discount = req.max_discount_pct
    min_margin = req.min_margin_pct
    budget = req.budget_limit

    if not brand_rule.empty:
        rule = brand_rule.iloc[0]
        max_discount = min(
            max_discount, float(rule["max_discount_pct"])
        )
        min_margin = max(
            min_margin, float(rule["min_margin_pct"])
        )
        budget = min(
            budget, float(rule["campaign_budget_limit"])
        )

    # Use a fixed, reproducible reference date for comparisons.
    month = 11
    day_of_week = 0
    price = float(product["base_price"])
    unit_cost = float(product["unit_cost"])

    candidates = []

    for discount in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
        demand = predict_demand(
            model=model,
            product_id=str(product["product_id"]),
            store_id="S01",
            price=price,
            unit_cost=unit_cost,
            discount_pct=discount,
            stock=req.stock,
            month=month,
            day_of_week=day_of_week,
        )

        # No-campaign baseline has no campaign expenditure.
        campaign_cost = 0.0 if discount == 0 else req.campaign_cost

        result = evaluate_offer(
            price=price,
            unit_cost=unit_cost,
            discount_pct=discount,
            predicted_units=demand,
            campaign_cost=campaign_cost,
            stock=req.stock,
            max_discount_pct=max_discount,
            min_margin_pct=min_margin,
            budget_limit=budget,
        )

        candidates.append({
            "discount_pct": discount,
            "feasible": result["feasible"],
            "reason": result.get("reason"),
            "predicted_units": result.get(
                "predicted_units", round(demand, 2)
            ),
            "revenue": result.get("revenue"),
            "gross_profit": result.get("gross_profit"),
            "margin_pct": result.get("margin_pct"),
            "campaign_cost": result.get("campaign_cost"),
            "net_profit": result.get("net_profit"),
        })

    baseline = next(
        (
            c for c in candidates
            if c["discount_pct"] == 0 and c["feasible"]
        ),
        None,
    )

    feasible = [c for c in candidates if c["feasible"]]

    best = (
        max(feasible, key=lambda c: c["net_profit"])
        if feasible else None
    )

    if best is not None and baseline is not None:
        best["incremental_profit"] = round(
            best["net_profit"] - baseline["net_profit"], 2
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
        "baseline": baseline,
        "effective_constraints": {
            "max_discount_pct": max_discount,
            "min_margin_pct": min_margin,
            "budget_limit": budget,
        },
        "best_strategy": best,
        "candidates": candidates,
        "note": (
            "Demand comes from the trained model. "
            "All data and business rules are synthetic. "
            "Predictions are for one product-store-day."
        ),
    }

    products = read_csv("products.csv")
    match = products.loc[products["product_id"] == req.product_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown product_id: {req.product_id}")
    product = match.iloc[0]
    rule_rows = read_csv("brand_rules.csv")
    rule = rule_rows.loc[rule_rows["brand"] == product["brand"]]
    if not rule.empty:
        rule = rule.iloc[0]
        max_discount = min(req.max_discount_pct, float(rule["max_discount_pct"]))
        min_margin = max(req.min_margin_pct, float(rule["min_margin_pct"]))
        budget = min(req.budget_limit, float(rule["campaign_budget_limit"]))
    else:
        max_discount, min_margin, budget = req.max_discount_pct, req.min_margin_pct, req.budget_limit

    candidates = []
    for discount in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
        demand = illustrative_demand(req.base_demand, discount)
        result = evaluate_offer(
            price=float(product["base_price"]),
            unit_cost=float(product["unit_cost"]),
            discount_pct=discount,
            predicted_units=demand,
            campaign_cost=req.campaign_cost,
            stock=req.stock,
            max_discount_pct=max_discount,
            min_margin_pct=min_margin,
            budget_limit=budget,
        )
        candidates.append({
            "discount_pct": discount,
            "feasible": result["feasible"],
            "reason": result.get("reason"),
            "predicted_units": result.get("predicted_units", round(min(demand, req.stock), 2)),
            "revenue": result.get("revenue"),
            "gross_profit": result.get("gross_profit"),
            "margin_pct": result.get("margin_pct"),
            "campaign_cost": result.get("campaign_cost"),
            "net_profit": result.get("net_profit"),
        })

    feasible = [c for c in candidates if c["feasible"]]
    # MVP selects the feasible offer with highest net profit. Incremental-profit baseline
    # and trained demand predictions will be added in the next milestone.
    best = max(feasible, key=lambda c: c["net_profit"]) if feasible else None
    return {
        "product": {
            "product_id": str(product["product_id"]),
            "product_name": str(product["product_name"]),
            "brand": str(product["brand"]),
            "category": str(product["category"]),
            "base_price": float(product["base_price"]),
            "unit_cost": float(product["unit_cost"]),
        },
        "effective_constraints": {
            "max_discount_pct": max_discount,
            "min_margin_pct": min_margin,
            "budget_limit": budget,
        },
        "best_strategy": best,
        "candidates": candidates,
        "note": "Prototype uses a transparent illustrative demand curve, not a trained model. All rules/data are synthetic.",
    }



def predict_demand(
    model,
    product_id,
    store_id,
    price,
    unit_cost,
    discount_pct,
    stock,
    month,
    day_of_week,
):
    features = pd.DataFrame([{
        "unit_price": price * (1 - discount_pct / 100),
        "discount_pct": discount_pct,
        "unit_cost": unit_cost,
        "stock_available": stock,
        "month": month,
        "day_of_week": day_of_week,
        "product_id": product_id,
        "store_id": store_id,
    }])

    prediction = float(model.predict(features)[0])
    return min(max(prediction, 0.0), float(stock))
