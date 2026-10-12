"""Transparent first-pass financial evaluation for a single-product campaign."""
from typing import Any

def evaluate_offer(
    *,
    price: float,
    unit_cost: float,
    discount_pct: float,
    predicted_units: float,
    campaign_cost: float,
    stock: int,
    max_discount_pct: float,
    min_margin_pct: float,
    budget_limit: float,
) -> dict[str, Any]:
    if price <= 0 or unit_cost < 0 or campaign_cost < 0 or stock < 0:
        return {"feasible": False, "reason": "Invalid non-positive price or negative cost/stock"}
    if discount_pct < 0 or discount_pct > max_discount_pct:
        return {"feasible": False, "reason": "Discount limit"}
    if campaign_cost > budget_limit:
        return {"feasible": False, "reason": "Budget limit"}

    units = min(max(float(predicted_units), 0.0), float(stock))
    selling_price = price * (1.0 - discount_pct / 100.0)
    revenue = selling_price * units
    cogs = unit_cost * units
    gross_profit = revenue - cogs
    margin_pct = (gross_profit / revenue * 100.0) if revenue > 0 else 0.0
    net_profit = gross_profit - campaign_cost

    if margin_pct < min_margin_pct:
        return {
            "feasible": False,
            "reason": "Minimum margin",
            "discount_pct": discount_pct,
            "margin_pct": round(margin_pct, 2),
        }

    return {
        "feasible": True,
        "discount_pct": float(discount_pct),
        "predicted_units": round(units, 2),
        "revenue": round(revenue, 2),
        "cost_of_goods": round(cogs, 2),
        "gross_profit": round(gross_profit, 2),
        "margin_pct": round(margin_pct, 2),
        "campaign_cost": round(campaign_cost, 2),
        "net_profit": round(net_profit, 2),
    }

def illustrative_demand(base_demand: float, discount_pct: float) -> float:
    """Temporary transparent demand curve. Replace with trained model predictions later."""
    return max(0.0, base_demand * (1.0 + 0.035 * discount_pct))
