from backend.financial_engine import evaluate_offer

def test_offer_is_rejected_when_discount_exceeds_limit():
    result = evaluate_offer(
        price=100, unit_cost=50, discount_pct=30, predicted_units=10,
        campaign_cost=0, stock=20, max_discount_pct=20,
        min_margin_pct=10, budget_limit=100,
    )
    assert result["feasible"] is False
    assert result["reason"] == "Discount limit"

def test_feasible_offer_calculates_profit():
    result = evaluate_offer(
        price=100, unit_cost=40, discount_pct=10, predicted_units=10,
        campaign_cost=50, stock=20, max_discount_pct=20,
        min_margin_pct=20, budget_limit=100,
    )
    assert result["feasible"] is True
    assert result["revenue"] == 900
    assert result["gross_profit"] == 500
    assert result["net_profit"] == 450

def test_units_are_capped_by_stock():
    result = evaluate_offer(
        price=100, unit_cost=40, discount_pct=0, predicted_units=50,
        campaign_cost=0, stock=5, max_discount_pct=20,
        min_margin_pct=20, budget_limit=100,
    )
    assert result["feasible"] is True
    assert result["predicted_units"] == 5
