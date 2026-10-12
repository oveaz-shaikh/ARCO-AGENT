import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

ALLOWED_OBJECTIVES = {"balanced", "profit", "sales", "inventory"}


def build_rule_based_strategy(optimization, objective):
    """Build an objective-specific marketing plan from optimizer results."""
    product = optimization["product"]
    candidates = optimization.get("candidates", [])
    baseline = optimization.get("baseline")
    constraints = optimization.get("effective_constraints", {})

    feasible = [item for item in candidates if item.get("feasible")]

    if not feasible:
        return {
            "status": "no_feasible_strategy",
            "product": product.get("product_name"),
            "product_id": product.get("product_id"),
            "objective": objective,
            "recommendation": (
                "No offer satisfies the current constraints. Review the "
                "available stock, budget, margin, and discount limits."
            ),
            "recommended_discount_pct": None,
            "expected_units": None,
            "expected_revenue": None,
            "expected_gross_profit": None,
            "expected_net_profit": None,
            "incremental_profit": None,
            "target_audience": [],
            "channels": [],
            "campaign_ideas": [],
            "risks": ["Do not launch until a feasible offer is available."],
            "next_actions": [
                "Review the effective business constraints.",
                "Verify stock and campaign expenditure.",
                "Run the optimizer again.",
            ],
            "generation_method": "rules",
        }

    # Establish the no-discount baseline if the optimizer did not supply one.
    if baseline is None:
        baseline = next(
            (
                item for item in feasible
                if float(item["discount_pct"]) == 0
            ),
            None,
        )

    # OBJECTIVE-SPECIFIC OFFER SELECTION
    if objective == "profit":
        # Maximize estimated net profit; units break ties.
        chosen = max(
            feasible,
            key=lambda item: (
                float(item["net_profit"]),
                float(item["predicted_units"]),
            ),
        )

    elif objective in {"sales", "inventory"}:
        # Prioritize predicted units; profit breaks ties.
        chosen = max(
            feasible,
            key=lambda item: (
                float(item["predicted_units"]),
                float(item["net_profit"]),
            ),
        )

    else:
        # Balanced: compare ALL feasible offers using 60% normalized
        # predicted units and 40% normalized net profit.
        # Do not filter by baseline profit: that can force 0% to win
        # regardless of the intended sales/profit trade-off.
        balanced_options = feasible

        min_units = min(
            float(item["predicted_units"]) for item in balanced_options
        )
        max_units = max(
            float(item["predicted_units"]) for item in balanced_options
        )
        min_profit = min(
            float(item["net_profit"]) for item in balanced_options
        )
        max_profit = max(
            float(item["net_profit"]) for item in balanced_options
        )

        def normalize(value, low, high):
            if high <= low:
                return 1.0
            return (value - low) / (high - low)

        def balanced_score(item):
            sales_score = normalize(
                float(item["predicted_units"]), min_units, max_units
            )
            profit_score = normalize(
                float(item["net_profit"]), min_profit, max_profit
            )
            return 0.60 * sales_score + 0.40 * profit_score

        chosen = max(
            balanced_options,
            key=lambda item: (
                balanced_score(item),
                float(item["predicted_units"]),
                float(item["net_profit"]),
            ),
        )

    # FINANCIAL SUMMARY
    discount = float(chosen["discount_pct"])
    expected_units = float(chosen["predicted_units"])
    net_profit = float(chosen["net_profit"])

    baseline_profit = (
        float(baseline["net_profit"]) if baseline is not None else None
    )
    incremental_profit = (
        round(net_profit - baseline_profit, 2)
        if baseline_profit is not None
        else None
    )

    # RECOMMENDATION EXPLANATION
    if discount == 0 and objective == "balanced":
        recommendation = (
            "The balanced score selected the base-price offer after comparing "
            "all feasible candidates using 60% normalized predicted units and "
            "40% normalized net profit. Review the candidate scores and "
            "predictions if you expected a discounted offer to rank higher."
        )
    elif discount == 0:
        recommendation = (
            "Keep the product at its base price. Under the current demand "
            "estimates and campaign-cost assumptions, the no-discount offer "
            "was selected for this objective."
        )
    elif objective == "profit":
        recommendation = (
            f"Test a {discount:g}% discount for "
            f"{optimization['campaign_days']} days in "
            f"{optimization['store_id']}. This offer has the highest "
            "estimated net profit among feasible candidates."
        )
    elif objective == "sales":
        recommendation = (
            f"Test a {discount:g}% discount for "
            f"{optimization['campaign_days']} days in "
            f"{optimization['store_id']}. This offer has the highest "
            "predicted unit sales among feasible candidates. Review the "
            "estimated profit trade-off before launch."
        )
    elif objective == "inventory":
        recommendation = (
            f"Consider a {discount:g}% discount for "
            f"{optimization['campaign_days']} days in "
            f"{optimization['store_id']} to prioritize stock movement. "
            "Monitor sales, margins, and remaining inventory."
        )
    else:
        recommendation = (
            f"Test a {discount:g}% discount for "
            f"{optimization['campaign_days']} days in "
            f"{optimization['store_id']}. This offer provides the best "
            "normalized balance of predicted sales and net profit among "
            "the eligible options that meet the baseline-profit policy."
        )

    # MARKETING PLAN
    if objective == "inventory":
        campaign_ideas = [
            "Prioritize eligible products with available inventory.",
            "Test a time-limited promotion within approved constraints.",
            "Monitor daily stock movement and remaining inventory.",
        ]
        audience = [
            "Customers interested in the product category",
            "Price-sensitive shoppers, where supported by available data",
        ]
    elif objective == "sales":
        campaign_ideas = [
            "Promote the product through relevant category campaigns.",
            "Test product-discovery placements and targeted offers.",
            "Compare actual sales against predicted demand before scaling.",
        ]
        audience = [
            "Customers interested in the product category",
            "Potential repeat purchasers, where supported by available data",
        ]
    elif objective == "profit":
        campaign_ideas = [
            "Prioritize profitable product placements.",
            "Avoid unnecessary discounts when the baseline is more profitable.",
            "Measure net profit after campaign expenditure.",
        ]
        audience = [
            "Relevant customers with demonstrated category interest",
            "Existing customers suitable for product recommendations",
        ]
    else:
        campaign_ideas = [
            "Use the selected feasible discount as the promotional limit.",
            "Promote the product to relevant category audiences.",
            "Track sales, margin, profit, and inventory together.",
        ]
        audience = [
            "Customers interested in the product category",
            "Potential repeat purchasers where supported by available data",
        ]

    risks = [
        "All data and business rules are synthetic.",
        "Demand predictions are estimates, not guaranteed sales.",
        "No stock replenishment is assumed during the campaign.",
        "The model does not establish that discounts cause demand increases.",
    ]

    if incremental_profit is not None and incremental_profit < 0:
        risks.append(
            "The selected offer has lower estimated net profit than the "
            "baseline. Review this trade-off before launching."
        )

    strategy = {
        "status": "success",
        "product": product.get("product_name"),
        "product_id": product.get("product_id"),
        "brand": product.get("brand"),
        "objective": objective,
        "campaign_days": optimization["campaign_days"],
        "store_id": optimization["store_id"],
        "recommendation": recommendation,
        "recommended_discount_pct": discount,
        "expected_units": round(expected_units, 2),
        "expected_revenue": chosen["revenue"],
        "expected_gross_profit": chosen["gross_profit"],
        "expected_net_profit": chosen["net_profit"],
        "incremental_profit": incremental_profit,
        "margin_pct": chosen["margin_pct"],
        "campaign_cost": chosen["campaign_cost"],
        "baseline_net_profit": baseline_profit,
        "target_audience": audience,
        "channels": [
            "Owned digital channels",
            "Relevant product/category placements",
            "Email or app campaigns, if available and authorized",
        ],
        "campaign_ideas": campaign_ideas,
        "constraints_applied": constraints,
        "risks": risks,
        "next_actions": [
            "Confirm the proposed campaign with the business owner.",
            "Compare actual daily sales with predicted sales.",
            "Monitor gross margin, campaign cost, and remaining stock.",
            "Re-evaluate if actual performance differs materially.",
        ],
        "generation_method": "rules",
    }
    return strategy



def add_llm_marketing_plan(strategy):
    """Enrich marketing content using Groq or Gemini."""
    import os
    import json
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    if strategy.get("status") != "success":
        return strategy

    # Load .env values when the application starts.
    from dotenv import load_dotenv
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "openai/chat/completions"
        )
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    else:
        strategy["llm_status"] = "Unsupported LLM_PROVIDER; using rules."
        return strategy

    if not api_key:
        strategy["llm_status"] = (
            f"{provider} API key is missing; using rules."
        )
        return strategy

    prompt = f"""
You are a cautious retail marketing strategist for a synthetic-data demo.
Use only facts in the supplied strategy.
Do not invent financial figures, customer attributes, or business rules.
Do not change the recommended discount or any financial values.
Return valid JSON only with these keys:
campaign_name, executive_summary, target_audience, messaging,
channel_plan, experiment_plan, success_metrics.

target_audience, channel_plan, experiment_plan, and success_metrics
must be lists.

Approved strategy:
{json.dumps(strategy, ensure_ascii=False)}
"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Create grounded retail marketing plans. Return JSON only."
            },
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3
    }

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))

        content = result["choices"][0]["message"]["content"]
        llm_plan = json.loads(content)

        strategy["llm_plan"] = llm_plan
        strategy["generation_method"] = f"rules+{provider}"
        strategy["llm_status"] = "success"

    except HTTPError as exc:
        # Read the API's error response without exposing the API key.
        try:
            error_body = exc.read().decode("utf-8", errors="replace")
            error_data = json.loads(error_body)
            message = error_data.get("error", {}).get(
                "message", "No detailed error message returned"
            )
        except (ValueError, AttributeError):
            message = "Could not parse the API error response"

        strategy["llm_status"] = (
            f"{provider} HTTP {exc.code}: {message[:500]}. "
            "Using rules-based strategy."
        )

    except (URLError, TimeoutError, ValueError, KeyError, IndexError) as exc:
        strategy["llm_status"] = (
            f"{provider} unavailable ({type(exc).__name__}); "
            "using rules-based strategy."
        )

    return strategy



def generate_strategy(optimization, objective="balanced"):
    """Generate an objective-specific strategy from optimizer results."""
    objective = (objective or "balanced").strip().lower()

    if objective not in ALLOWED_OBJECTIVES:
        raise ValueError(
            "objective must be balanced, profit, sales, or inventory"
        )

    strategy = build_rule_based_strategy(optimization, objective)
    return add_llm_marketing_plan(strategy)
