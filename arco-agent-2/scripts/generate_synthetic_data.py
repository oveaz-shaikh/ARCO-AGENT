"""Generate reproducible synthetic retail data for ARCO Agent 2.0.

All values and rules are fictional and for software demonstration only.
Run from the project root: python -m scripts.generate_synthetic_data
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    rng = np.random.default_rng(SEED)
    n_products, n_customers, n_stores = 100, 2000, 5
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

    brands = ["Demo Brand A", "Demo Brand B", "Demo Brand C", "Demo Brand D"]
    categories = ["Footwear", "Apparel", "Beauty", "Accessories", "Home"]
    products = pd.DataFrame({
        "product_id": [f"P{i:04d}" for i in range(1, n_products + 1)],
        "brand": rng.choice(brands, n_products),
        "category": rng.choice(categories, n_products),
        "product_name": [f"Demo Product {i:03d}" for i in range(1, n_products + 1)],
    })
    products["unit_cost"] = np.round(rng.uniform(8, 90, n_products), 2)
    products["base_price"] = np.round(
        products["unit_cost"] * rng.uniform(1.8, 3.2, n_products), 2
    )
    products["launch_date"] = pd.to_datetime(
        rng.choice(dates, n_products)
    ).strftime("%Y-%m-%d")
    products["base_daily_demand"] = np.round(rng.uniform(0.8, 4.0, n_products), 3)

    customers = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(1, n_customers + 1)],
        "signup_date": pd.to_datetime(
            rng.choice(dates, n_customers)
        ).strftime("%Y-%m-%d"),
        "country": rng.choice(["Kuwait", "UAE", "Saudi Arabia", "Bahrain"], n_customers),
        "preferred_category": rng.choice(categories, n_customers),
    })
    stores = pd.DataFrame({
        "store_id": [f"S{i:02d}" for i in range(1, n_stores + 1)],
        "country": ["Kuwait", "UAE", "Saudi Arabia", "Bahrain", "Kuwait"],
        "city": ["Demo City 1", "Demo City 2", "Demo City 3", "Demo City 4", "Demo City 5"],
        "store_type": ["Mall", "Mall", "Street", "Mall", "Outlet"],
    })

    # Generate daily sales with a clear, moderate synthetic promotion effect.
    # Discount levels occur often enough for the model to learn their relationship
    # with demand. This is a simulated relationship, not real-world causal evidence.
    rows = []
    daily_sales_rows = []
    txn_num = 1
    discount_levels = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)
    discount_probabilities = np.array([0.35, 0.12, 0.13, 0.13, 0.12, 0.09, 0.06])

    for p in products.itertuples(index=False):
        for s in stores.itertuples(index=False):
            stock_remaining = int(rng.integers(60, 181))
            for current_date in dates:
                seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * current_date.dayofyear / 365.25)
                weekend = 1.15 if current_date.dayofweek >= 4 else 1.0
                discount = float(rng.choice(discount_levels, p=discount_probabilities))

                # Fictional demand response: up to a 75% uplift at 30% discount.
                # A stochastic Poisson draw adds realistic variation around the mean.
                promo_uplift = 1.0 + 0.025 * discount
                expected = float(p.base_daily_demand) * seasonal * weekend * promo_uplift
                expected = max(expected, 0.05)

                available_at_start = stock_remaining
                sold = int(min(rng.poisson(expected), available_at_start))
                selling_price = round(float(p.base_price) * (1.0 - discount / 100.0), 2)

                daily_sales_rows.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "product_id": p.product_id,
                    "store_id": s.store_id,
                    "units_sold": sold,
                    "discount_pct": discount,
                    "unit_price": selling_price,
                    "unit_cost": float(p.unit_cost),
                    "stock_available": available_at_start,
                    "month": current_date.month,
                    "day_of_week": current_date.dayofweek,
                })

                # Create exactly one transaction record per unit sold.
                if sold > 0:
                    cust_ids = rng.choice(
                        customers["customer_id"].to_numpy(), size=sold, replace=True
                    )
                    for customer_id in cust_ids:
                        rows.append({
                            "transaction_id": f"T{txn_num:08d}",
                            "date": current_date.strftime("%Y-%m-%d"),
                            "customer_id": customer_id,
                            "product_id": p.product_id,
                            "store_id": s.store_id,
                            "units": 1,
                            "unit_price": selling_price,
                            "discount_pct": discount,
                            "revenue": selling_price,
                            "unit_cost": float(p.unit_cost),
                        })
                        txn_num += 1
                    stock_remaining -= sold

                # Fictional monthly replenishment after the day's sales.
                if current_date.day == 1:
                    stock_remaining += int(rng.integers(10, 50))

    transactions = pd.DataFrame(rows)
    if transactions.empty:
        raise RuntimeError("Synthetic generator created no transactions.")

    inventory = []
    for p in products.itertuples(index=False):
        for s in stores.itertuples(index=False):
            inventory.append({
                "snapshot_date": dates[-1].strftime("%Y-%m-%d"),
                "store_id": s.store_id,
                "product_id": p.product_id,
                "stock_quantity": int(rng.integers(0, 160)),
                "stock_age_days": int(rng.integers(0, 240)),
            })

    campaigns = []
    channels = ["Email", "App Notification", "Social"]
    offer_types = ["Discount", "Bundle", "Featured Product"]
    for i in range(1, 61):
        sample = transactions.sample(min(80, len(transactions)), random_state=SEED + i)
        units = int(sample["units"].sum())
        impressions = int(rng.integers(1000, 12000))
        clicks = int(rng.integers(
            max(1, int(impressions * 0.02)), max(2, int(impressions * 0.18))
        ))
        start_date = pd.Timestamp(rng.choice(dates))
        duration = int(rng.integers(1, 15))
        end_date = min(start_date + pd.Timedelta(days=duration), dates[-1])
        campaigns.append({
            "campaign_id": f"CAM{i:04d}",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "segment_id": f"SEG{int(rng.integers(1, 5)):02d}",
            "product_id": str(rng.choice(products["product_id"])),
            "channel": str(rng.choice(channels)),
            "offer_type": str(rng.choice(offer_types)),
            "discount_pct": int(rng.choice([0, 5, 10, 15, 20, 25, 30])),
            "campaign_cost": round(float(rng.uniform(100, 2500)), 2),
            "impressions": impressions,
            "clicks": clicks,
            "orders": int(rng.integers(5, 100)),
            "units_sold": units,
        })

    # Fictional constraints only; these are not Alshaya policies.
    rules = pd.DataFrame({
        "brand": brands,
        "max_discount_pct": [30, 25, 35, 20],
        "min_margin_pct": [25, 30, 20, 25],
        "min_stock_age_days": [60, 45, 30, 60],
        "campaign_budget_limit": [5000, 4000, 6000, 3500],
    })

    datasets = {
        "daily_product_sales.csv": pd.DataFrame(daily_sales_rows),
        "products.csv": products.drop(columns=["base_daily_demand"]),
        "customers.csv": customers,
        "stores.csv": stores,
        "transactions.csv": transactions,
        "inventory.csv": pd.DataFrame(inventory),
        "campaigns.csv": pd.DataFrame(campaigns),
        "brand_rules.csv": rules,
    }
    for filename, dataframe in datasets.items():
        dataframe.to_csv(OUT / filename, index=False)
        print(f"Wrote {filename}: {len(dataframe):,} rows")

    daily = datasets["daily_product_sales.csv"]
    print("\nDiscount distribution in daily sales:")
    print(daily["discount_pct"].value_counts(normalize=True).sort_index().round(3).to_string())
    print(f"\nSynthetic data ready at: {OUT}")
    print("Reminder: the promotion uplift is an explicit fictional assumption, not real-world evidence.")


if __name__ == "__main__":
    main()
