
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "arco_campaigns.db"


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                product_id TEXT,
                product TEXT,
                brand TEXT,
                objective TEXT,
                discount_pct REAL,
                campaign_days INTEGER,
                expected_units REAL,
                expected_revenue REAL,
                expected_gross_profit REAL,
                expected_net_profit REAL,
                incremental_profit REAL,
                margin_pct REAL,
                campaign_cost REAL,
                status TEXT NOT NULL DEFAULT 'Generated',
                strategy_json TEXT NOT NULL
            )
        """)


def save_campaign(strategy):
    initialize_database()

    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO campaigns (
                created_at,
                product_id,
                product,
                brand,
                objective,
                discount_pct,
                campaign_days,
                expected_units,
                expected_revenue,
                expected_gross_profit,
                expected_net_profit,
                incremental_profit,
                margin_pct,
                campaign_cost,
                strategy_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            created_at,
            strategy.get("product_id"),
            strategy.get("product"),
            strategy.get("brand"),
            strategy.get("objective"),
            strategy.get("recommended_discount_pct"),
            strategy.get("campaign_days"),
            strategy.get("expected_units"),
            strategy.get("expected_revenue"),
            strategy.get("expected_gross_profit"),
            strategy.get("expected_net_profit"),
            strategy.get("incremental_profit"),
            strategy.get("margin_pct"),
            strategy.get("campaign_cost"),
            json.dumps(strategy, ensure_ascii=False, default=str)
        ))

        return cursor.lastrowid


def get_campaigns():
    initialize_database()

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT
                id,
                created_at,
                product_id,
                product,
                brand,
                objective,
                discount_pct,
                campaign_days,
                expected_units,
                expected_revenue,
                expected_gross_profit,
                expected_net_profit,
                incremental_profit,
                margin_pct,
                campaign_cost,
                status
            FROM campaigns
            ORDER BY id DESC
        """).fetchall()

        return [dict(row) for row in rows]
