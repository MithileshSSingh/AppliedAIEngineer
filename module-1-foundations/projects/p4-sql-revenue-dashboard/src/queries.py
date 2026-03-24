"""SQL query library for the Revenue Dashboard.

Each function takes a sqlite3 connection and returns a pandas DataFrame.
All revenue calculations use: quantity * unit_price * (1 - discount)
Only completed orders are included unless otherwise noted.
"""

import sqlite3
import pandas as pd


def monthly_revenue(conn: sqlite3.Connection) -> pd.DataFrame:
    """Monthly revenue, cost, profit, and MoM growth."""
    return pd.read_sql("""
    WITH monthly AS (
        SELECT
            strftime('%Y-%m', o.order_date) AS month,
            ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue,
            ROUND(SUM(oi.quantity * p.cost), 2) AS cost
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.status = 'completed'
        GROUP BY strftime('%Y-%m', o.order_date)
    )
    SELECT
        month,
        revenue,
        cost,
        ROUND(revenue - cost, 2) AS profit,
        ROUND((revenue - cost) / revenue * 100, 1) AS margin_pct,
        LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
        ROUND(
            (revenue - LAG(revenue) OVER (ORDER BY month))
            / LAG(revenue) OVER (ORDER BY month) * 100, 1
        ) AS growth_pct
    FROM monthly
    ORDER BY month
    """, conn)


def revenue_by_category(conn: sqlite3.Connection) -> pd.DataFrame:
    """Total revenue by product category."""
    return pd.read_sql("""
    SELECT
        p.category,
        ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue,
        ROUND(SUM(oi.quantity * p.cost), 2) AS cost,
        COUNT(DISTINCT o.order_id) AS orders
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.status = 'completed'
    GROUP BY p.category
    ORDER BY revenue DESC
    """, conn)


def revenue_by_region(conn: sqlite3.Connection) -> pd.DataFrame:
    """Total revenue by customer region."""
    return pd.read_sql("""
    SELECT
        c.region,
        ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue,
        COUNT(DISTINCT o.order_id) AS orders,
        COUNT(DISTINCT c.customer_id) AS customers
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.region
    ORDER BY revenue DESC
    """, conn)


def top_products(conn: sqlite3.Connection, n: int = 10) -> pd.DataFrame:
    """Top N products by revenue."""
    return pd.read_sql(f"""
    SELECT
        p.name AS product,
        p.category,
        ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue,
        SUM(oi.quantity) AS units_sold,
        COUNT(DISTINCT o.order_id) AS orders
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.status = 'completed'
    GROUP BY p.product_id, p.name, p.category
    ORDER BY revenue DESC
    LIMIT {n}
    """, conn)


def customer_ltv(conn: sqlite3.Connection) -> pd.DataFrame:
    """Customer lifetime value with RFM metrics."""
    return pd.read_sql("""
    WITH customer_orders AS (
        SELECT
            o.customer_id,
            o.order_date,
            SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS order_value
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        WHERE o.status = 'completed'
        GROUP BY o.order_id, o.customer_id, o.order_date
    )
    SELECT
        c.name,
        c.tier,
        c.region,
        COUNT(*) AS total_orders,
        ROUND(SUM(co.order_value), 2) AS lifetime_value,
        ROUND(AVG(co.order_value), 2) AS avg_order_value,
        MIN(co.order_date) AS first_order,
        MAX(co.order_date) AS last_order,
        CAST(JULIANDAY('2024-09-30') - JULIANDAY(MAX(co.order_date)) AS INTEGER) AS days_since_last
    FROM customer_orders co
    JOIN customers c ON co.customer_id = c.customer_id
    GROUP BY co.customer_id, c.name, c.tier, c.region
    ORDER BY lifetime_value DESC
    """, conn)


def executive_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """One-row executive summary with key metrics."""
    return pd.read_sql("""
    WITH order_values AS (
        SELECT
            o.order_id,
            o.customer_id,
            SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) AS revenue,
            SUM(oi.quantity * p.cost) AS cost
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        JOIN products p ON oi.product_id = p.product_id
        WHERE o.status = 'completed'
        GROUP BY o.order_id, o.customer_id
    )
    SELECT
        ROUND(SUM(revenue), 2) AS total_revenue,
        ROUND(SUM(revenue - cost), 2) AS total_profit,
        ROUND(SUM(revenue - cost) / SUM(revenue) * 100, 1) AS margin_pct,
        COUNT(*) AS total_orders,
        COUNT(DISTINCT customer_id) AS unique_customers,
        ROUND(AVG(revenue), 2) AS avg_order_value,
        ROUND(SUM(revenue) / COUNT(DISTINCT customer_id), 2) AS revenue_per_customer
    FROM order_values
    """, conn)
