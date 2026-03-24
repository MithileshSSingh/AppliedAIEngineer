"""Create and populate the SQLite database for the Revenue Dashboard project.

Usage: python setup_db.py
"""

import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path


def create_tables(conn: sqlite3.Connection):
    """Create the database schema."""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        tier TEXT CHECK(tier IN ('Gold', 'Silver', 'Bronze')),
        signup_date DATE,
        region TEXT
    );

    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT,
        price REAL,
        cost REAL
    );

    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        order_date DATE,
        status TEXT CHECK(status IN ('completed', 'pending', 'cancelled', 'returned')),
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );

    CREATE TABLE IF NOT EXISTS order_items (
        item_id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        unit_price REAL,
        discount REAL DEFAULT 0,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
    """)


def populate_data(conn: sqlite3.Connection, seed: int = 42):
    """Populate tables with realistic e-commerce data."""
    np.random.seed(seed)

    # --- Customers ---
    first_names = [
        'Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank', 'Grace', 'Hank',
        'Iris', 'Jack', 'Karen', 'Leo', 'Mona', 'Nick', 'Olivia', 'Paul',
        'Quinn', 'Rita', 'Sam', 'Tina', 'Uma', 'Vic', 'Wendy', 'Xander',
        'Yuki', 'Zara', 'Amit', 'Beth', 'Chris', 'Dana', 'Eli', 'Fay',
        'Gus', 'Helen', 'Ivan', 'Jill', 'Kurt', 'Lily', 'Max', 'Nina',
        'Omar', 'Priya', 'Roy', 'Sara', 'Tom', 'Ursula', 'Val', 'Walt',
        'Xena', 'Yolanda'
    ]
    tiers = ['Gold', 'Silver', 'Bronze']
    regions = ['North', 'South', 'East', 'West']

    for i, name in enumerate(first_names, 1):
        signup = f"2022-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}"
        conn.execute(
            "INSERT OR IGNORE INTO customers VALUES (?, ?, ?, ?, ?, ?)",
            (i, name, f"{name.lower()}@email.com",
             np.random.choice(tiers, p=[0.2, 0.35, 0.45]),
             signup, np.random.choice(regions))
        )

    # --- Products ---
    products = [
        (1, 'Laptop Pro 15', 'Electronics', 1299.99, 780),
        (2, 'Wireless Mouse', 'Peripherals', 29.99, 10),
        (3, 'Mech Keyboard', 'Peripherals', 149.99, 55),
        (4, '4K Monitor 27"', 'Electronics', 449.99, 220),
        (5, 'USB-C Hub', 'Accessories', 59.99, 18),
        (6, 'Webcam HD', 'Peripherals', 89.99, 30),
        (7, 'Standing Desk', 'Furniture', 399.99, 180),
        (8, 'Ergo Chair', 'Furniture', 299.99, 130),
        (9, 'Headset Pro', 'Accessories', 79.99, 25),
        (10, 'Laptop Stand', 'Accessories', 49.99, 15),
        (11, 'Desk Lamp LED', 'Furniture', 39.99, 12),
        (12, 'Cable Organizer', 'Accessories', 14.99, 4),
        (13, 'Monitor Arm', 'Accessories', 89.99, 35),
        (14, 'Tablet Pro', 'Electronics', 599.99, 350),
        (15, 'Noise-Cancel Buds', 'Accessories', 129.99, 45),
    ]
    conn.executemany("INSERT OR IGNORE INTO products VALUES (?, ?, ?, ?, ?)", products)

    # --- Orders & Items (2 years of data) ---
    statuses = ['completed'] * 7 + ['pending', 'cancelled', 'returned']
    order_id = 1

    for year in [2023, 2024]:
        for month in range(1, 13):
            if year == 2024 and month > 9:
                break
            # Seasonal variation: more orders in Q4
            base_orders = 30 if month >= 10 else 20
            n_orders = np.random.randint(base_orders - 5, base_orders + 10)

            for _ in range(n_orders):
                cust_id = np.random.randint(1, len(first_names) + 1)
                day = np.random.randint(1, 29)
                date = f"{year}-{month:02d}-{day:02d}"
                status = np.random.choice(statuses)

                conn.execute(
                    "INSERT INTO orders VALUES (?, ?, ?, ?)",
                    (order_id, cust_id, date, status)
                )

                # 1-4 items per order
                n_items = np.random.randint(1, 5)
                prod_ids = np.random.choice(range(1, 16), n_items, replace=False)
                for pid in prod_ids:
                    qty = np.random.randint(1, 6)
                    price = products[pid - 1][3]
                    discount = np.random.choice([0, 0, 0, 0, 0.05, 0.10, 0.15, 0.20])
                    conn.execute(
                        "INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (order_id, int(pid), qty, price, discount)
                    )
                order_id += 1

    conn.commit()


def main():
    db_path = Path(__file__).parent.parent / "data" / "store.db"
    db_path.parent.mkdir(exist_ok=True)

    # Remove existing database to start fresh
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    create_tables(conn)
    populate_data(conn)

    # Print summary
    for table in ['customers', 'products', 'orders', 'order_items']:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    print(f"\nDatabase created at: {db_path}")
    conn.close()


if __name__ == "__main__":
    main()
