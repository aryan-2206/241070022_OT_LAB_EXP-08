"""Track H: DBMS Optimization - Synthetic TPC-H Schema and Workload Generator.

Creates an SQLite database populated with synthetic TPC-H benchmark tables:
- Customer (500 rows)
- Orders (2,000 rows)
- LineItem (5,000 rows)
"""

import sqlite3
import numpy as np


def create_and_populate_tpch_db(db_path=":memory:", seed=42):
    """Creates tables and inserts synthetic relational dataset."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Drop existing tables
    cur.execute("DROP TABLE IF EXISTS LineItem;")
    cur.execute("DROP TABLE IF EXISTS Orders;")
    cur.execute("DROP TABLE IF EXISTS Customer;")

    # Table 1: Customer
    cur.execute("""
    CREATE TABLE Customer (
        c_custkey INTEGER PRIMARY KEY,
        c_name TEXT,
        c_mktsegment TEXT,
        c_acctbal REAL
    );
    """)

    # Table 2: Orders
    cur.execute("""
    CREATE TABLE Orders (
        o_orderkey INTEGER PRIMARY KEY,
        o_custkey INTEGER,
        o_orderstatus TEXT,
        o_totalprice REAL,
        o_orderdate TEXT,
        FOREIGN KEY(o_custkey) REFERENCES Customer(c_custkey)
    );
    """)

    # Table 3: LineItem
    cur.execute("""
    CREATE TABLE LineItem (
        l_orderkey INTEGER,
        l_linenumber INTEGER,
        l_quantity REAL,
        l_extendedprice REAL,
        l_discount REAL,
        l_shipdate TEXT,
        PRIMARY KEY (l_orderkey, l_linenumber),
        FOREIGN KEY(l_orderkey) REFERENCES Orders(o_orderkey)
    );
    """)

    np.random.seed(seed)
    segments = ["BUILDING", "AUTOMOBILE", "MACHINERY", "HOUSEHOLD", "FURNITURE"]
    statuses = ["O", "F", "P"]

    # Insert 500 Customers
    n_cust = 500
    cust_rows = []
    for c in range(1, n_cust + 1):
        seg = np.random.choice(segments)
        bal = float(np.random.uniform(-500.0, 10000.0))
        cust_rows.append((c, f"Customer#{c:06d}", seg, bal))
    cur.executemany("INSERT INTO Customer VALUES (?, ?, ?, ?);", cust_rows)

    # Insert 2,000 Orders
    n_orders = 2000
    order_rows = []
    for o in range(1, n_orders + 1):
        c_id = int(np.random.randint(1, n_cust + 1))
        st = np.random.choice(statuses)
        price = float(np.random.uniform(50.0, 5000.0))
        day = int(np.random.randint(1, 28))
        month = int(np.random.randint(1, 13))
        dt = f"2025-{month:02d}-{day:02d}"
        order_rows.append((o, c_id, st, price, dt))
    cur.executemany("INSERT INTO Orders VALUES (?, ?, ?, ?, ?);", order_rows)

    # Insert 5,000 LineItems
    n_lines = 5000
    line_rows = []
    line_counts = {}
    for _ in range(n_lines):
        o_id = int(np.random.randint(1, n_orders + 1))
        line_no = line_counts.get(o_id, 0) + 1
        line_counts[o_id] = line_no

        qty = float(np.random.randint(1, 50))
        ext_price = float(qty * np.random.uniform(10.0, 100.0))
        disc = float(np.random.uniform(0.0, 0.10))
        day = int(np.random.randint(1, 28))
        month = int(np.random.randint(1, 13))
        ship_dt = f"2025-{month:02d}-{day:02d}"
        line_rows.append((o_id, line_no, qty, ext_price, disc, ship_dt))
    cur.executemany("INSERT OR IGNORE INTO LineItem VALUES (?, ?, ?, ?, ?, ?);", line_rows)

    conn.commit()
    return conn
