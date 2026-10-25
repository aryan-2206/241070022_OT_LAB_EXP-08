"""Unit and Correctness Tests for Track H: DBMS Optimization."""

import sqlite3
import pytest

from src.track_h_dbms.database_setup import create_and_populate_tpch_db
from src.track_h_dbms.query_optimizer import JoinOptimizer
from src.track_h_dbms.index_selection import IndexSelector


def test_database_population_and_query_equivalence():
    """Validates that query results before and after adding indexes are strictly identical."""
    conn = create_and_populate_tpch_db(":memory:", seed=42)
    cur = conn.cursor()

    query = """
    SELECT c.c_mktsegment, COUNT(*), SUM(o.o_totalprice)
    FROM Customer c
    JOIN Orders o ON c.c_custkey = o.o_custkey
    GROUP BY c.c_mktsegment;
    """
    cur.execute(query)
    res_before = cur.fetchall()

    # Add indexes
    cur.execute("CREATE INDEX idx_c ON Customer(c_custkey, c_mktsegment);")
    cur.execute("CREATE INDEX idx_o ON Orders(o_custkey);")
    conn.commit()

    cur.execute(query)
    res_after = cur.fetchall()

    assert res_before == res_after, "Index modification altered query results!"
    assert len(res_before) > 0


def test_index_selection_budget_constraint():
    """Index selection ILP solver must strictly respect storage budget limit B."""
    candidates = [
        {"name": "idx1", "size_kb": 50.0, "benefit_score": 60.0, "sql": ""},
        {"name": "idx2", "size_kb": 40.0, "benefit_score": 50.0, "sql": ""},
        {"name": "idx3", "size_kb": 30.0, "benefit_score": 40.0, "sql": ""},
        {"name": "idx4", "size_kb": 70.0, "benefit_score": 75.0, "sql": ""},
    ]
    budget = 100.0
    selector = IndexSelector(candidates, storage_budget_kb=budget)
    res = selector.solve_ilp()

    assert res["total_size_kb"] <= budget, f"Storage budget exceeded! {res['total_size_kb']} > {budget}"
    assert res["total_benefit"] > 0


def test_system_r_dp_join_enumeration():
    """System-R DP must find valid permutation of tables with lowest estimated cost."""
    cards = {"R": 100, "S": 1000, "T": 500}
    v_vals = {("R", "a"): 100, ("S", "a"): 100, ("S", "b"): 500, ("T", "b"): 500}
    joins = {("R", "S"): ("a", "a"), ("S", "T"): ("b", "b")}

    opt = JoinOptimizer(cards, v_vals, joins)
    res = opt.solve_system_r_dp(["R", "S", "T"])

    assert len(res["order"]) == 3
    assert set(res["order"]) == {"R", "S", "T"}
    assert res["estimated_cost"] > 0
