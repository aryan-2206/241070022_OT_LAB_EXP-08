# Track H Prompt & Engineering Audit Log
## DBMS Optimization (TPC-H Schema, System-R DP Join Ordering, Index Knapsack ILP)
**Track:** Track H  
**AI Assistant:** Gemini 3.8 Flash  
**Date:** 2026-10-25  

---

### Original Prompt
```text
Implement Track H: DBMS Optimization on an SQLite database:
1. Synthetic TPC-H relational schema: Customer (500 rows), Orders (2000 rows), LineItem (5000 rows).
2. Query rewriting and Join-order selection: Implement Selinger cost model and System-R Dynamic Programming over subsets of tables.
3. Formulate Index Selection as an optimization problem: 0/1 Knapsack ILP maximizing query workload benefit subject to a strict disk storage budget (130 KB). Compare with Greedy selection.
4. Apply the chosen index set to SQLite, verify query plan transition using EXPLAIN QUERY PLAN (replacing SCAN with SEARCH USING COVERING INDEX).
5. Verify bit-for-bit query result equivalence before and after indexing.
```

### AI Response Summary
The AI implemented `database_setup.py` seeding TPC-H tables, `JoinOptimizer` performing subset memoization for join tree selection, and `IndexSelector` formulating the Knapsack ILP via `scipy.optimize.milp`. It ran queries and verified `EXPLAIN QUERY PLAN` plan changes.

### Revision & Refinement
Formulated candidate covering indices (`Orders(o_custkey, o_orderdate)`) and evaluated multi-query workloads across varying market segments to accurately observe performance gains and verify identical result row sets.

### Verification Notes
- Unit test `test_database_population_and_query_equivalence` confirmed that index additions preserve identical SQL row outputs.
- `test_index_selection_budget_constraint` confirmed selected index size ($127.0$ KB) strictly adhered to the $130.0$ KB budget.
- EXPLAIN QUERY PLAN verified the full scan of the 2,000-row `Orders` table was eliminated and replaced with binary B-tree searches.

### Linked Commit
Milestone 9: `[AI:gemini-3.8] Track H: DBMS optimization, TPC-H schema, System-R DP, and index selection ILP`
