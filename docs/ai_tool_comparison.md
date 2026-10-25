# AI Tool Comparison Study
## Empirical Evaluation of AI Coding Assistants on an Optimization Task
**Task Evaluated:** Mathematical Formulation and Dynamic Programming Implementation of System-R Relational Join-Order Optimization (Track H)  
**Tools Compared:** Gemini 3.8 Flash vs Claude 3.5 Sonnet  
**Date:** 2026-10-25  

---

### 1. Comparative Evaluation Matrix

| Metric | Gemini 3.8 Flash | Claude 3.5 Sonnet |
|---|---|---|
| **Mathematical Formulation Quality** | **Excellent (5/5):** Formulated exact Selinger cost model ($C = \text{I/O} + W \cdot \text{CPU}$) with explicit cardinality estimation formulas for equijoins $Card(R \bowtie S) = \frac{Card(R) \cdot Card(S)}{\max(V(R.a), V(S.a))}$. | **Very Good (4.5/5):** Clear cost model and recurrence relation, slightly less detailed on multi-column histogram estimation. |
| **Initial Code Correctness** | **Good (4/5):** Working subset dynamic programming memoization. Failed on empty relation edge cases and Cartesian cross-product penalty on disconnected tables. | **Good (4/5):** Correct memoization structure. Minor issue with permutation ordering of commutativity $R \bowtie S$ vs $S \bowtie R$. |
| **Hallucinations & Erroneous Inventions** | **Low (0 instances):** Used standard relational algebra and Python dictionary keys representing relation sets as frozensets. | **Low (0 instances):** Faithful to Selinger 1979 literature; standard dynamic programming over subsets. |
| **Prompt Iterations to Pass Tests** | **2 iterations:** Iteration 1 generated core DP; Iteration 2 added Cartesian product handling and SQLite AST query generation. | **2 iterations:** Iteration 1 generated DP; Iteration 2 refined plan tree serialization. |
| **Generation Speed & Response Latency** | **Very Fast (~3.2 seconds):** Near-instantaneous response for complete 120-line algorithm. | **Moderate (~8.5 seconds):** Thorough narrative explanation with code stream. |
| **Explanation Clarity & Viva Suitability** | **Concise & Direct:** Clear algorithmic steps, directly linked to time complexity $O(3^N)$ for subset enumeration. | **Expansive:** Provided textbook-like commentary on interesting orders and physical operators. |

---

### 2. Analysis & Synthesis

Both state-of-the-art models demonstrated deep understanding of dynamic programming over subsets ($O(3^N)$ subproblems for $N$ relations). Gemini 3.8 Flash was chosen as the primary coding engine due to its exceptional speed, clean integration into the terminal loop, and concise, error-free mathematical definitions. Claude 3.5 was utilized for cross-checking subtle edge cases such as join graph connectivity and interference graph coloring.
