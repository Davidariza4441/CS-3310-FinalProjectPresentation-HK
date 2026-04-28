# TSP Analysis: Brute Force vs Held–Karp (Dynamic Programming)

This project compares two exact algorithms for the Traveling Salesman Problem (TSP):

- **Brute Force (Permutation Search)**: tries every possible tour that starts at city `0`.
- **Held–Karp (Dynamic Programming / Bitmask DP)**: uses subset DP to compute the optimal tour in exponential time but much faster than brute force.

The code also tracks a simple **basic operations (ops)** counter for each algorithm to empirically compare growth rates.

---

## Problem Setup

We work with a **complete weighted graph** where:

- Nodes represent cities.
- Edge weights represent travel cost/distance.
- The tour must start at city `0`, visit every city exactly once, and return to city `0`.

The graph can be created in two ways:

1. **Random Euclidean graph** using random (x,y) coordinates (float weights).
2. **Random integer distance matrix** (symmetric, diagonal = 0) converted to a graph (integer weights).

For the analysis section in `__main__`, the code uses **random integer matrices** so the weights are simple integers.

---

## Key Files / Functions

### `create_graph(num_nodes, seed=42)`
Creates a random Euclidean complete graph using random coordinates per node and Euclidean distance as weights.
Returns:
- `G` (NetworkX graph)
- `cities` dictionary of node -> (x,y) positions

> Note: This is mainly used for visual demonstrations and plotting.

---

### `route_distance(G, route)`
Computes the total cost of a tour:
- adds weight of each consecutive edge in the route
- adds return edge from last node back to the first node

---

## Algorithms

### 1) Brute Force: `solve_tsp_brute_force(G)`
Enumerates all permutations of cities `[1..n-1]` and builds tours of the form:
`[0] + permutation`

Returns:
- `best_route`
- `min_distance`
- `bf_ops`

**Ops metric (Brute Force)**  
`bf_ops` counts **how many tours (permutations) were evaluated**.  
For `n` cities, the theoretical number of tours is `(n-1)!`.

---

### 2) Held–Karp DP: `solve_tsp_held_karp(G)`
Implements the Held–Karp algorithm using subset DP:

- DP state stored in a dictionary `dp[(mask, end)] = (cost, prev)`
- `mask` is a bitmask representing the subset of visited cities (excluding city 0)
- Transition tries all possible predecessor cities for each state
- The tour is closed by returning from the final endpoint back to city `0`
- Route reconstruction is performed using stored `prev` pointers.

Returns:
- `route`
- `min_cost`
- `hk_ops`

**Ops metric (Held–Karp)**  
`hk_ops` counts **candidate predecessor evaluations** inside the recurrence (dominant work).  
This aligns with the known time complexity: **O(2^n · n^2)**.

---
