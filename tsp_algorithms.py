#Code by Diego Lopez Yse, Medium 
# https://lopezyse.medium.com/graphs-solving-the-travelling-salesperson-problem-tsp-in-python-54ec2b315977
# Some modifications were made for analysis and specific requirements.
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import time
import random
import numpy as np
import pandas as pd
import math

# Function to create a graph with a specified number of nodes
def create_graph(num_nodes, seed=42):
    random.seed(seed)
    G = nx.Graph()
    cities = {i: (random.uniform(0, 10), random.uniform(0, 10)) for i in range(num_nodes)}
    for city, pos in cities.items():
        G.add_node(city, pos=pos)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            x1, y1 = cities[i]
            x2, y2 = cities[j]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            G.add_edge(i, j, weight=distance)
    return G, cities

def route_distance(G, route):
    total = 0
    for i in range(len(route) - 1):
        total += G[route[i]][route[i + 1]]['weight']
    total += G[route[-1]][route[0]]['weight']
    return total

def solve_tsp_brute_force(G):
    num_nodes = G.number_of_nodes()
    other_cities = list(range(1, num_nodes))
    all_routes = itertools.permutations(other_cities)
    best_route = None
    min_distance = float('inf')
    bf_ops = 0
    for perm in all_routes:
        bf_ops+=1
        current_route = [0] + list(perm)
        current_distance = route_distance(G, current_route)
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = current_route
    return best_route, min_distance, bf_ops

def solve_tsp_held_karp(G):
    n = G.number_of_nodes()
    dp = {}
    hk_ops = 0
    for i in range(1, n):
        dp[(1 << i, i)] = (G[0][i]['weight'], 0)
    for size in range(2, n):
        for subset in combinations(range(1, n), size):
            bits = sum(1 << i for i in subset)
            for end in subset:
                prev_bits = bits & ~(1 << end)
                min_cost = float('inf')
                min_prev = None
                for prev in subset:
                    if prev == end or (prev_bits, prev) not in dp:
                        continue
                    hk_ops+=1
                    cost = dp[(prev_bits, prev)][0] + G[prev][end]['weight']
                    if cost < min_cost:
                        min_cost = cost
                        min_prev = prev
                dp[(bits, end)] = (min_cost, min_prev)
    all_bits = (1 << n) - 2
    min_cost = float('inf')
    final_end = None
    for end in range(1, n):
        if (all_bits, end) in dp:
            hk_ops+=1
            cost = dp[(all_bits, end)][0] + G[end][0]['weight']
            if cost < min_cost:
                min_cost = cost
                final_end = end
    route = [final_end]
    current_bits = all_bits
    while current_bits:
        last = route[-1]
        cost, prev = dp[(current_bits, last)]
        route.append(prev)
        current_bits &= ~(1 << last)
    route = route[::-1]
    route[0] = 0
    return route, min_cost, hk_ops

def plot_tsp(G, route, distance, title, time_taken):
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    best_edges = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
    best_edges.append((route[-1], route[0]))
    nx.draw_networkx_edges(G, pos, edgelist=best_edges, edge_color='red', width=2)
    edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    plt.title(f"{title}\nRoute: {route}, Distance: {distance:.2f}, Time: {time_taken:.6f}s", fontsize=14)
    plt.axis('off')
    plt.show()

def run_tsp_experiments_df(
    ns,
    trials_per_n: int = 3,
    low: int = 1,
    high: int = 20,
    seed_base: int = 123,
    brute_force_max_n: int = 10
) -> pd.DataFrame:
    rows = []

    for n in ns:
        for t in range(trials_per_n):
            seed = seed_base + 1000*n + t
            dist = random_int_dist_matrix(n, low=low, high=high, seed=seed)
            G = graph_from_matrix(dist)

            # Held–Karp
            t0 = time.perf_counter()
            hk_route, hk_cost, hk_ops = solve_tsp_held_karp(G)
            hk_time = time.perf_counter() - t0

            # Brute Force (only small n)
            bf_cost = bf_ops = bf_time = None
            if n <= brute_force_max_n:
                t0 = time.perf_counter()
                bf_route, bf_cost, bf_ops = solve_tsp_brute_force(G)
                bf_time = time.perf_counter() - t0

            rows.append({
                "n": n,
                "trial": t,
                "seed": seed,

                "bf_time_s": bf_time,
                "bf_ops": bf_ops,

                "hk_time_s": hk_time,
                "hk_ops": hk_ops,
            })

    return pd.DataFrame(rows)

def print_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("n", as_index=False)
          .agg(
              bf_time_mean=("bf_time_s", "mean"),
              bf_ops_mean=("bf_ops", "mean"),
              hk_time_mean=("hk_time_s", "mean"),
              hk_ops_mean=("hk_ops", "mean"),
          )
          .sort_values("n")
    )

    # Theory columns for analysis
    summary["bf_ops_theory"] = summary["n"].apply(lambda n: math.factorial(n-1) if n <= summary["n"].max() else None)
    summary["hk_ops_theory"] = summary["n"].apply(lambda n: (2**n) * (n**2))
    summary["hk_ratio"] = summary["hk_ops_mean"] / summary["hk_ops_theory"]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n=== Summary (mean over trials) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    return summary

def plot_ops_by_n(summary: pd.DataFrame):
    # brute force might be NaN for large n
    plot_df = summary[["n", "bf_ops_mean", "hk_ops_mean"]].copy()

    ns = plot_df["n"].to_list()
    x = range(len(ns))
    width = 0.4

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], plot_df["bf_ops_mean"], width, label="Brute Force ops (mean)")
    plt.bar([i + width/2 for i in x], plot_df["hk_ops_mean"], width, label="Held–Karp ops (mean)")

    plt.xticks(list(x), ns)
    plt.xlabel("n (number of cities)")
    plt.ylabel("Basic operations (mean over trials)")
    plt.title("Basic Ops vs n (Brute Force vs Held–Karp)")
    plt.yscale("log")
    plt.grid(True, which="both", axis="y", ls="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def random_int_dist_matrix(n: int, low: int = 1, high: int = 20, seed: int | None = None):
    """Symmetric integer distances, diagonal = 0."""
    rng = random.Random(seed)
    dist = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            w = rng.randint(low, high)
            dist[i][j] = w
            dist[j][i] = w
    return dist


import networkx as nx

def graph_from_matrix(dist):
    n = len(dist)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=dist[i][j])  # assumes symmetric
    return G

def run_tsp_table(
    ns,
    trials_per_n: int = 3,
    low: int = 1,
    high: int = 20,
    seed_base: int = 123,
    brute_force_max_n: int = 10
):
    """
    For each n in ns:
      - generate 'trials_per_n' random matrices
      - run HK always
      - run brute force only if n <= brute_force_max_n (factorial explosion)
    Returns: list of rows (dicts)
    """
    rows = []

    for n in ns:
        for t in range(trials_per_n):
            seed = seed_base + 1000*n + t
            dist = random_int_dist_matrix(n, low=low, high=high, seed=seed)
            G = graph_from_matrix(dist)

            # --- Held–Karp ---
            t0 = time.perf_counter()
            hk_route, hk_cost, hk_ops = solve_tsp_held_karp(G)
            hk_time = time.perf_counter() - t0

            # --- Brute Force (only for small n) ---
            bf_route = bf_cost = bf_ops = bf_time = None
            if n <= brute_force_max_n:
                t0 = time.perf_counter()
                bf_route, bf_cost, bf_ops = solve_tsp_brute_force(G)
                bf_time = time.perf_counter() - t0

            rows.append({
                "n": n,
                "trial": t,
                "seed": seed,

                "bf_cost": bf_cost,
                "bf_time_s": bf_time,
                "bf_ops": bf_ops,
                "bf_ops_theory": math.factorial(n-1),

                "hk_cost": hk_cost,
                "hk_time_s": hk_time,
                "hk_ops": hk_ops,
                "hk_ops_over_2n_n2": hk_ops / ((2**n) * (n**2)),  
            })

    return rows

def to_dataframe(rows):
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Brute force grows as (n-1)!
    ns = [5, 6, 7, 8, 9, 10]   
    trials_per_n = 3
    brute_force_max_n = 10

    df = run_tsp_experiments_df(
        ns=ns,
        trials_per_n=trials_per_n,
        low=1,
        high=20,
        seed_base=123,
        brute_force_max_n=brute_force_max_n
    )

    summary = print_summary_table(df)
    plot_ops_by_n(summary)