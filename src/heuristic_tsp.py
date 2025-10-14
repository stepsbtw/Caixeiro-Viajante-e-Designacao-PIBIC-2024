#!/usr/bin/env python3
"""Heuristic TSP solvers using libraries (preferred) with light fallback code.

Features:
 - Parse AMPL-style TSP .dat (set NODES, param dist)
 - Construction methods:
     * OR-Tools (if available) with PATH_CHEAPEST_ARC / random start
     * Christofides heuristic ( --method christofides ) using networkx (exact matching) or greedy fallback
     * Simple nearest-neighbor / random + 2-opt fallback when no OR-Tools
 - Local search options (--ls): 2opt, 3opt, lk (Lin-Kernighan via OR-Tools flag)
 - Internal numpy 2-opt (and light LK-like variable-depth fallback) when OR-Tools absent
 - Batch run over directory

CLI examples:
    python heuristic_tsp.py --pattern gr21.dat --ls 2opt
    python heuristic_tsp.py --all --limit 60 --ls 3opt
    python heuristic_tsp.py --pattern gr21.dat --method christofides --ls 2opt

Dependencies (auto-detected):
    - ortools (recommended) for high-quality metaheuristic search
    - networkx (optional) for full Christofides (perfect matching). Without it a greedy matching fallback is used.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import time
import os as _os  # for environment tweaks
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import numpy as np

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2  # type: ignore
    _HAS_ORTOOLS = True
except Exception:  # pragma: no cover - import guard
    _HAS_ORTOOLS = False
_WARNED_CHR_FALLBACK = False  # internal flag for one-time notice

try:  # optional for exact matching in Christofides
    import networkx as nx  # type: ignore
    _HAS_NETWORKX = True
except Exception:  # pragma: no cover - import guard
    _HAS_NETWORKX = False


@dataclass
class TSPSolution:
    tour: List[int]            # sequence of node indices (0-based) ending w/o repeat
    cost: float
    runtime: float
    method: str
    improvements: int = 0


def parse_tsp_dat(path: str) -> np.ndarray:
    """Parse AMPL .dat with 'set NODES' and 'param dist :' matrix."""
    with open(path, 'r') as f:
        content = f.read().splitlines()
    rows: List[List[int]] = []
    in_matrix = False
    header_consumed = False
    for line in content:
        line = line.strip()
        if not line or line.startswith('#'):  # comments/empty
            continue
        if line.startswith('param dist'):
            in_matrix = True
            continue
        if in_matrix:
            if line.startswith(';'):
                break
            parts = line.split()
            # First line after 'param dist :' is the column header; skip once
            if not header_consumed:
                # Expect something like: 1 2 3 ... :
                header_consumed = True
                continue
            if parts[0].isdigit():
                # row line begins with index followed by distances
                # filter potential trailing comments or semicolons
                numeric_tokens = []
                for tok in parts[1:]:
                    if tok.startswith('#'):
                        break
                    if tok == ';':
                        break
                    numeric_tokens.append(tok)
                rows.append([int(x) for x in numeric_tokens])
    dist = np.array(rows, dtype=float)
    if dist.shape[0] != dist.shape[1]:
        raise ValueError(f"Distance matrix not square in {path}: {dist.shape}")
    return dist


def build_routing_solution(dist: np.ndarray, time_limit: int, ls: str, seed: int, start_strategy: str) -> TSPSolution:
    if not _HAS_ORTOOLS:
        raise RuntimeError("OR-Tools not available for routing solution.")
    # Safety: restrict threads (some environments segfault with high concurrency)
    _os.environ.setdefault('ORTOOLS_NUM_THREADS', '1')
    n = dist.shape[0]
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index: int, to_index: int) -> int:
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(dist[f, t])

    transit_idx = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, start_strategy
    )
    # Local search operators
    if ls == 'none':
        # No local search - just use first solution strategy
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
        search_params.time_limit.seconds = 1  # Minimal time since no local search needed
    else:
        # Use automatic local search that stops when improvement plateaus
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC
        search_params.time_limit.seconds = time_limit
    search_params.log_search = False
    search_params.use_full_propagation = False
    
    # Set solution limit to stop early when good solution found
    try:
        search_params.solution_limit = 1  # Stop after finding first improvement
    except AttributeError:
        pass  # Some versions don't support this
    
    # Track if OR-Tools LK is available
    ortools_lk_available = False
    
    # For controlling operators we rely on defaults; adjust per selection
    if ls == '2opt':
        # Basic 2-opt configuration - use default operators
        pass
    elif ls == '3opt':
        # Enable 3-opt if available in this version
        try:
            search_params.use_three_opt = True
        except AttributeError:
            pass  # fallback to default operators
    elif ls == 'lk':
        # Try to use OR-Tools Lin-Kernighan if available
        try:
            search_params.use_lin_kernighan = True
            ortools_lk_available = True
        except AttributeError:
            pass  # will fall back to our LK implementation after OR-Tools call
    random.seed(seed)
    # Note: Some OR-Tools versions don't support setting random_seed directly
    # The random.seed() call above should affect OR-Tools internal randomness

    start_t = time.time()
    solution = routing.SolveWithParameters(search_params)
    runtime = time.time() - start_t
    if solution is None:
        raise RuntimeError("Routing solver failed to find a solution within time limit")

    index = routing.Start(0)
    tour: List[int] = []
    route_cost = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        tour.append(node)
        prev = index
        index = solution.Value(routing.NextVar(index))
        if not routing.IsEnd(index):
            route_cost += dist[node, manager.IndexToNode(index)]
        else:
            # return to start
            route_cost += dist[node, tour[0]]
    
    # If LK was requested but OR-Tools LK wasn't available, apply our LK implementation
    if ls == 'lk' and not ortools_lk_available:
        lk_start_t = time.time()
        improved_tour, improved_cost, lk_iters = lk_like_improvement(tour, dist)
        lk_runtime = time.time() - lk_start_t
        total_runtime = runtime + lk_runtime
        return TSPSolution(tour=improved_tour, cost=improved_cost, runtime=total_runtime, method=f"ortools_{ls}_fallback", improvements=lk_iters)
    
    return TSPSolution(tour=tour, cost=route_cost, runtime=runtime, method=f"ortools_{ls}")


def two_opt_numpy(tour: List[int], dist: np.ndarray, max_iters: int = 2000) -> Tuple[List[int], float, int]:
    """Simple but efficient 2-opt improvement (first-improvement)."""
    n = len(tour)
    best = tour[:]
    best_cost = tour_cost(best, dist)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, n - 2):
            bi = best[i - 1]
            ci = best[i]
            for k in range(i + 1, n - 1):
                dj = best[k]
                ej = best[(k + 1) % n]
                delta = (dist[bi, dj] + dist[ci, ej]) - (dist[bi, ci] + dist[dj, ej])
                if delta < -1e-9:
                    best[i : k + 1] = reversed(best[i : k + 1])
                    best_cost += delta
                    improved = True
                    break
            if improved:
                break
    return best, best_cost, iters


def lk_like_improvement(tour: List[int], dist: np.ndarray, max_passes: int = 20) -> Tuple[List[int], float, int]:
    """Very lightweight Lin-Kernighan style variable-depth search.

    Not a full LK: repeatedly apply improving 2-opt moves; then attempt
    chained sequences by allowing a temporary worsening first move and
    looking for net gain over a short chain. Keeps it cheap for medium n.
    Returns (tour, cost, improvement_passes).
    """
    n = len(tour)
    best = tour[:]
    best_cost = tour_cost(best, dist)
    passes = 0
    improved = True
    while improved and passes < max_passes:
        improved = False
        passes += 1
        # baseline 2-opt first-improvement sweep
        best, best_cost, _ = two_opt_numpy(best, dist, max_iters=1_000)
        # variable-depth (depth 2-3) exploration
        for i in range(1, n - 3):
            for k in range(i + 2, n - 1):
                # First move (might be slightly uphill)
                a, b = best[i - 1], best[i]
                c, d = best[k], best[(k + 1) % n]
                delta1 = (dist[a, c] + dist[b, d]) - (dist[a, b] + dist[c, d])
                # allow limited uphill
                if delta1 < 5.0:  # threshold
                    new_tour = best[:]
                    new_tour[i : k + 1] = reversed(new_tour[i : k + 1])
                    # Second improvement move (2-opt) on modified tour
                    new_tour2, new_cost2, _ = two_opt_numpy(new_tour, dist, max_iters=200)
                    if new_cost2 + 1e-9 < best_cost:
                        best = new_tour2
                        best_cost = new_cost2
                        improved = True
                        break
            if improved:
                break
    return best, best_cost, passes


def tour_cost(tour: List[int], dist: np.ndarray) -> float:
    return float(sum(dist[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))))


def random_tour(n: int, seed: int) -> List[int]:
    r = list(range(n))
    random.seed(seed)
    random.shuffle(r)
    return r


def christofides_tour(dist: np.ndarray) -> List[int]:
    """Compute a tour using the Christofides heuristic.

    Steps:
      1. Minimum Spanning Tree (MST)
      2. Find vertices of odd degree in MST
      3. Minimum weight perfect matching on induced subgraph
      4. Combine MST + matching -> multigraph of all even degrees
      5. Eulerian tour -> shortcut repeated vertices

    Requires networkx for convenience; provides a lightweight fallback
    greedy matching if networkx not installed.
    """
    n = dist.shape[0]
    if n <= 2:
        return list(range(n))
    global _WARNED_CHR_FALLBACK
    if _HAS_NETWORKX:
        # Prefer ready-made NetworkX Christofides / TSP helper if present.
        try:
            from networkx.algorithms import approximation as approx  # type: ignore
            G = nx.Graph()
            for i in range(n):
                for j in range(i + 1, n):
                    w = float(dist[i, j])
                    G.add_edge(i, j, weight=w)
            tour: List[int]
            if hasattr(approx, 'christofides'):
                tour = approx.christofides(G, weight='weight')  # returns list of nodes
            else:
                # Fallback to generic TSP interface with method='christofides'
                tsp_nodes = approx.traveling_salesman_problem(G, weight='weight', cycle=True, method='christofides')
                # traveling_salesman_problem may return a cycle list (with start duplicated at end)
                if len(tsp_nodes) > 1 and tsp_nodes[0] == tsp_nodes[-1]:
                    tsp_nodes = tsp_nodes[:-1]
                tour = list(tsp_nodes)
            # Ensure length n (if graph complete and algorithm works this should hold)
            if len(tour) != n:
                # fallback to set completion ordering
                seen = set(tour)
                tour.extend([k for k in range(n) if k not in seen])
            return tour
        except Exception as e:  # pragma: no cover - defensive
            if not _WARNED_CHR_FALLBACK:
                print(f"[warn] NetworkX Christofides helper failed ({e}); using manual implementation")
                _WARNED_CHR_FALLBACK = True
            # fall through to manual code below
    # Fallback without networkx: use Prim for MST and greedy matching
    if not _WARNED_CHR_FALLBACK:
        print('[info] networkx not installed; using greedy matching fallback for Christofides')
        _WARNED_CHR_FALLBACK = True
    n_range = range(n)
    in_tree = [False] * n
    key = [float('inf')] * n
    parent = [-1] * n
    key[0] = 0.0
    for _ in n_range:
        u = -1
        best = float('inf')
        for v in n_range:
            if not in_tree[v] and key[v] < best:
                best = key[v]
                u = v
        if u == -1:
            break
        in_tree[u] = True
        for v in n_range:
            if not in_tree[v] and dist[u, v] < key[v]:
                key[v] = dist[u, v]
                parent[v] = u
    # Build adjacency list
    adj = {i: [] for i in n_range}
    for v in n_range:
        if parent[v] != -1:
            u = parent[v]
            adj[u].append(v)
            adj[v].append(u)
    odd_nodes = [i for i in n_range if len(adj[i]) % 2 == 1]
    # Greedy matching of odd nodes
    unmatched = set(odd_nodes)
    while unmatched:
        u = unmatched.pop()
        v = min(unmatched, key=lambda x: dist[u, x]) if unmatched else None
        if v is None:
            break
        unmatched.remove(v)
        adj[u].append(v)
        adj[v].append(u)
    # Hierholzer style Euler walk
    # Build multiset of edges
    multiedges = {}
    for u in n_range:
        for v in adj[u]:
            if u < v:
                multiedges[(u, v)] = multiedges.get((u, v), 0) + 1
    # adjacency copy for traversal
    local_adj = {i: adj[i][:] for i in n_range}
    stack = [0]
    path = []
    while stack:
        v = stack[-1]
        if local_adj[v]:
            u = local_adj[v].pop()
            local_adj[u].remove(v)
            stack.append(u)
        else:
            path.append(stack.pop())
    euler_nodes = path[::-1]
    # shortcut duplicates preserving order
    seen = set()
    tour = []
    for node in euler_nodes:
        if node not in seen:
            seen.add(node)
            tour.append(node)
    return tour


def run_single_tsp(path: str, args) -> TSPSolution:
    dist = parse_tsp_dat(path)
    n = dist.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 nodes")

    # Christofides path (metric symmetric TSP assumption)
    if getattr(args, 'method', 'auto') == 'christofides':
        if not np.allclose(dist, dist.T, atol=1e-9):
            print(f"[warn] Distance matrix not symmetric; Christofides guarantee invalid (still proceeding)")
        start_t = time.time()
        base_tour = christofides_tour(dist)
        base_cost = tour_cost(base_tour, dist)
        improvements = 0
        if args.ls == 'lk':
            best, best_cost, iters = lk_like_improvement(base_tour, dist)
            runtime = time.time() - start_t
            method = 'christofides_lk'
            return TSPSolution(tour=best, cost=best_cost, runtime=runtime, method=method, improvements=iters)
        elif args.ls in ('2opt', '3opt'):
            best, best_cost, iters = two_opt_numpy(base_tour, dist)
            runtime = time.time() - start_t
            method = 'christofides_2opt'
            return TSPSolution(tour=best, cost=best_cost, runtime=runtime, method=method, improvements=iters)
        elif args.ls == 'none':
            runtime = time.time() - start_t  # measure Christofides construction time
            return TSPSolution(tour=base_tour, cost=base_cost, runtime=runtime, method='christofides', improvements=improvements)
        runtime = 0.0  # negligible vs MST/matching (not timed separately)
        return TSPSolution(tour=base_tour, cost=base_cost, runtime=runtime, method='christofides', improvements=improvements)

    # Random construction with local search
    if getattr(args, 'method', 'auto') == 'random':
        start_t = time.time()
        base_tour = random_tour(n, getattr(args, 'seed', 0))
        base_cost = tour_cost(base_tour, dist)
        improvements = 0
        if args.ls == 'lk':
            best, best_cost, iters = lk_like_improvement(base_tour, dist)
            runtime = time.time() - start_t
            method = 'random_lk'
            return TSPSolution(tour=best, cost=best_cost, runtime=runtime, method=method, improvements=iters)
        elif args.ls in ('2opt', '3opt'):
            best, best_cost, iters = two_opt_numpy(base_tour, dist)
            runtime = time.time() - start_t
            method = 'random_2opt'
            return TSPSolution(tour=best, cost=best_cost, runtime=runtime, method=method, improvements=iters)
        elif args.ls == 'none':
            runtime = time.time() - start_t  # measure random construction time
            return TSPSolution(tour=base_tour, cost=base_cost, runtime=runtime, method='random', improvements=improvements)
        runtime = time.time() - start_t  # measure random construction time
        return TSPSolution(tour=base_tour, cost=base_cost, runtime=runtime, method='random', improvements=improvements)

    if _HAS_ORTOOLS:
        start_strategy = 'PATH_CHEAPEST_ARC' if args.init == 'nearest' else 'PATH_RANDOM'
        sol = build_routing_solution(dist, time_limit=args.limit, ls=args.ls, seed=args.seed, start_strategy=start_strategy)
        return sol
    # Fallback manual
    if args.init == 'random':
        base = random_tour(n, args.seed)
    else:  # nearest neighbor
        unused = set(range(n))
        current = 0
        tour = [current]
        unused.remove(current)
        while unused:
            nxt = min(unused, key=lambda j: dist[current, j])
            tour.append(nxt)
            unused.remove(nxt)
            current = nxt
        base = tour
    start_t = time.time()
    if args.ls == 'none':
        best, cost, iters = base, tour_cost(base, dist), 0
        runtime = 0.0
        method_name = 'nearest_neighbor' if args.init == 'nearest' else 'random_fallback'
    elif args.ls == '2opt' or args.ls == '3opt':
        best, cost, iters = two_opt_numpy(base, dist)
        runtime = time.time() - start_t
        method_name = f"fallback_{args.ls}"
    elif args.ls == 'lk':
        best, cost, iters = lk_like_improvement(base, dist)
        runtime = time.time() - start_t
        method_name = f"fallback_{args.ls}"
    else:
        best, cost, iters = two_opt_numpy(base, dist)
        runtime = time.time() - start_t
        method_name = f"fallback_{args.ls}"
    return TSPSolution(tour=best, cost=cost, runtime=runtime, method=method_name, improvements=iters)


def solve_tsp_heuristic(dist: np.ndarray, *, method: str = 'auto', ls: str = '2opt', init: str = 'nearest',
                         seed: int = 0, limit: int = 5) -> TSPSolution:
    """Programmatic API to obtain a heuristic TSP solution.

    Parameters:
      dist: square numpy array of distances
      method: 'auto' (OR-Tools / fallback) or 'christofides' or 'random'
      ls: '2opt' | '3opt' | 'lk' | 'none'
      init: 'nearest' | 'random' (only relevant for 'auto' fallback or OR-Tools start strategy)
      seed: random seed passed to OR-Tools / random initializers
      limit: time limit (seconds) for OR-Tools search
    """
    class _Args:  # lightweight shim for reuse of existing logic
        pass
    args = _Args()
    args.method = method
    args.ls = ls
    args.init = init
    args.seed = seed
    args.limit = limit
    # Reuse core logic by temporarily writing a small helper replicating run_single_tsp minus file I/O.
    n = dist.shape[0]
    if n < 3:
        raise ValueError('Need at least 3 nodes')
    if method == 'christofides':
        if not np.allclose(dist, dist.T, atol=1e-9):
            pass  # silent; benchmarking may spam otherwise
        start_t = time.time()
        base_tour = christofides_tour(dist)
        if ls == 'lk':
            best, best_cost, iters = lk_like_improvement(base_tour, dist)
            runtime = time.time() - start_t
            return TSPSolution(tour=best, cost=best_cost, runtime=runtime, method='christofides_lk', improvements=iters)
        elif ls in ('2opt', '3opt'):
            best, best_cost, iters = two_opt_numpy(base_tour, dist)
            runtime = time.time() - start_t
            return TSPSolution(tour=best, cost=best_cost, runtime=runtime, method='christofides_2opt', improvements=iters)
        elif ls == 'none':
            runtime = time.time() - start_t
            return TSPSolution(tour=base_tour, cost=tour_cost(base_tour, dist), runtime=runtime, method='christofides', improvements=0)
        runtime = time.time() - start_t
        return TSPSolution(tour=base_tour, cost=tour_cost(base_tour, dist), runtime=runtime, method='christofides', improvements=0)
    if method == 'random':
        start_t = time.time()
        base_tour = random_tour(n, seed)
        if ls == 'lk':
            best, best_cost, iters = lk_like_improvement(base_tour, dist)
            runtime = time.time() - start_t
            return TSPSolution(tour=best, cost=best_cost, runtime=runtime, method='random_lk', improvements=iters)
        elif ls in ('2opt', '3opt'):
            best, best_cost, iters = two_opt_numpy(base_tour, dist)
            runtime = time.time() - start_t
            return TSPSolution(tour=best, cost=best_cost, runtime=runtime, method='random_2opt', improvements=iters)
        elif ls == 'none':
            runtime = time.time() - start_t
            return TSPSolution(tour=base_tour, cost=tour_cost(base_tour, dist), runtime=runtime, method='random', improvements=0)
        runtime = time.time() - start_t
        return TSPSolution(tour=base_tour, cost=tour_cost(base_tour, dist), runtime=runtime, method='random', improvements=0)
    if _HAS_ORTOOLS:
        start_strategy = 'PATH_CHEAPEST_ARC' if init == 'nearest' else 'PATH_RANDOM'
        return build_routing_solution(dist, time_limit=limit, ls=ls, seed=seed, start_strategy=start_strategy)
    # fallback manual construction
    if init == 'random':
        base = random_tour(n, seed)
    else:
        unused = set(range(n))
        current = 0
        tour = [current]
        unused.remove(current)
        while unused:
            nxt = min(unused, key=lambda j: dist[current, j])
            tour.append(nxt)
            unused.remove(nxt)
            current = nxt
        base = tour
    start_t = time.time()
    if ls == 'none':
        best, cost, iters = base, tour_cost(base, dist), 0
        runtime = 0.0
        method_name = 'nearest_neighbor' if init == 'nearest' else 'random_fallback'
    elif ls in ('2opt', '3opt'):
        best, cost, iters = two_opt_numpy(base, dist)
        runtime = time.time() - start_t
        method_name = f"fallback_{ls}"
    elif ls == 'lk':
        best, cost, iters = lk_like_improvement(base, dist)
        runtime = time.time() - start_t
        method_name = f"fallback_{ls}"
    else:
        best, cost, iters = two_opt_numpy(base, dist)
        runtime = time.time() - start_t
        method_name = f"fallback_{ls}"
    return TSPSolution(tour=best, cost=cost, runtime=runtime, method=method_name, improvements=iters)


def iter_instances(data_dir: str, pattern: Optional[str], all_flag: bool) -> Iterable[str]:
    if pattern:
        for p in glob.glob(os.path.join(data_dir, pattern)):
            yield p
        return
    if all_flag:
        for p in sorted(glob.glob(os.path.join(data_dir, '*.dat'))):
            yield p
        return
    raise ValueError("Provide --pattern or --all")


def main():  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Heuristic TSP solver (nearest/random + 2/3-opt via OR-Tools)")
    ap.add_argument('--data-dir', default='dat/tsp')
    ap.add_argument('--file', help='Solve a single explicit .dat file (overrides pattern/all)')
    ap.add_argument('--pattern', help='Filename or glob pattern (e.g., gr21.dat or gr*.dat)')
    ap.add_argument('--all', action='store_true', help='Run all .dat in directory')
    ap.add_argument('--method', choices=['auto', 'christofides', 'random'], default='auto', help='Solution construction method (auto uses OR-Tools or fallback, random uses random tour + local search)')
    ap.add_argument('--ls', choices=['2opt', '3opt', 'lk', 'none'], default='2opt', help='Local search flavor (lk activates Lin-Kernighan in OR-Tools, none skips local search)')
    ap.add_argument('--init', choices=['nearest', 'random'], default='nearest')
    ap.add_argument('--limit', type=int, default=5, help='Time limit seconds for OR-Tools search')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--max-n', type=int, help='Only solve instances with number of nodes < max-n')
    ap.add_argument('--json', action='store_true', help='Emit JSON array of results to stdout instead of plain text lines')
    args = ap.parse_args()

    # Default behavior: if neither --pattern nor --all provided, assume --all
    if args.file:
        # Direct single-file mode
        targets = [args.file]
    else:
        if not args.pattern and not args.all:
            args.all = True
            print("[info] No --pattern or --all specified; defaulting to all instances in", args.data_dir)
        targets = list(iter_instances(args.data_dir, args.pattern, args.all))

    results: List[Tuple[str, float, float, str, int]] = []
    skipped: List[Tuple[str, int]] = []
    for path in targets:
        name = os.path.basename(path)
        try:
            # Pre-parse to filter on size if requested
            if args.max_n is not None:
                try:
                    dist_preview = parse_tsp_dat(path)
                    n_nodes = dist_preview.shape[0]
                    if n_nodes >= args.max_n:
                        skipped.append((name, n_nodes))
                        continue
                    # reuse parsed matrix by monkey-patching into args for efficiency? Simpler: pass along; run_single_tsp parses again (overhead small)
                except Exception as e:
                    print(f"{name:20s} PARSE_ERROR {e}")
                    continue
            sol = run_single_tsp(path, args)
            if not args.json:
                print(f"{name:20s} cost={sol.cost:12.2f} time={sol.runtime:6.3f}s method={sol.method}")
            results.append((name, sol.cost, sol.runtime, sol.method, sol.improvements))
        except Exception as e:
            if not args.json:
                print(f"{name:20s} ERROR {e}")
    if args.json:
        import json as _json
        out_list = [
            {
                'instance': r[0],
                'cost': r[1],
                'runtime': r[2],
                'method': r[3],
                'improvements': r[4],
            }
            for r in results
        ]
        print(_json.dumps(out_list))
    else:
        if args.summary and results:
            print("\nSummary:")
            for r in results:
                print(r)
        if skipped and not args.file:
            print(f"\nSkipped (>= max-n={args.max_n}):")
            for nm, sz in skipped:
                print(f"  {nm} (n={sz})")
        if skipped:
            print(f"\nSkipped (>= max-n={args.max_n}):")
            for nm, sz in skipped:
                print(f"  {nm} (n={sz})")


if __name__ == '__main__':
    main()
