#!/usr/bin/env python3
"""
TSP Construction + Local Search Benchmark

This script runs a more efficient benchmark that:
1. Constructs tours using different methods (random, nearest_neighbor, christofides)
2. For each constructed tour, applies different local search methods (none, 2opt, lk)
3. Generates separate result files for construction-only, construction+2opt, construction+lk

This avoids redundant construction computation.
"""

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

# Import our heuristic functions
from heuristic_tsp import (
    parse_tsp_dat, random_tour, christofides_tour, tour_cost,
    two_opt_numpy, lk_like_improvement, TSPSolution
)

@dataclass
class ConstructionResult:
    """Result of running a construction method"""
    instance: str
    n: int
    construction_method: str
    seed: int
    run: int
    tour: List[int]
    cost: float
    construction_time: float

@dataclass 
class LocalSearchResult:
    """Result of applying local search to a constructed tour"""
    instance: str
    n: int
    construction_method: str
    ls_method: str
    seed: int
    run: int
    initial_cost: float
    final_cost: float
    improvement: float
    ls_time: float
    ls_iterations: int

def load_existing_results(output_files: List[str]) -> Set[Tuple[str, str, int]]:
    """Load existing results to determine what work has already been done.
    Returns set of (instance, construction_method, run) tuples that are complete.
    """
    completed = set()
    
    for file_path in output_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                # Skip header
                next(f)
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        instance = parts[0]
                        construction_method = parts[2] if 'construction_method' in file_path else parts[2]
                        run = int(parts[3]) if 'construction_method' in file_path else int(parts[3])
                        completed.add((instance, construction_method, run))
        except Exception as e:
            print(f"Warning: Could not read existing results from {file_path}: {e}")
    
    return completed

def load_all_existing_results(output_files: List[str], bks: Dict[str, float]) -> Tuple[List[ConstructionResult], List[LocalSearchResult]]:
    """Load all existing results from output files"""
    construction_results = []
    ls_results = []
    
    # Load construction results
    construction_file = output_files[0]  # construction_only.csv
    if os.path.exists(construction_file):
        with open(construction_file, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 7:
                    construction_results.append(ConstructionResult(
                        instance=parts[0],
                        n=int(parts[1]),
                        construction_method=parts[2],
                        seed=0,  # We don't store seed in CSV, not critical for resuming
                        run=int(parts[3]),
                        tour=[],  # We don't store the actual tour, not needed for result files
                        cost=float(parts[4]),
                        construction_time=float(parts[6])
                    ))
    
    # Load local search results (from both 2opt and lk files)
    for i, ls_method in enumerate(['2opt', 'lk']):
        ls_file = output_files[i + 1]  # construction_2opt.csv or construction_lk.csv
        if os.path.exists(ls_file):
            with open(ls_file, 'r') as f:
                # Skip header  
                next(f)
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 12:
                        ls_results.append(LocalSearchResult(
                            instance=parts[0],
                            n=int(parts[1]),
                            construction_method=parts[2],
                            ls_method=ls_method,
                            seed=0,  # Not stored in CSV
                            run=int(parts[3]),
                            initial_cost=float(parts[4]),
                            final_cost=float(parts[5]),
                            improvement=float(parts[6]),
                            ls_time=float(parts[9]),
                            ls_iterations=int(parts[11])
                        ))
    
    return construction_results, ls_results

def save_results_incrementally(construction_results: List[ConstructionResult],
                              ls_results: List[LocalSearchResult], 
                              bks: Dict[str, float], out_dir: str) -> None:
    """Save results incrementally after each instance"""
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Construction-only results
    construction_file = os.path.join(out_dir, "construction_only.csv")
    write_construction_results(construction_results, ls_results, bks, construction_file)
    
    # 2. Construction + 2opt results  
    twopt_file = os.path.join(out_dir, "construction_2opt.csv")
    write_ls_results(construction_results, ls_results, bks, "2opt", twopt_file)
    
    # 3. Construction + LK results
    lk_file = os.path.join(out_dir, "construction_lk.csv")
    write_ls_results(construction_results, ls_results, bks, "lk", lk_file)

def load_bks(bks_file: str = "tsplib_bks.txt") -> Dict[str, float]:
    """Load best known solutions from file"""
    bks = {}
    if not os.path.exists(bks_file):
        return bks
    
    with open(bks_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    instance = parts[0]
                    try:
                        value = float(parts[1])
                        bks[instance] = value
                    except ValueError:
                        continue
    return bks
    """Load best known solutions from file"""
    bks = {}
    if not os.path.exists(bks_file):
        return bks
    
    with open(bks_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    instance = parts[0]
                    try:
                        value = float(parts[1])
                        bks[instance] = value
                    except ValueError:
                        continue
    return bks

def extract_instance_name(path: str) -> str:
    """Extract instance name from file path"""
    basename = os.path.basename(path)
    return basename.replace('.dat', '')

def construct_tour(method: str, dist: np.ndarray, seed: int, init: str = 'nearest') -> Tuple[List[int], float]:
    """Construct a tour using the specified method"""
    n = dist.shape[0]
    
    if method == 'random':
        tour = random_tour(n, seed)
        construction_time = 0.001  # Negligible
        
    elif method == 'nearest_neighbor':
        # Manual nearest neighbor implementation
        np.random.seed(seed)
        start = np.random.randint(0, n) if init == 'random' else 0
        unused = set(range(n))
        current = start
        tour = [current]
        unused.remove(current)
        
        start_t = time.time()
        while unused:
            nxt = min(unused, key=lambda j: dist[current, j])
            tour.append(nxt)
            unused.remove(nxt)
            current = nxt
        construction_time = time.time() - start_t
        
    elif method == 'christofides':
        start_t = time.time()
        if not np.allclose(dist, dist.T, atol=1e-9):
            print(f"[warn] Distance matrix not symmetric; Christofides guarantee invalid")
        tour = christofides_tour(dist)
        construction_time = time.time() - start_t
        
    else:
        raise ValueError(f"Unknown construction method: {method}")
    
    cost = tour_cost(tour, dist)
    return tour, construction_time

def apply_local_search(tour: List[int], dist: np.ndarray, method: str) -> Tuple[List[int], float, int, float]:
    """Apply local search to a tour"""
    if method == 'none':
        return tour, tour_cost(tour, dist), 0, 0.0
    
    start_t = time.time()
    if method == '2opt':
        improved_tour, improved_cost, iterations = two_opt_numpy(tour, dist)
    elif method == 'lk':
        improved_tour, improved_cost, iterations = lk_like_improvement(tour, dist)
    else:
        raise ValueError(f"Unknown local search method: {method}")
    
    ls_time = time.time() - start_t
    return improved_tour, improved_cost, iterations, ls_time

def run_construction_benchmark(instances: List[str], construction_methods: List[str], 
                              ls_methods: List[str], runs: int = 3, seed_offset: int = 0,
                              out_dir: str = "results", bks_file: str = "tsplib_bks.txt", 
                              max_n: int = 100) -> None:
    """Run the efficient construction + local search benchmark with incremental saving"""
    
    # Filter instances by size
    filtered_instances = []
    for instance_path in instances:
        try:
            dist = parse_tsp_dat(instance_path)
            if dist.shape[0] < max_n:
                filtered_instances.append(instance_path)
        except Exception as e:
            print(f"Warning: Could not parse {instance_path}: {e}")
            continue
    
    instances = filtered_instances
    
    # Load BKS
    bks = load_bks(bks_file)
    
    # Check what's already been done
    os.makedirs(out_dir, exist_ok=True)
    output_files = [
        os.path.join(out_dir, "construction_only.csv"),
        os.path.join(out_dir, "construction_2opt.csv"), 
        os.path.join(out_dir, "construction_lk.csv")
    ]
    completed_work = load_existing_results(output_files)
    
    # Results storage
    construction_results: List[ConstructionResult] = []
    ls_results: List[LocalSearchResult] = []
    
    # Load existing results if resuming
    if completed_work:
        print(f"Found {len(completed_work)} completed runs, resuming...")
        construction_results, ls_results = load_all_existing_results(output_files, bks)
    
    print(f"Running construction benchmark on {len(instances)} instances (n < {max_n})")
    print(f"Construction methods: {construction_methods}")
    print(f"Local search methods: {ls_methods}")
    print(f"Runs per configuration: {runs}")
    
    total_configs = len(instances) * len(construction_methods) * runs
    completed_configs = len(completed_work)
    
    print(f"Progress: {completed_configs}/{total_configs} configurations already completed")
    
    for instance_path in instances:
        instance_name = extract_instance_name(instance_path)
        
        # Check if this entire instance is already complete
        instance_complete = True
        for construction_method in construction_methods:
            for run in range(runs):
                if (instance_name, construction_method, run + 1) not in completed_work:
                    instance_complete = False
                    break
            if not instance_complete:
                break
        
        if instance_complete:
            print(f"Skipping {instance_name} (already complete)")
            continue
            
        print(f"\nInstance {instance_name}")
        
        try:
            dist = parse_tsp_dat(instance_path)
            n = dist.shape[0]
            bks_value = bks.get(instance_name, None)
            
            print(f"  Size: {n}, BKS: {bks_value if bks_value else 'unknown'}")
            
            instance_has_new_results = False
            
            for construction_method in construction_methods:
                print(f"  Construction: {construction_method}")
                
                for run in range(runs):
                    # Skip if this specific run is already done
                    if (instance_name, construction_method, run + 1) in completed_work:
                        print(f"    Run {run + 1}: already completed")
                        continue
                        
                    seed = seed_offset + run
                    
                    # 1. Construct tour
                    tour, construction_time = construct_tour(construction_method, dist, seed)
                    initial_cost = tour_cost(tour, dist)
                    
                    # Store construction result
                    construction_results.append(ConstructionResult(
                        instance=instance_name,
                        n=n,
                        construction_method=construction_method,
                        seed=seed,
                        run=run + 1,
                        tour=tour,
                        cost=initial_cost,
                        construction_time=construction_time
                    ))
                    
                    # 2. Apply each local search method to the same constructed tour
                    for ls_method in ls_methods:
                        improved_tour, improved_cost, iterations, ls_time = apply_local_search(tour.copy(), dist, ls_method)
                        
                        improvement = initial_cost - improved_cost
                        
                        # Store local search result
                        ls_results.append(LocalSearchResult(
                            instance=instance_name,
                            n=n,
                            construction_method=construction_method,
                            ls_method=ls_method,
                            seed=seed,
                            run=run + 1,
                            initial_cost=initial_cost,
                            final_cost=improved_cost,
                            improvement=improvement,
                            ls_time=ls_time,
                            ls_iterations=iterations
                        ))
                        
                        gap_str = ""
                        if bks_value:
                            gap = (improved_cost - bks_value) / bks_value * 100
                            gap_str = f" (gap: {gap:.2f}%)"
                        
                        print(f"    Run {run + 1} {ls_method}: {initial_cost:.0f} → {improved_cost:.0f} "
                              f"(improv: {improvement:.0f}, iters: {iterations}){gap_str}")
                    
                    instance_has_new_results = True
            
            # Save results after completing each instance
            if instance_has_new_results:
                print(f"  ✓ Saving results after {instance_name}")
                save_results_incrementally(construction_results, ls_results, bks, out_dir)
        
        except Exception as e:
            print(f"  Error processing {instance_name}: {e}")
            continue
    
    print(f"\n✓ Final results written to {out_dir}/")
    print(f"  - construction_only.csv")
    print(f"  - construction_2opt.csv") 
    print(f"  - construction_lk.csv")

def write_construction_results(construction_results: List[ConstructionResult], 
                              ls_results: List[LocalSearchResult], bks: Dict[str, float],
                              output_file: str) -> None:
    """Write construction-only results"""
    with open(output_file, 'w') as f:
        f.write("instance,n,construction_method,run,cost,gap_percent,construction_time,bks\n")
        
        for result in construction_results:
            gap_percent = ""
            if result.instance in bks:
                gap = (result.cost - bks[result.instance]) / bks[result.instance] * 100
                gap_percent = f"{gap:.4f}"
            
            bks_value = bks.get(result.instance, "")
            
            f.write(f"{result.instance},{result.n},{result.construction_method},"
                   f"{result.run},{result.cost},{gap_percent},{result.construction_time:.6f},"
                   f"{bks_value}\n")

def write_ls_results(construction_results: List[ConstructionResult],
                    ls_results: List[LocalSearchResult], bks: Dict[str, float], 
                    ls_method: str, output_file: str) -> None:
    """Write construction + local search results"""
    with open(output_file, 'w') as f:
        f.write("instance,n,construction_method,run,initial_cost,final_cost,improvement,"
               "gap_percent,construction_time,ls_time,total_time,ls_iterations,bks\n")
        
        # Create lookup for construction results
        construction_lookup = {}
        for cr in construction_results:
            key = (cr.instance, cr.construction_method, cr.run)
            construction_lookup[key] = cr
        
        for result in ls_results:
            if result.ls_method != ls_method:
                continue
                
            key = (result.instance, result.construction_method, result.run)
            construction_result = construction_lookup.get(key)
            construction_time = construction_result.construction_time if construction_result else 0.0
            
            gap_percent = ""
            if result.instance in bks:
                gap = (result.final_cost - bks[result.instance]) / bks[result.instance] * 100
                gap_percent = f"{gap:.4f}"
            
            bks_value = bks.get(result.instance, "")
            total_time = construction_time + result.ls_time
            
            f.write(f"{result.instance},{result.n},{result.construction_method},"
                   f"{result.run},{result.initial_cost},{result.final_cost},"
                   f"{result.improvement},{gap_percent},{construction_time:.6f},"
                   f"{result.ls_time:.6f},{total_time:.6f},{result.ls_iterations},"
                   f"{bks_value}\n")

def main():
    parser = argparse.ArgumentParser(description="Efficient TSP Construction + Local Search Benchmark")
    parser.add_argument('--data-dir', default='dat/tsp', help='Directory containing .dat files')
    parser.add_argument('--pattern', help='Glob pattern for instances (e.g., "gr*.dat")')
    parser.add_argument('--all', action='store_true', help='Run all .dat files in directory')
    parser.add_argument('--max-n', type=int, help='Maximum number of cities')
    parser.add_argument('--construction', default='random,nearest_neighbor,christofides',
                       help='Comma-separated construction methods')
    parser.add_argument('--local-search', default='none,2opt,lk', 
                       help='Comma-separated local search methods')
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per configuration')
    parser.add_argument('--seed', type=int, default=0, help='Base seed offset')
    parser.add_argument('--out-dir', default='results/construction_benchmark', help='Output directory')
    parser.add_argument('--bks-file', default='tsplib_bks.txt', help='Best known solutions file')
    
    args = parser.parse_args()
    
    # Collect instances
    instances = []
    if args.pattern:
        instances.extend(glob.glob(os.path.join(args.data_dir, args.pattern)))
    elif args.all:
        instances.extend(glob.glob(os.path.join(args.data_dir, '*.dat')))
    else:
        print("Specify --pattern or --all")
        return
    
    instances = sorted(instances)
    
    if not instances:
        print("No instances found")
        return
    
    construction_methods = [m.strip() for m in args.construction.split(',')]
    ls_methods = [m.strip() for m in args.local_search.split(',')]
    
    run_construction_benchmark(
        instances=instances,
        construction_methods=construction_methods,
        ls_methods=ls_methods,
        runs=args.runs,
        seed_offset=args.seed,
        out_dir=args.out_dir,
        bks_file=args.bks_file,
        max_n=args.max_n or 10000  # Default to large number if not specified
    )

if __name__ == "__main__":
    main()
