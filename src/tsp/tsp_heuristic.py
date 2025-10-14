#!/usr/bin/env python3
"""Benchmark multiple heuristic TSP configurations over .dat instances.

Uses functions in heuristic_tsp.py (parse_tsp_dat, solve_tsp_heuristic).

Features:
 - Select instances by glob (--pattern) or all (--all)
 - Size filtering (--max-n)
 - Multiple heuristic method/LS configurations (predefined + user filter)
 - Multiple runs per instance with different seeds
 - Incremental saving of detailed JSON and CSV summaries
 - Optional statistical comparison (Friedman / Wilcoxon) if scipy present
 - Graceful degradation if pandas / scipy missing

Example:
  python HEURISTIC_BENCHMARK.py --all --max-n 800 --runs 5 --methods ortools_lk,christofides_2opt --summary

Method names (default set):
  ortools_2opt        -> method=auto         ls=2opt
  ortools_3opt        -> method=auto         ls=3opt
  ortools_lk          -> method=auto         ls=lk
  christofides_2opt   -> method=christofides ls=2opt
  christofides_lk     -> method=christofides ls=lk (variable-depth search)
  random_2opt         -> method=random       ls=2opt
  random_lk           -> method=random       ls=lk

Outputs (in results/ by default):
  heuristic_detailed_results.json  (append-only detailed per-run records)
  heuristic_detailed_results.txt   (human-readable log)
  heuristic_results_incremental.csv (overwritten summary after each method)
  heuristic_results_final.csv       (final summary)
  heuristic_statistics.txt          (if statistical tests run)

"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import subprocess
import json as _json

# Soft dependencies
try:
    import numpy as np
except Exception as e:  # pragma: no cover
    print("[fatal] numpy required for benchmark: pip install numpy")
    raise

try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except Exception:  # pragma: no cover
    _HAS_PANDAS = False

try:
    from scipy.stats import friedmanchisquare, wilcoxon  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

# Import solver helpers - temporarily disabled to avoid circular import issues
_HAS_INPROC = False
print("[info] Using subprocess mode only to avoid import issues.")
# try:
#     from heuristic_tsp import parse_tsp_dat, solve_tsp_heuristic, tour_cost  # type: ignore
#     _HAS_INPROC = True
# except ImportError:
#     print("[warn] Cannot import heuristic_tsp in-process; will rely on subprocess mode only.")
#     _HAS_INPROC = False

DEFAULT_METHODS: Dict[str, Dict[str, str]] = {
    # Construction only (no local search)
    'random': {'method': 'random', 'ls': 'none'},
    'nearest_neighbor': {'method': 'auto', 'ls': 'none', 'init': 'nearest'},
    'christofides': {'method': 'christofides', 'ls': 'none'},
    # Construction + 2opt
    'random_2opt': {'method': 'random', 'ls': '2opt'},
    'nearest_neighbor_2opt': {'method': 'auto', 'ls': '2opt', 'init': 'nearest'},
    'christofides_2opt': {'method': 'christofides', 'ls': '2opt'},
    # Construction + Lin-Kernighan
    'random_lk': {'method': 'random', 'ls': 'lk'},
    'nearest_neighbor_lk': {'method': 'auto', 'ls': 'lk', 'init': 'nearest'},
    'christofides_lk': {'method': 'christofides', 'ls': 'lk'},
    # Legacy OR-Tools methods
    'ortools_2opt': {'method': 'auto', 'ls': '2opt'},
    'ortools_3opt': {'method': 'auto', 'ls': '3opt'},
    'ortools_lk': {'method': 'auto', 'ls': 'lk'},
}

@dataclass
class RunRecord:
    instance: str
    n: int
    method_name: str
    method: str
    ls: str
    run: int
    seed: int
    status: str
    cost: Optional[float]
    runtime: float
    improvements: int
    timestamp: str


def iter_instances(data_dir: str, pattern: Optional[str], all_flag: bool) -> List[str]:
    if pattern:
        paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    elif all_flag:
        paths = sorted(glob.glob(os.path.join(data_dir, '*.dat')))
    else:
        raise ValueError('Provide --pattern or --all')
    return paths


def load_or_parse(path: str) -> Tuple[np.ndarray, int]:
    """Extract TSP instance size from filename patterns."""
    basename = os.path.basename(path).replace('.dat', '').replace('.tsp', '')
    
    # Extract size from common TSP instance naming patterns
    import re
    
    # Pattern matching for various TSP instance formats
    patterns = [
        (r'^gr(\d+)$', lambda m: int(m.group(1))),           # gr17, gr21, gr24, etc.
        (r'^att(\d+)$', lambda m: int(m.group(1))),          # att48, att532, etc.
        (r'^berlin(\d+)$', lambda m: int(m.group(1))),       # berlin52
        (r'^ch(\d+)$', lambda m: int(m.group(1))),           # ch130, ch150
        (r'^eil(\d+)$', lambda m: int(m.group(1))),          # eil51, eil76, eil101
        (r'^pr(\d+)$', lambda m: int(m.group(1))),           # pr76, pr107, pr124, etc.
        (r'^kro[A-Z](\d+)$', lambda m: int(m.group(1))),     # kroA100, kroB150, etc.
        (r'^kro(\d+)p?$', lambda m: int(m.group(1))),        # kro124p
        (r'^lin(\d+)$', lambda m: int(m.group(1))),          # lin105, lin318
        (r'^d(\d+)$', lambda m: int(m.group(1))),            # d198, d493, d657, etc.
        (r'^a(\d+)$', lambda m: int(m.group(1))),            # a280
        (r'^bier(\d+)$', lambda m: int(m.group(1))),         # bier127
        (r'^brazil(\d+)$', lambda m: int(m.group(1))),       # brazil58
        (r'^burma(\d+)$', lambda m: int(m.group(1))),        # burma14
        (r'^fri(\d+)$', lambda m: int(m.group(1))),          # fri26
        (r'^gil(\d+)$', lambda m: int(m.group(1))),          # gil262
        (r'^st(\d+)$', lambda m: int(m.group(1))),           # st70
        (r'^bayg(\d+)$', lambda m: int(m.group(1))),         # bayg29
        (r'^bays(\d+)$', lambda m: int(m.group(1))),         # bays29
        (r'^dantzig(\d+)$', lambda m: int(m.group(1))),      # dantzig42
        (r'^hk(\d+)$', lambda m: int(m.group(1))),           # hk48
        (r'^rat(\d+)$', lambda m: int(m.group(1))),          # rat99, rat195, etc.
        (r'^rd(\d+)$', lambda m: int(m.group(1))),           # rd100, rd400
        (r'^swiss(\d+)$', lambda m: int(m.group(1))),        # swiss42
        (r'^ulysses(\d+)$', lambda m: int(m.group(1))),      # ulysses16, ulysses22
        (r'^u(\d+)$', lambda m: int(m.group(1))),            # u159, u574, etc.
        (r'^vm(\d+)$', lambda m: int(m.group(1))),           # vm1084, vm1748
        (r'^pcb(\d+)$', lambda m: int(m.group(1))),          # pcb442, pcb1173
        (r'^fl(\d+)$', lambda m: int(m.group(1))),           # fl417, fl1400, fl1577
        (r'^ts(\d+)$', lambda m: int(m.group(1))),           # ts225
        (r'^tsp(\d+)$', lambda m: int(m.group(1))),          # tsp225
        (r'^si(\d+)$', lambda m: int(m.group(1))),           # si175, si535, si1032
        (r'^ali(\d+)$', lambda m: int(m.group(1))),          # ali535
        (r'^pa(\d+)$', lambda m: int(m.group(1))),           # pa561
        (r'^p(\d+)$', lambda m: int(m.group(1))),            # p654
        (r'^brg(\d+)$', lambda m: int(m.group(1))),          # brg180
        (r'^dsj(\d+)$', lambda m: int(m.group(1))),          # dsj1000
        (r'^rl(\d+)$', lambda m: int(m.group(1))),           # rl1304, rl1323, etc.
        (r'^nrw(\d+)$', lambda m: int(m.group(1))),          # nrw1379
        (r'^linhp(\d+)$', lambda m: int(m.group(1))),        # linhp318
        (r'^pla(\d+)$', lambda m: int(m.group(1))),          # pla7397
        (r'^fnl(\d+)$', lambda m: int(m.group(1))),          # fnl4461
        (r'^fl(\d+)$', lambda m: int(m.group(1))),           # fl3795
    ]
    
    for pattern, extractor in patterns:
        match = re.match(pattern, basename)
        if match:
            n = extractor(match)
            return np.zeros((n, n)), n
    
    # Fallback: try to extract any number from the filename
    numbers = re.findall(r'\d+', basename)
    if numbers:
        # Usually the largest number is the instance size
        n = max(int(num) for num in numbers)
        return np.zeros((n, n)), n
    
    raise ValueError(f"Could not determine size for {basename} from {path}")


def run_single(dist: np.ndarray, cfg: Dict[str, str], seed: int, time_limit: int, instance_path: str) -> Tuple[str, float, int, float]:
    """Run one configuration using subprocess calls only."""
    method = cfg['method']
    ls = cfg['ls']
    init = cfg.get('init', 'nearest')  # Default to nearest if not specified
    
    # Use subprocess call - this avoids the circular import issue
    cmd = [sys.executable, 'heuristic_tsp.py', '--file', instance_path, '--method', method, '--ls', ls, '--init', init, '--limit', str(time_limit), '--seed', str(seed), '--json']
    start_t = time.time()
    
    # Set timeout based on method - Christofides is fast, OR-Tools may use full time
    if method == 'christofides':
        subprocess_timeout = min(30, time_limit + 10)  # Christofides should finish quickly
    else:
        subprocess_timeout = time_limit + 20  # OR-Tools may need the full time
    
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=subprocess_timeout, universal_newlines=True)
        elapsed = time.time() - start_t
        # Expect JSON list
        lines = out.strip().splitlines()
        if lines:
            data = _json.loads(lines[-1])
            if data and isinstance(data, list) and len(data) > 0:
                rec = data[0]
                return 'ok', float(rec['cost']), int(rec.get('improvements', 0)), elapsed
        return 'no_data', float('nan'), 0, elapsed
    except subprocess.TimeoutExpired:
        return 'timeout', float('nan'), 0, time.time() - start_t
    except subprocess.CalledProcessError as e:
        return f'error_subproc:{e.returncode}', float('nan'), 0, time.time() - start_t
    except Exception as e:
        return f'error_subproc:{e.__class__.__name__}', float('nan'), 0, time.time() - start_t


def append_detailed(results_dir: str, new_records: List[RunRecord]):
    if not new_records:
        return
    json_path = os.path.join(results_dir, 'heuristic_detailed_results.json')
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        data = []
    existing_keys = {(d['instance'], d['method_name'], d['run']) for d in data}
    added = 0
    for r in new_records:
        key = (r.instance, r.method_name, r.run)
        if key in existing_keys:
            continue
        data.append(r.__dict__)
        added += 1
    if added:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Added {added} detailed records")
    # Text log
    txt_path = os.path.join(results_dir, 'heuristic_detailed_results.txt')
    with open(txt_path, 'a') as f:
        for r in new_records:
            f.write(f"{r.timestamp} {r.instance} n={r.n} {r.method_name} run={r.run} seed={r.seed} status={r.status} cost={r.cost} time={r.runtime:.4f}s imp={r.improvements}\n")


def summarize(records: List[RunRecord]):
    if not _HAS_PANDAS:
        print('[warn] pandas not installed; skipping summary DataFrame creation')
        return None
    import pandas as pd  # local
    rows = [r.__dict__ for r in records]
    df = pd.DataFrame(rows)
    grp = df.groupby(['instance', 'method_name'])
    summary = grp.agg(
        n=('n', 'first'),
        runs=('run', 'count'),
        successes=('status', lambda s: (s == 'ok').sum()),
        cost_best=('cost', 'min'),
        cost_mean=('cost', 'mean'),
        runtime_mean=('runtime', 'mean'),
        runtime_std=('runtime', 'std'),
    ).reset_index()
    return summary


def statistical_tests(summary_df, results_dir: str):
    if summary_df is None or not _HAS_SCIPY:
        print('[info] Skipping statistical tests (missing pandas or scipy)')
        return
    # Pivot runtime_mean for Friedman/Wilcoxon
    piv = summary_df.pivot(index='instance', columns='method_name', values='runtime_mean').dropna()
    out_lines: List[str] = []
    out_lines.append('Heuristic Benchmark Statistical Comparison')
    out_lines.append('=' * 60)
    if piv.shape[1] >= 3:
        try:
            stat, p = friedmanchisquare(*[piv[c] for c in piv.columns])
            out_lines.append(f'Friedman runtime_mean: stat={stat:.4f} p={p:.3e}')
        except Exception as e:
            out_lines.append(f'Friedman test failed: {e}')
    # Pairwise Wilcoxon
    cols = list(piv.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i+1:]:
            try:
                stat, p = wilcoxon(piv[c1], piv[c2])
                out_lines.append(f'Wilcoxon {c1} vs {c2}: stat={stat} p={p:.3e}')
            except Exception as e:
                out_lines.append(f'Wilcoxon {c1} vs {c2} failed: {e}')
    path = os.path.join(results_dir, 'heuristic_statistics.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    print(f'✓ Statistical analysis written to {path}')


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description='Benchmark heuristic TSP configurations over .dat instances')
    ap.add_argument('--data-dir', default='dat/tsp')
    ap.add_argument('--pattern', help='Glob for .dat e.g. gr*.dat')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--min-n', type=int, default=0, help='Minimum nodes (inclusive)')
    ap.add_argument('--max-n', type=int, default=1000, help='Maximum nodes (exclusive)')
    ap.add_argument('--runs', type=int, default=3, help='Runs (seeds) per method per instance')
    ap.add_argument('--methods', help='Comma list of method names to include (default: all)')
    ap.add_argument('--time-limit', type=int, default=100, help='OR-Tools time limit per run (s)')
    ap.add_argument('--seed', type=int, default=0, help='Base seed offset')
    ap.add_argument('--out-dir', default='results', help='Output directory')
    ap.add_argument('--summary', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.pattern and not args.all:
        args.all = True
        print('[info] no pattern/all specified; defaulting to --all')

    selected_methods = DEFAULT_METHODS
    if args.methods:
        requested = [m.strip() for m in args.methods.split(',') if m.strip()]
        missing = [m for m in requested if m not in DEFAULT_METHODS]
        if missing:
            print(f'[error] unknown method names: {missing}')
            print(f'Known: {list(DEFAULT_METHODS.keys())}')
            sys.exit(1)
        selected_methods = {m: DEFAULT_METHODS[m] for m in requested}

    paths = iter_instances(args.data_dir, args.pattern, args.all)
    if not paths:
        print('[warn] no instances found')
        return

    print(f'Instances found: {len(paths)} (filter {args.min_n} <= n < {args.max_n})')

    all_records: List[RunRecord] = []

    for ipath in paths:
        name = os.path.basename(ipath)
        try:
            dist, n = load_or_parse(ipath)
        except Exception as e:
            print(f'{name:20s} PARSE_ERROR {e}')
            continue
        if n < args.min_n:
            print(f'{name:20s} skip n={n} < {args.min_n}')
            continue
        if n >= args.max_n:
            print(f'{name:20s} skip n={n} >= {args.max_n}')
            continue
        print(f'\nInstance {name} (n={n})')
        for mname, cfg in selected_methods.items():
            print(f'  Method {mname}')
            for r in range(1, args.runs + 1):
                seed = args.seed + r - 1
                status, cost, improvements, runtime = run_single(dist, cfg, seed, args.time_limit, ipath)
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                rec = RunRecord(
                    instance=name,
                    n=n,
                    method_name=mname,
                    method=cfg['method'],
                    ls=cfg['ls'],
                    run=r,
                    seed=seed,
                    status=status,
                    cost=None if cost != cost else cost,  # NaN guard
                    runtime=runtime,
                    improvements=improvements,
                    timestamp=ts,
                )
                all_records.append(rec)
                print(f'    run {r}/{args.runs}: status={status} cost={rec.cost} time={runtime:.3f}s imp={improvements}')
            # incremental save after each method per instance
            append_detailed(args.out_dir, [r for r in all_records if r.instance == name and r.method_name == mname])
            if _HAS_PANDAS:
                summary_df = summarize([r for r in all_records if r.instance == name])
                if summary_df is not None:
                    inc_csv = os.path.join(args.out_dir, 'heuristic_results_incremental.csv')
                    summary_df.to_csv(inc_csv, index=False)
                    print(f'  → incremental summary written ({mname})')

    # Final summary
    if _HAS_PANDAS:
        final_summary = summarize(all_records)
        if final_summary is not None:
            final_csv = os.path.join(args.out_dir, 'heuristic_results_final.csv')
            final_summary.to_csv(final_csv, index=False)
            print(f'\n✓ Final summary: {final_csv}')
            if args.summary:
                print(final_summary.head(30).to_string())
            statistical_tests(final_summary, args.out_dir)
    else:
        print('[info] Install pandas for CSV summary output (pip install pandas)')


if __name__ == '__main__':  # pragma: no cover
    main()
