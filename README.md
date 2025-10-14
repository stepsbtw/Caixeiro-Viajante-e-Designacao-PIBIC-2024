# Integer Programming aplied to the Travelling Salesman and Assignment Problems

Codes utilized for solving Traveling Salesman Problem (TSP), Asymmetric TSP (ATSP), Assignment Problem, and Generalized Assignment Problem (GAP) using multiple solvers and heuristics.

## Repository Structure

```
src/
├── assign/               # GAP and Assignment Problem solvers
├── tsp/                  # TSP and ATSP solvers
├── analysis/             # Statistics from outputs
└── parsers/              # Instance converters
```

## Problem Types

### 1. **TSP (Traveling Salesman Problem)**
Find the shortest route visiting all cities exactly once.

### 2. **ATSP (Asymmetric TSP)**
TSP variant where distance from city A to B may differ from B to A.

### 3. **Assignment Problem**
Optimal assignment of workers to tasks minimizing total cost.

### 4. **GAP (Generalized Assignment Problem)**
Assignment with capacity constraints on workers.

## Main Components

### tsp/

#### Core Solvers:
- **`TSP_AMPL_SOLVERS.py`** - AMPL-based exact solvers (CPLEX, Gurobi, HiGHS, CBC)
- **`ATSP_AMPL_SOLVERS.py`** - AMPL-based ATSP solvers
- **`LKH_CONCORDE_BENCHMARK.py`** - State-of-the-art heuristics (LKH, Concorde)
- **`heuristic_tsp.py`** - Heuristic solver "library" (OR-Tools based)
- **`tsp_heuristic.py`** - Benchmark framework for multiple heuristics

#### Analysis Tools:
- **`analyze_tsp_benchmark.py`** - Statistical analysis of TSP results
- **`analyze_tsp_results.py`** - Results visualization and reporting
- **`combine_tsp_results.py`** - Merge results from multiple runs

#### Utilities:
- **`tsp_to_dat_converter.py`** - Convert TSPLIB to AMPL .dat format
- **`atsp_to_dat_converter.py`** - Convert ATSP instances to AMPL format

### assign/

#### Core Solvers:
- **`GAP_AMPL_SOLVERS.py`** - AMPL-based exact GAP solvers
- **`GAP_AMPL_SOLVERS_others.py`** - Additional GAP instance types
- **`ASSIGNMENT_AMPL_SOLVERS.py`** - Assignment problem solvers
- **`gap_heuristic.py`** - Comprehensive GAP heuristic framework

#### Analysis Tools:
- **`analyze_gap_results.py`** - Statistical analysis of GAP solver results
- **`benchmark_gap_heuristics.py`** - Heuristic comparison framework
- **`analyze_tsp_results.py`** - TSP results analysis

#### Utilities:
- **`gap_to_dat_converter.py`** - Convert GAP instances to AMPL format
- **`assignment_to_dat_converter.py`** - Convert assignment instances
- **`gap_others_parser.py`** - Parse alternative GAP formats
- **`assignment_sparse_parser.py`** - Parse sparse assignment matrices

## Setup and Installation

### Prerequisites
```bash

# Required packages
pip install numpy pandas scipy matplotlib seaborn ortools

# AMPL with solvers (CPLEX, Gurobi, HiGHS, CBC)
# LKH, Concorde (for TSP heuristics)
```

### Environment Setup
```bash
# Clone repository
cd ~/projects/ic

# Install Python dependencies
pip install -r requirements.txt  # Create this file

# Set AMPL path (if using AMPL solvers)
export AMPL_PATH=/path/to/ampl
```

## Usage Examples

### TSP Benchmarking

#### Run Exact Solvers on TSP Instances:
```bash
cd "TSP SCRIPTS"
python TSP_AMPL_SOLVERS.py --instances small --solvers gurobi cplex highs
```

#### Run Heuristic Methods:
```bash
cd "TSP SCRIPTS"
python tsp_heuristic.py --all --max-n 500 --runs 3 --methods ortools_lk,christofides_2opt
```

#### Run LKH/Concorde:
```bash
cd "TSP SCRIPTS"
python LKH_CONCORDE_BENCHMARK.py --pattern "*.dat" --runs 3
```

#### Analyze Results:
```bash
cd "TSP SCRIPTS"
python analyze_tsp_benchmark.py results/tsp_benchmark_results.json
```

### GAP Benchmarking

#### Run Exact Solvers:
```bash
cd "ASSIGNMENT SCRIPTS"
python GAP_AMPL_SOLVERS.py
# Edit SOLVERS and INSTANCES lists in the script
```

#### Run Heuristics:
```bash
cd "ASSIGNMENT SCRIPTS"
# Single instance
python gap_heuristic.py --file ../INSTANCES/gap/gapa --construction greedy_ratio --improvement local_search

# Batch processing
python gap_heuristic.py --dir ../INSTANCES/gap --pattern "*.txt" --output results.csv
```

#### Benchmark All Heuristics:
```bash
cd "ASSIGNMENT SCRIPTS"
python benchmark_gap_heuristics.py
```

#### Analyze Results:
```bash
cd "ASSIGNMENT SCRIPTS"
python analyze_gap_results.py ../results/gap/gap_detailed_results.json --output-dir analysis
```

### Assignment Problem

#### Run Exact Solvers:
```bash
cd "ASSIGNMENT SCRIPTS"
python ASSIGNMENT_AMPL_SOLVERS.py
```

## Data Format

### TSP Instance (.dat format):
```ampl
param n := 5;
param cost: 1 2 3 4 5 :=
1  0  20  42  35  
2  20  0  30  34
3  42  30  0  12
4  35  34  12  0
5  ...
```

### GAP Instance (.txt format):
```
m n
c11 c12 ... c1n
c21 c22 ... c2n
...
w11 w12 ... w1n
w21 w22 ... w2n
...
b1 b2 ... bm
```

## Output Files

### Results Directory Structure:
```
results/
├── gap/
│   ├── gap_detailed_results.json      # Detailed solver runs
│   ├── gap_detailed_results.txt       # Human-readable log
│   ├── gap_ampl_solvers.csv          # Summary CSV
│   └── gap_instance_summary.csv      # Per-instance best results
├── tsp_heuristics_small/
│   ├── christofides_2opt_small.json
│   └── nn_2opt_small.json
└── lkh_concorde_small/
    └── lkh_concorde_small_detailed.json
```

### JSON Result Format:
```json
{
  "instance": "gr17",
  "solver": "gurobi",
  "objective": 2085,
  "status": "solved",
  "solve_time": 0.234,
  "gap": 0.0,
  "nodes": 17
}
```

## 🔬 Algorithms Implemented

### TSP/ATSP:
- **Exact**: MTZ formulation, DFJ formulation, Branch-and-cut
- **Heuristics**: 
  - Construction: Nearest Neighbor, Christofides, Random
  - Local Search: 2-opt, 3-opt, Lin-Kernighan
  - Meta-heuristics: Genetic Algorithm, Simulated Annealing

### GAP:
- **Exact**: MIP formulation
- **Heuristics**:
  - Construction: Greedy, Greedy Ratio, Best Fit, Linear Sum Assignment
  - Improvement: Local Search, 2-opt, VNS, Simulated Annealing

### Assignment:
- **Exact**: Hungarian Algorithm, MIP formulation
- **Heuristics**: Greedy assignment

## Testing and Validation

### Run Test Suite:
```bash
# Individual problem type tests are in deprecated/test_*.py
# Main validation is through benchmark scripts
```

## Configuration

### AMPL Solver Configuration:
Edit in respective `*_AMPL_SOLVERS.py` files:
```python
SOLVERS = ['gurobi', 'cplex', 'highs', 'cbc']
TIME_LIMIT = 300  # seconds
RUNS = 3         # repetitions per instance
```

### Heuristic Configuration:
Edit in `tsp_heuristic.py` or `gap_heuristic.py`:
```python
METHODS = ['ortools_lk', 'christofides_2opt', 'nearest_2opt']
TIME_LIMIT = 100  # seconds per run
RUNS = 5         # seeds per method per instance
```

## Analysis

### Generated Reports:
- Performance rankings by solver/method
- Success rates and solution quality
- Runtime statistics (mean, median, std)
- Gap distributions and boxplots

## Instance Sources

### TSPLIB:
Standard TSP benchmark instances from [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)

### OR-Library:
GAP and Assignment instances from [Beasley's OR-Library](http://people.brunel.ac.uk/~mastjjb/jeb/info.html)

## 📚 References

### Papers/Methods:
- Christofides algorithm for TSP
- Lin-Kernighan heuristic
- MTZ and DFJ formulations
- Hungarian algorithm for assignment

### Dependencies:
- [OR-Tools](https://developers.google.com/optimization) - Google's optimization toolkit
- [AMPL](https://ampl.com/) - Mathematical programming language
- [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/) - TSP instances
