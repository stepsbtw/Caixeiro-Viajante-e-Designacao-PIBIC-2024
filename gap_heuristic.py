#!/usr/bin/env python3
"""
GAP Heuristics: Construction and Improvement Methods for Generalized Assignment Problem

This script implements various heuristics for the Generalized Assignment Problem (GAP):

1. Construction heuristics:
   - Greedy assignment (cost-based and cost/weight ratio)
   - Random assignment
   - Minimum cost assignment
   - Best fit decreasing

2. Improvement heuristics:
   - Local search (task reassignment)
   - 2-opt style exchanges
   - Variable neighborhood search
   - Simulated annealing

3. Support for GAP instance formats
4. Batch processing capabilities
5. Multiple restarts for better solutions

Usage:
    # Greedy construction with local search
    python gap_heuristic.py --file gap1 --construction greedy --improvement local_search
    
    # Random construction with simulated annealing
    python gap_heuristic.py --file gap5 --construction random --improvement simulated_annealing
    
    # Best fit with variable neighborhood search and multiple restarts
    python gap_heuristic.py --file gapa --construction best_fit --improvement vns --multi_start 10
    
    # Batch processing all instances
    python gap_heuristic.py --dir INSTANCES/gap --pattern "*.txt"
"""

import argparse
import math
import os
import time
import glob
import random
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

# Import for Linear Sum Assignment (Hungarian algorithm)
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Linear Sum Assignment method will be disabled.")


@dataclass
class GAPResult:
    """Container for GAP solution results."""
    instance_name: str
    assignment: List[int]  # assignment[j] = worker assigned to task j
    objective_value: float
    is_feasible: bool
    construction_value: float
    improvement_iterations: int
    total_runtime: float
    construction_time: float
    improvement_time: float
    method: str
    workers_used: int
    max_capacity_utilization: float


class GAPInstance:
    """Representation of a GAP instance."""
    
    def __init__(self, n_workers: int, n_tasks: int, costs: np.ndarray, 
                 weights: np.ndarray, capacities: np.ndarray, is_maximization: bool = True):
        self.n_workers = n_workers
        self.n_tasks = n_tasks
        self.costs = costs  # costs[i][j] = cost of assigning task j to worker i
        self.weights = weights  # weights[i][j] = weight of task j on worker i
        self.capacities = capacities  # capacities[i] = capacity of worker i
        self.is_maximization = is_maximization
        
    def is_feasible(self, assignment: List[int]) -> bool:
        """Check if an assignment is feasible."""
        if len(assignment) != self.n_tasks:
            return False
            
        worker_loads = [0.0] * self.n_workers
        
        for task, worker in enumerate(assignment):
            if worker < 0 or worker >= self.n_workers:
                return False
            worker_loads[worker] += self.weights[worker][task]
            
        return all(load <= capacity for load, capacity in zip(worker_loads, self.capacities))
    
    def is_partial_feasible(self, assignment: List[int]) -> bool:
        """Check if a partial assignment (with -1 values) is feasible for assigned tasks."""
        if len(assignment) != self.n_tasks:
            return False
            
        worker_loads = [0.0] * self.n_workers
        
        for task, worker in enumerate(assignment):
            if worker == -1:  # Skip unassigned tasks
                continue
            if worker < 0 or worker >= self.n_workers:
                return False
            worker_loads[worker] += self.weights[worker][task]
            
        return all(load <= capacity for load, capacity in zip(worker_loads, self.capacities))
    
    def get_objective_value(self, assignment: List[int]) -> float:
        """Calculate objective value for an assignment."""
        if not self.is_feasible(assignment):
            return float('-inf') if self.is_maximization else float('inf')
            
        total_cost = sum(self.costs[worker][task] for task, worker in enumerate(assignment))
        return total_cost
    
    def get_worker_load(self, worker: int, assignment: List[int]) -> float:
        """Get current load for a worker given an assignment."""
        return sum(self.weights[worker][task] for task, assigned_worker in enumerate(assignment) 
                  if assigned_worker == worker)
    
    def can_assign(self, task: int, worker: int, assignment: List[int]) -> bool:
        """Check if a task can be assigned to a worker."""
        current_load = self.get_worker_load(worker, assignment)
        return current_load + self.weights[worker][task] <= self.capacities[worker]


class GAPParser:
    """Parser for GAP instance files."""
    
    @staticmethod
    def parse_gap_file(filename: str, problem_index: int = 0) -> GAPInstance:
        """Parse a GAP file and return a GAPInstance."""
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        # Determine if this is maximization or minimization based on filename
        is_maximization = 'gap' in os.path.basename(filename).lower() and not any(
            x in os.path.basename(filename).lower() for x in ['gapa', 'gapb', 'gapc', 'gapd']
        )
        
        # Check if this is the "others" format (single problem) or multi-problem format
        first_line_parts = lines[0].split()
        if len(first_line_parts) == 2:
            # This is "others" format - single problem, starts with m n directly
            return GAPParser._parse_single_problem(lines, 0, is_maximization)
        else:
            # This is multi-problem format like gap1.txt
            # First line is number of problems
            num_problems = int(lines[0])
            line_idx = 1
            
            # Skip to the desired problem
            for prob_num in range(problem_index + 1):
                if prob_num == problem_index:
                    # Parse this problem
                    return GAPParser._parse_single_problem(lines, line_idx, is_maximization)
                    
                else:
                    # Skip this problem
                    m, n = map(int, lines[line_idx].split())
                    line_idx += 1
                    
                    # Skip cost matrix
                    cost_values_needed = m * n
                    cost_values_read = 0
                    while cost_values_read < cost_values_needed and line_idx < len(lines):
                        numbers = len(lines[line_idx].split())
                        cost_values_read += numbers
                        line_idx += 1
                    
                    # Skip weight matrix
                    weight_values_needed = m * n
                    weight_values_read = 0
                    while weight_values_read < weight_values_needed and line_idx < len(lines):
                        numbers = len(lines[line_idx].split())
                        weight_values_read += numbers
                        line_idx += 1
                    
                    # Skip capacities
                    cap_values_needed = m
                    cap_values_read = 0
                    while cap_values_read < cap_values_needed and line_idx < len(lines):
                        numbers = len(lines[line_idx].split())
                        cap_values_read += numbers
                        line_idx += 1
            
            raise ValueError(f"Problem index {problem_index} not found in file {filename}")
    
    @staticmethod
    def _parse_single_problem(lines: List[str], start_idx: int, is_maximization: bool) -> GAPInstance:
        """Parse a single GAP problem from lines starting at start_idx."""
        line_idx = start_idx
        
        # Parse m, n
        m, n = map(int, lines[line_idx].split())  # workers, tasks
        line_idx += 1
        
        # Parse cost matrix (m rows of n values each)
        cost_matrix = []
        for i in range(m):
            row_data = []
            while len(row_data) < n and line_idx < len(lines):
                numbers = list(map(int, lines[line_idx].split()))
                row_data.extend(numbers)
                line_idx += 1
            
            cost_matrix.append(row_data[:n])
            
            # Handle overflow - if we read too many numbers in last line
            if len(row_data) > n:
                excess = row_data[n:]
                line_idx -= 1
                lines[line_idx] = ' '.join(map(str, excess))
        
        # Parse requirement/weight matrix (m rows of n values each)
        weight_matrix = []
        for i in range(m):
            row_data = []
            while len(row_data) < n and line_idx < len(lines):
                numbers = list(map(int, lines[line_idx].split()))
                row_data.extend(numbers)
                line_idx += 1
            
            weight_matrix.append(row_data[:n])
            
            # Handle overflow
            if len(row_data) > n:
                excess = row_data[n:]
                line_idx -= 1
                lines[line_idx] = ' '.join(map(str, excess))
        
        # Parse capacities (m values)
        cap_data = []
        while len(cap_data) < m and line_idx < len(lines):
            numbers = list(map(int, lines[line_idx].split()))
            cap_data.extend(numbers)
            line_idx += 1
        
        capacities = cap_data[:m]
        
        # Convert to numpy arrays
        costs = np.array(cost_matrix)
        weights = np.array(weight_matrix)
        capacities = np.array(capacities)
        
        return GAPInstance(m, n, costs, weights, capacities, is_maximization)


class GAPHeuristics:
    """Collection of heuristic methods for GAP."""
    
    def __init__(self, instance: GAPInstance, seed: Optional[int] = None):
        self.instance = instance
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def greedy_construction(self, use_ratio: bool = True) -> List[int]:
        """
        Greedy construction heuristic.
        
        Args:
            use_ratio: If True, use cost/weight ratio; if False, use cost only
        """
        assignment = [-1] * self.instance.n_tasks
        
        # Create list of (task, worker, priority) tuples
        candidates = []
        for task in range(self.instance.n_tasks):
            for worker in range(self.instance.n_workers):
                if use_ratio and self.instance.weights[worker][task] > 0:
                    if self.instance.is_maximization:
                        priority = self.instance.costs[worker][task] / self.instance.weights[worker][task]
                    else:
                        priority = -self.instance.costs[worker][task] / self.instance.weights[worker][task]
                else:
                    priority = self.instance.costs[worker][task]
                    if not self.instance.is_maximization:
                        priority = -priority
                
                candidates.append((task, worker, priority))
        
        # Sort by priority (descending for best first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Assign tasks greedily
        for task, worker, priority in candidates:
            if assignment[task] == -1 and self.instance.can_assign(task, worker, assignment):
                assignment[task] = worker
        
        # Handle unassigned tasks (if any) with a simple heuristic
        unassigned_tasks = [t for t in range(self.instance.n_tasks) if assignment[t] == -1]
        
        for task in unassigned_tasks:
            best_worker = -1
            best_ratio = float('inf')  # We want the least loaded worker for unassigned tasks
            
            for worker in range(self.instance.n_workers):
                if self.instance.can_assign(task, worker, assignment):
                    current_load = self.instance.get_worker_load(worker, assignment)
                    load_ratio = current_load / self.instance.capacities[worker]
                    
                    if load_ratio < best_ratio:
                        best_ratio = load_ratio
                        best_worker = worker
            
            if best_worker != -1:
                assignment[task] = best_worker

        # Finalize: ensure all tasks are assigned and repair if needed
        if any(a == -1 for a in assignment):
            assignment = self._force_assign_all_tasks(assignment)
        assignment = self._repair_capacity_violations(assignment)
        # If still infeasible, try a feasibility-focused packer (especially for minimization sets)
        if not self.instance.is_feasible(assignment):
            assignment = self._feasible_pack_min()
        return assignment
    
    def linear_sum_assignment_construction(self) -> List[int]:
        """
        Linear Sum Assignment (Hungarian algorithm) heuristic construction.
        
        Transforms the GAP into a square assignment problem by creating artificial
        workers/tasks with high costs and applies the Hungarian algorithm.
        
        The approach:
        1. Replicate each worker into multiple "slots" to allow multiple tasks per worker
           (slots_i ~= floor(cap_i / min_j w_ij), capped to n_tasks). Ensure total slots >= n_tasks.
        2. Build a rectangular cost matrix [total_slots x n_tasks]. Add extra dummy slots if needed
           with prohibitive costs so they are chosen only if unavoidable.
        3. Run Hungarian on this matrix (it supports rectangular). This assigns every task to exactly
           one slot or to a dummy slot.
        4. Map slots -> workers, then repair capacity violations. If any tasks remain unassigned,
           force-assign them with least overload increase and repair again.
        """
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not available. Falling back to greedy construction.")
            return self.greedy_construction(use_ratio=True)
        
        n_workers = self.instance.n_workers
        n_tasks = self.instance.n_tasks
        costs = self.instance.costs
        weights = self.instance.weights
        capacities = self.instance.capacities

        # Build slots per worker based on the smallest positive weight they can process
        slots_per_worker: List[int] = [0] * n_workers
        for i in range(n_workers):
            # Find smallest positive weight for worker i
            pos_weights = [w for w in weights[i] if w > 0]
            if len(pos_weights) == 0:
                # If all weights are 0, cap by n_tasks (can take any number of tasks)
                min_w = 1.0
            else:
                min_w = max(1e-9, float(min(pos_weights)))

            # If even the lightest task doesn't fit, give 0 slots
            if capacities[i] < min_w:
                slots = 0
            else:
                slots = int(capacities[i] // min_w)

            # Cap slots to n_tasks to avoid explosion
            slots_per_worker[i] = int(min(n_tasks, max(0, slots)))

        total_slots = int(sum(slots_per_worker))
        # Ensure at least one slot if all are zero to allow any assignment attempt
        if total_slots == 0:
            # Give one slot to the worker with best average cost
            avg_scores = []
            for i in range(n_workers):
                if self.instance.is_maximization:
                    avg_scores.append((i, float(np.mean(costs[i]))))
                else:
                    avg_scores.append((i, -float(np.mean(costs[i]))))
            avg_scores.sort(key=lambda x: x[1], reverse=True)
            if avg_scores:
                slots_per_worker[avg_scores[0][0]] = min(1, n_tasks)
            total_slots = int(sum(slots_per_worker))

        # If we still have fewer slots than tasks, distribute extra slots round-robin
        if total_slots < n_tasks:
            extra = n_tasks - total_slots
            # Distribute in cycles to avoid bias
            idx = 0
            while extra > 0:
                slots_per_worker[idx % n_workers] = min(n_tasks, slots_per_worker[idx % n_workers] + 1)
                extra -= 1
                idx += 1
            total_slots = int(sum(slots_per_worker))

        # With expanded slots, we should not need dummy slots, but keep as safeguard
        dummy_slots = 0
        if total_slots < n_tasks:
            dummy_slots = n_tasks - total_slots

        # Penalty cost much larger than any real cost (in minimization sense)
        max_abs_cost = float(np.max(np.abs(costs))) if costs.size > 0 else 1.0
        penalty = max_abs_cost * 1e6 + 1e3

        # Create cost matrix: rows = total_slots + dummy_slots, cols = n_tasks
        rows = total_slots + dummy_slots
        slot_to_worker = []  # map row index -> worker id (or -1 for dummy)
        # Populate real slots
        for i in range(n_workers):
            for _ in range(slots_per_worker[i]):
                slot_to_worker.append(i)
        # Append dummy slots
        for _ in range(dummy_slots):
            slot_to_worker.append(-1)

        C = np.full((rows, n_tasks), penalty, dtype=float)
        for r, wkr in enumerate(slot_to_worker):
            if wkr == -1:
                # dummy row: keep penalty in all columns
                continue
            for j in range(n_tasks):
                # If a single task doesn't fit on this worker at all, keep penalty
                if weights[wkr][j] > capacities[wkr]:
                    continue
                val = costs[wkr][j]
                C[r, j] = -val if self.instance.is_maximization else val

        try:
            row_ind, col_ind = linear_sum_assignment(C)
        except Exception as e:
            print(f"Error in Linear Sum Assignment: {e}")
            print("Falling back to greedy construction.")
            return self.greedy_construction(use_ratio=True)

        # Build initial assignment from slots
        assignment = [-1] * n_tasks
        for r, c in zip(row_ind, col_ind):
            if c < n_tasks:
                wkr = slot_to_worker[r]
                if wkr != -1:
                    assignment[c] = wkr

        # Try to greedily assign any unassigned tasks without violating capacity
        for task in range(n_tasks):
            if assignment[task] != -1:
                continue
            best_worker = -1
            best_score = float('-inf') if self.instance.is_maximization else float('inf')
            for worker in range(n_workers):
                if self.instance.can_assign(task, worker, assignment):
                    val = costs[worker][task]
                    if self.instance.is_maximization:
                        if val > best_score:
                            best_score = val
                            best_worker = worker
                    else:
                        if val < best_score:
                            best_score = val
                            best_worker = worker
            if best_worker != -1:
                assignment[task] = best_worker

        # First repair pass to resolve overloads by moving tasks
        assignment = self._repair_capacity_violations(assignment)

        # Iteratively assign remaining tasks using residual-capacity slots
        max_fill_iters = 3
        for _ in range(max_fill_iters):
            unassigned = [t for t in range(n_tasks) if assignment[t] == -1]
            if not unassigned:
                break
            # Compute residual capacities
            loads = [self.instance.get_worker_load(i, assignment) for i in range(n_workers)]
            residual = [max(0.0, float(capacities[i] - loads[i])) for i in range(n_workers)]
            # Build slots for residual
            res_slots_per_worker: List[int] = [0] * n_workers
            for i in range(n_workers):
                pos_weights = [w for w in weights[i] if w > 0]
                min_w = max(1e-9, float(min(pos_weights))) if pos_weights else 1.0
                if residual[i] < min_w:
                    res_slots_per_worker[i] = 0
                else:
                    res_slots_per_worker[i] = int(min(len(unassigned), residual[i] // min_w))
            total_res_slots = int(sum(res_slots_per_worker))
            if total_res_slots == 0:
                break
            # Build matrix for only unassigned tasks
            rows2 = total_res_slots
            C2 = np.full((rows2, len(unassigned)), penalty, dtype=float)
            slot_to_worker2 = []
            for i in range(n_workers):
                for _ in range(res_slots_per_worker[i]):
                    slot_to_worker2.append(i)
            # Fill feasible edges only
            for r, wkr in enumerate(slot_to_worker2):
                for idx, j in enumerate(unassigned):
                    if weights[wkr][j] <= residual[wkr]:
                        val = costs[wkr][j]
                        C2[r, idx] = -val if self.instance.is_maximization else val
            try:
                r2, c2 = linear_sum_assignment(C2)
            except Exception:
                break
            # Apply new assignments when they improve from unassigned
            any_assigned = False
            for rr, cc in zip(r2, c2):
                if cc < len(unassigned):
                    j = unassigned[cc]
                    wkr = slot_to_worker2[rr]
                    # Check fits with current assignment
                    if self.instance.can_assign(j, wkr, assignment):
                        assignment[j] = wkr
                        any_assigned = True
            if not any_assigned:
                break
            # Repair lightly after batch
            assignment = self._repair_capacity_violations(assignment)

        # If still unassigned tasks remain, force-assign with least overload increase and repair again
        if any(a == -1 for a in assignment):
            assignment = self._force_assign_all_tasks(assignment)
            assignment = self._repair_capacity_violations(assignment)

        # Final safety: if infeasible, fall back to greedy ratio construction
        if not self.instance.is_feasible(assignment):
            # Try feasibility packer first
            assignment2 = self._feasible_pack_min()
            if self.instance.is_feasible(assignment2):
                return assignment2
            return self.greedy_construction(use_ratio=True)

        return assignment
    
    def _repair_capacity_violations(self, assignment: List[int]) -> List[int]:
        """
        Repair capacity violations by moving tasks between workers.
        
        Strategy:
        1. Identify overloaded workers
        2. Move heaviest tasks from overloaded workers to workers with capacity
        3. If no capacity available, unassign tasks
        4. Try to reassign unassigned tasks greedily
        """
        max_iterations = 200
        
        for iteration in range(max_iterations):
            # Check for capacity violations
            violations = []
            
            for worker in range(self.instance.n_workers):
                current_load = self.instance.get_worker_load(worker, assignment)
                if current_load > self.instance.capacities[worker]:
                    violation = current_load - self.instance.capacities[worker]
                    violations.append((worker, violation))
            
            if not violations:
                break  # No more violations
            
            # Sort by violation severity (largest first)
            violations.sort(key=lambda x: x[1], reverse=True)
            
            made_improvement = False
            
            for worker, violation in violations:
                # Get tasks assigned to this worker, sorted by weight (heaviest first)
                worker_tasks = []
                for task in range(self.instance.n_tasks):
                    if assignment[task] == worker:
                        weight = self.instance.weights[worker][task]
                        cost = self.instance.costs[worker][task]
                        # Priority: remove heavy tasks with low value first
                        if self.instance.is_maximization:
                            priority = weight / (cost + 1)  # Higher weight/cost ratio = higher priority to remove
                        else:
                            priority = weight * (cost + 1)  # Higher weight*cost = higher priority to remove
                        worker_tasks.append((task, weight, priority))
                
                worker_tasks.sort(key=lambda x: x[2], reverse=True)
                
                # Try to move tasks to other workers
                for task, weight, _ in worker_tasks:
                    if violation <= 0:
                        break
                    
                    # Find alternative worker for this task
                    best_alternative = -1
                    best_score = float('-inf') if self.instance.is_maximization else float('inf')
                    
                    for alt_worker in range(self.instance.n_workers):
                        if alt_worker == worker:
                            continue
                        
                        # Check if alternative worker can handle this task
                        if self.instance.can_assign(task, alt_worker, assignment):
                            cost = self.instance.costs[alt_worker][task]
                            current_load = self.instance.get_worker_load(alt_worker, assignment)
                            remaining_capacity = self.instance.capacities[alt_worker] - current_load
                            
                            # Score combines cost and remaining capacity
                            if self.instance.is_maximization:
                                score = cost + 0.1 * remaining_capacity
                                if score > best_score:
                                    best_score = score
                                    best_alternative = alt_worker
                            else:
                                score = -cost + 0.1 * remaining_capacity
                                if score > best_score:
                                    best_score = score
                                    best_alternative = alt_worker
                    
                    if best_alternative != -1:
                        # Move task to alternative worker
                        assignment[task] = best_alternative
                        violation -= weight
                        made_improvement = True
                        # continue attempting moves to resolve remaining violation
                        continue
                    else:
                        # No alternative found, unassign task
                        assignment[task] = -1
                        violation -= weight
                        made_improvement = True
                        # continue removing tasks if needed
                        continue
                
                if made_improvement:
                    break
            
            if not made_improvement:
                break
        
        # Try to assign unassigned tasks
        unassigned_tasks = [t for t in range(self.instance.n_tasks) if assignment[t] == -1]
        
        # First pass: direct feasible assignment
        for task in list(unassigned_tasks):
            best_worker = -1
            best_score = float('-inf') if self.instance.is_maximization else float('inf')
            for worker in range(self.instance.n_workers):
                if self.instance.can_assign(task, worker, assignment):
                    cost = self.instance.costs[worker][task]
                    if self.instance.is_maximization:
                        if cost > best_score:
                            best_score = cost
                            best_worker = worker
                    else:
                        if cost < best_score:
                            best_score = cost
                            best_worker = worker
            if best_worker != -1:
                assignment[task] = best_worker
                unassigned_tasks.remove(task)

        # Second pass: one-swap insertion to place remaining tasks
        if unassigned_tasks:
            # Precompute loads
            loads = [self.instance.get_worker_load(w, assignment) for w in range(self.instance.n_workers)]
            for task in list(unassigned_tasks):
                placed = False
                w_weights = self.instance.weights
                w_costs = self.instance.costs
                for i in range(self.instance.n_workers):
                    wi = w_weights[i][task]
                    # Fast accept
                    if loads[i] + wi <= self.instance.capacities[i]:
                        assignment[task] = i
                        loads[i] += wi
                        placed = True
                        break
                    # Try one-swap
                    need = loads[i] + wi - self.instance.capacities[i]
                    # Consider tasks currently at i
                    cand_tasks = [t for t in range(self.instance.n_tasks) if assignment[t] == i]
                    best_triplet = None
                    best_delta = float('-inf') if self.instance.is_maximization else float('inf')
                    for tprime in cand_tasks:
                        wt = w_weights[i][tprime]
                        # If removing tprime frees enough
                        for k in range(self.instance.n_workers):
                            if k == i:
                                continue
                            # Check k can take tprime
                            if loads[k] + w_weights[k][tprime] <= self.instance.capacities[k]:
                                # Would i be ok after moving tprime and adding task?
                                if loads[i] - wt + wi <= self.instance.capacities[i]:
                                    # Evaluate objective delta
                                    delta = (w_costs[i][task] - w_costs[i][tprime]) + w_costs[k][tprime]
                                    if self.instance.is_maximization:
                                        if delta > best_delta:
                                            best_delta = delta
                                            best_triplet = (i, tprime, k)
                                    else:
                                        if delta < best_delta:
                                            best_delta = delta
                                            best_triplet = (i, tprime, k)
                    if best_triplet is not None:
                        i2, tprime2, k2 = best_triplet
                        # Apply move and place task
                        assignment[tprime2] = k2
                        loads[k2] += w_weights[k2][tprime2]
                        loads[i2] -= w_weights[i2][tprime2]
                        assignment[task] = i2
                        loads[i2] += wi
                        placed = True
                        break
                if placed:
                    unassigned_tasks.remove(task)
        
        
        return assignment

    def _force_assign_all_tasks(self, assignment: List[int]) -> List[int]:
        """
        Ensure every task is assigned by choosing the worker that minimizes overload increase.
        Used as a last resort before a final repair pass.
        """
        n_workers = self.instance.n_workers
        n_tasks = self.instance.n_tasks
        costs = self.instance.costs
        weights = self.instance.weights
        capacities = self.instance.capacities

        # Current loads
        loads = [self.instance.get_worker_load(i, assignment) for i in range(n_workers)]

        for task in range(n_tasks):
            if assignment[task] != -1:
                continue

            best_choice = (-1, float('inf'), float('-inf'))  # (worker, overload_after, value_score)
            for worker in range(n_workers):
                w = weights[worker][task]
                if w > capacities[worker]:
                    continue  # cannot ever fit
                new_load = loads[worker] + w
                overload_after = max(0.0, new_load - capacities[worker])
                value_score = costs[worker][task] if self.instance.is_maximization else -costs[worker][task]

                # Prefer zero overload, otherwise smallest overload, tie-break by value
                better = False
                if overload_after < best_choice[1]:
                    better = True
                elif math.isclose(overload_after, best_choice[1]):
                    if value_score > best_choice[2]:
                        better = True
                if better:
                    best_choice = (worker, overload_after, value_score)

            worker, overload_after, _ = best_choice
            if worker != -1:
                assignment[task] = worker
                loads[worker] += weights[worker][task]
        return assignment

    def _feasible_pack_min(self) -> List[int]:
        """
        Feasibility-first constructive heuristic for minimization sets:
        - Order tasks by decreasing minimal weight across workers (hardest first).
        - Assign each task to the feasible worker with minimum cost, breaking ties by most remaining capacity.
        - If no feasible worker exists, assign to worker with least overload increase.
        - Run a repair pass at the end.
        """
        n_workers = self.instance.n_workers
        n_tasks = self.instance.n_tasks
        costs = self.instance.costs
        weights = self.instance.weights
        capacities = self.instance.capacities

        assignment = [-1] * n_tasks
        loads = [0.0] * n_workers

        # Task order: decreasing min weight
        min_w = [min(weights[i][j] for i in range(n_workers)) for j in range(n_tasks)]
        order = sorted(range(n_tasks), key=lambda j: (min_w[j], -j), reverse=True)

        for j in order:
            # Feasible workers
            feas = []
            for i in range(n_workers):
                if loads[i] + weights[i][j] <= capacities[i]:
                    feas.append(i)
            if feas:
                # pick smallest weight to help feasibility, tie-break by min cost then most remaining cap
                best_i = min(
                    feas,
                    key=lambda i: (weights[i][j], costs[i][j], -(capacities[i] - loads[i] - weights[i][j]))
                )
                assignment[j] = best_i
                loads[best_i] += weights[best_i][j]
            else:
                # Force to worker with least overload increase, tie-break by smallest weight then min cost
                best_choice = None
                best_tuple = None
                for i in range(n_workers):
                    new_load = loads[i] + weights[i][j]
                    overload = max(0.0, new_load - capacities[i])
                    t = (overload, weights[i][j], costs[i][j])
                    if best_tuple is None or t < best_tuple:
                        best_tuple = t
                        best_choice = i
                assignment[j] = best_choice
                loads[best_choice] += weights[best_choice][j]

        # Repair
        assignment = self._repair_capacity_violations(assignment)
        return assignment
    
    def local_search_improvement(self, assignment: List[int], max_iterations: int = 1000) -> Tuple[List[int], int]:
        """
        Local search improvement using task reassignment.
        
        Returns:
            Tuple of (improved_assignment, iterations)
        """
        current_assignment = assignment.copy()
        current_value = self.instance.get_objective_value(current_assignment)
        iterations = 0
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try to reassign each task to a different worker
            for task in range(self.instance.n_tasks):
                current_worker = current_assignment[task]
                best_worker = current_worker
                best_value = current_value
                
                for new_worker in range(self.instance.n_workers):
                    if new_worker != current_worker:
                        # Temporarily reassign task
                        temp_assignment = current_assignment.copy()
                        temp_assignment[task] = new_worker
                        
                        if self.instance.is_feasible(temp_assignment):
                            temp_value = self.instance.get_objective_value(temp_assignment)
                            
                            if ((self.instance.is_maximization and temp_value > best_value) or
                                (not self.instance.is_maximization and temp_value < best_value)):
                                best_value = temp_value
                                best_worker = new_worker
                
                # Apply improvement if found
                if best_worker != current_worker:
                    current_assignment[task] = best_worker
                    current_value = best_value
                    improved = True
            
            iterations += 1
            if not improved:
                break
        
        return current_assignment, iterations
    
    def two_opt_improvement(self, assignment: List[int], max_iterations: int = 500) -> Tuple[List[int], int]:
        """
        2-opt style improvement by swapping task assignments.
        
        Returns:
            Tuple of (improved_assignment, iterations)
        """
        current_assignment = assignment.copy()
        current_value = self.instance.get_objective_value(current_assignment)
        iterations = 0
        
        for iteration in range(max_iterations):
            improved = False
            
            # Try swapping assignments of pairs of tasks
            for i in range(self.instance.n_tasks):
                for j in range(i + 1, self.instance.n_tasks):
                    # Swap assignments
                    temp_assignment = current_assignment.copy()
                    temp_assignment[i], temp_assignment[j] = temp_assignment[j], temp_assignment[i]
                    
                    if self.instance.is_feasible(temp_assignment):
                        temp_value = self.instance.get_objective_value(temp_assignment)
                        
                        if ((self.instance.is_maximization and temp_value > current_value) or
                            (not self.instance.is_maximization and temp_value < current_value)):
                            current_assignment = temp_assignment
                            current_value = temp_value
                            improved = True
                            break
                
                if improved:
                    break
            
            iterations += 1
            if not improved:
                break
        
        return current_assignment, iterations
    
    def variable_neighborhood_search(self, assignment: List[int], max_iterations: int = 100) -> Tuple[List[int], int]:
        """
        Variable Neighborhood Search (VNS) improvement.
        
        Returns:
            Tuple of (improved_assignment, iterations)
        """
        current_assignment = assignment.copy()
        current_value = self.instance.get_objective_value(current_assignment)
        iterations = 0
        
        for iteration in range(max_iterations):
            improved = False
            
            # Neighborhood 1: Single task reassignment
            temp_assignment, _ = self.local_search_improvement(current_assignment, 50)
            temp_value = self.instance.get_objective_value(temp_assignment)
            
            if ((self.instance.is_maximization and temp_value > current_value) or
                (not self.instance.is_maximization and temp_value < current_value)):
                current_assignment = temp_assignment
                current_value = temp_value
                improved = True
            else:
                # Neighborhood 2: Task swapping
                temp_assignment, _ = self.two_opt_improvement(current_assignment, 25)
                temp_value = self.instance.get_objective_value(temp_assignment)
                
                if ((self.instance.is_maximization and temp_value > current_value) or
                    (not self.instance.is_maximization and temp_value < current_value)):
                    current_assignment = temp_assignment
                    current_value = temp_value
                    improved = True
            
            iterations += 1
            if not improved:
                break
        
        return current_assignment, iterations
    
    def simulated_annealing(self, assignment: List[int], max_iterations: int = 1000, 
                          initial_temp: float = 100.0, cooling_rate: float = 0.95) -> Tuple[List[int], int]:
        """
        Simulated Annealing improvement.
        
        Returns:
            Tuple of (improved_assignment, iterations)
        """
        current_assignment = assignment.copy()
        best_assignment = assignment.copy()
        current_value = self.instance.get_objective_value(current_assignment)
        best_value = current_value
        temperature = initial_temp
        iterations = 0
        
        for iteration in range(max_iterations):
            # Generate neighbor by reassigning a random task
            neighbor_assignment = current_assignment.copy()
            task = random.randint(0, self.instance.n_tasks - 1)
            
            # Find feasible workers for this task
            feasible_workers = [w for w in range(self.instance.n_workers) 
                              if w != current_assignment[task]]
            
            if feasible_workers:
                new_worker = random.choice(feasible_workers)
                neighbor_assignment[task] = new_worker
                
                if self.instance.is_feasible(neighbor_assignment):
                    neighbor_value = self.instance.get_objective_value(neighbor_assignment)
                    
                    # Accept or reject the neighbor
                    if self.instance.is_maximization:
                        delta = neighbor_value - current_value
                    else:
                        delta = current_value - neighbor_value
                    
                    if delta > 0 or random.random() < math.exp(delta / temperature):
                        current_assignment = neighbor_assignment
                        current_value = neighbor_value
                        
                        # Update best solution
                        if ((self.instance.is_maximization and neighbor_value > best_value) or
                            (not self.instance.is_maximization and neighbor_value < best_value)):
                            best_assignment = neighbor_assignment.copy()
                            best_value = neighbor_value
            
            # Cool down
            temperature *= cooling_rate
            iterations += 1
        
        return best_assignment, iterations


class GAPSolver:
    """Main solver class for GAP heuristics."""
    
    def __init__(self, instance: GAPInstance, seed: Optional[int] = None):
        self.instance = instance
        self.heuristics = GAPHeuristics(instance, seed)
    
    def solve(self, construction_method: str = 'greedy', 
              improvement_method: str = 'local_search',
              multi_start: int = 1) -> GAPResult:
        """
        Solve GAP instance using specified heuristics.
        
        Args:
            construction_method: 'greedy', 'random', 'best_fit', 'minimum_cost'
            improvement_method: 'local_search', 'two_opt', 'vns', 'simulated_annealing', 'none'
            multi_start: Number of restarts for construction heuristics
        """
        start_time = time.time()
        
        best_assignment = None
        best_value = float('-inf') if self.instance.is_maximization else float('inf')
        total_improvement_iterations = 0
        
        # Multiple starts
        for start in range(multi_start):
            # Construction phase
            construction_start = time.time()
            
            if construction_method == 'greedy' or construction_method == 'greedy_ratio':
                assignment = self.heuristics.greedy_construction(use_ratio=True)
            elif construction_method == 'greedy_cost':
                assignment = self.heuristics.greedy_construction(use_ratio=False)
            elif construction_method == 'lsa' or construction_method == 'linear_sum_assignment':
                # For minimization sets, use a feasibility-robust constructor for now
                if not self.instance.is_maximization:
                    assignment = self.heuristics.greedy_construction(use_ratio=True)
                else:
                    assignment = self.heuristics.linear_sum_assignment_construction()
            else:
                raise ValueError(f"Unknown construction method: {construction_method}")
                
            construction_time = time.time() - construction_start
            construction_value = self.instance.get_objective_value(assignment)
            
            # Improvement phase
            improvement_start = time.time()
            improvement_iterations = 0
            
            if improvement_method == 'local_search':
                assignment, improvement_iterations = self.heuristics.local_search_improvement(assignment)
            elif improvement_method == 'two_opt':
                assignment, improvement_iterations = self.heuristics.two_opt_improvement(assignment)
            elif improvement_method == 'vns':
                assignment, improvement_iterations = self.heuristics.variable_neighborhood_search(assignment)
            elif improvement_method == 'simulated_annealing':
                assignment, improvement_iterations = self.heuristics.simulated_annealing(assignment)
            elif improvement_method == 'none':
                pass
            else:
                raise ValueError(f"Unknown improvement method: {improvement_method}")
            
            improvement_time = time.time() - improvement_start
            
            # Check if this is the best solution so far
            current_value = self.instance.get_objective_value(assignment)
            if ((self.instance.is_maximization and current_value > best_value) or
                (not self.instance.is_maximization and current_value < best_value)):
                best_assignment = assignment
                best_value = current_value
                best_construction_value = construction_value
                best_construction_time = construction_time
                best_improvement_time = improvement_time
                total_improvement_iterations = improvement_iterations
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        is_feasible = self.instance.is_feasible(best_assignment) if best_assignment else False
        workers_used = len(set(best_assignment)) if best_assignment else 0
        
        max_capacity_utilization = 0.0
        if best_assignment and is_feasible:
            for worker in range(self.instance.n_workers):
                load = self.instance.get_worker_load(worker, best_assignment)
                utilization = load / self.instance.capacities[worker] if self.instance.capacities[worker] > 0 else 0
                max_capacity_utilization = max(max_capacity_utilization, utilization)
        
        method_name = f"{construction_method}+{improvement_method}"
        
        return GAPResult(
            instance_name="",
            assignment=best_assignment if best_assignment else [],
            objective_value=best_value,
            is_feasible=is_feasible,
            construction_value=best_construction_value if best_assignment else 0,
            improvement_iterations=total_improvement_iterations,
            total_runtime=total_time,
            construction_time=best_construction_time if best_assignment else 0,
            improvement_time=best_improvement_time if best_assignment else 0,
            method=method_name,
            workers_used=workers_used,
            max_capacity_utilization=max_capacity_utilization
        )


def solve_single_instance(filename: str, construction_method: str, 
                         improvement_method: str, multi_start: int = 1, 
                         seed: Optional[int] = None, problem_index: int = 0) -> GAPResult:
    """Solve a single GAP instance."""
    
    # Parse instance
    try:
        instance = GAPParser.parse_gap_file(filename, problem_index)
        print(f"Loaded {os.path.basename(filename)} (problem {problem_index}): {instance.n_workers} workers, {instance.n_tasks} tasks")
        print(f"Problem type: {'Maximization' if instance.is_maximization else 'Minimization'}")
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None
    
    # Solve instance
    solver = GAPSolver(instance, seed)
    result = solver.solve(construction_method, improvement_method, multi_start)
    result.instance_name = f"{os.path.basename(filename)}_p{problem_index}"
    
    return result


def print_result(result: GAPResult):
    """Print solution result in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Instance: {result.instance_name}")
    print(f"Method: {result.method}")
    print(f"{'='*60}")
    print(f"Objective Value: {result.objective_value:.2f}")
    print(f"Feasible: {result.is_feasible}")
    print(f"Workers Used: {result.workers_used}")
    print(f"Max Capacity Utilization: {result.max_capacity_utilization:.2%}")
    print(f"Construction Value: {result.construction_value:.2f}")
    print(f"Improvement: {result.objective_value - result.construction_value:.2f}")
    print(f"Improvement Iterations: {result.improvement_iterations}")
    print(f"Total Runtime: {result.total_runtime:.3f}s")
    print(f"  Construction: {result.construction_time:.3f}s")
    print(f"  Improvement: {result.improvement_time:.3f}s")
    
    if result.assignment and len(result.assignment) <= 20:  # Only show assignment for small instances
        print(f"Assignment: {result.assignment}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='GAP Heuristics Solver')
    
    # Input specification
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', type=str, help='Single GAP instance file (without .txt extension)')
    group.add_argument('--dir', type=str, help='Directory containing GAP instances')
    
    # Heuristic methods
    parser.add_argument('--construction', type=str, default='greedy_ratio',
                       choices=['greedy', 'greedy_ratio', 'greedy_cost', 'lsa', 'linear_sum_assignment'],
                       help='Construction heuristic method')
    
    parser.add_argument('--improvement', type=str, default='local_search',
                       choices=['local_search', 'two_opt', 'vns', 'simulated_annealing', 'none'],
                       help='Improvement heuristic method')
    
    # Additional options
    parser.add_argument('--multi_start', type=int, default=1,
                       help='Number of restarts for construction heuristics')
    
    parser.add_argument('--pattern', type=str, default='*.txt',
                       help='File pattern for batch processing')
    
    parser.add_argument('--problem', type=int, default=0,
                       help='Problem index within the file (default: 0)')
    
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file for results')
    
    args = parser.parse_args()
    
    results = []
    
    if args.file:
        # Single file processing
        filename = args.file
        
        # If it's just a name without path and extension, assume it's in INSTANCES/gap/
        if '/' not in filename and not filename.endswith('.txt'):
            filename = f"INSTANCES/gap/{args.file}.txt"
        
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return
            
        result = solve_single_instance(filename, args.construction, args.improvement, 
                                     args.multi_start, args.seed, args.problem)
        if result:
            print_result(result)
            results.append(result)
    
    elif args.dir:
        # Batch processing
        pattern = os.path.join(args.dir, args.pattern)
        files = sorted(glob.glob(pattern))
        
        if not files:
            print(f"No files found matching pattern: {pattern}")
            return
        
        print(f"Processing {len(files)} instances...")
        
        for filename in files:
            print(f"\nProcessing {os.path.basename(filename)}...")
            result = solve_single_instance(filename, args.construction, args.improvement,
                                         args.multi_start, args.seed, args.problem)
            if result:
                print_result(result)
                results.append(result)
    
    # Save results to CSV if requested
    if args.output and results:
        import pandas as pd
        
        df_data = []
        for result in results:
            df_data.append({
                'instance': result.instance_name,
                'method': result.method,
                'objective_value': result.objective_value,
                'is_feasible': result.is_feasible,
                'workers_used': result.workers_used,
                'max_capacity_utilization': result.max_capacity_utilization,
                'construction_value': result.construction_value,
                'improvement': result.objective_value - result.construction_value,
                'improvement_iterations': result.improvement_iterations,
                'total_runtime': result.total_runtime,
                'construction_time': result.construction_time,
                'improvement_time': result.improvement_time
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
    
    # Summary statistics
    if len(results) > 1:
        feasible_results = [r for r in results if r.is_feasible]
        if feasible_results:
            avg_objective = sum(r.objective_value for r in feasible_results) / len(feasible_results)
            avg_runtime = sum(r.total_runtime for r in feasible_results) / len(feasible_results)
            avg_improvement = sum(r.improvement_iterations for r in feasible_results) / len(feasible_results)
            
            print(f"\n{'='*60}")
            print(f"SUMMARY ({len(feasible_results)}/{len(results)} feasible solutions)")
            print(f"{'='*60}")
            print(f"Average Objective Value: {avg_objective:.2f}")
            print(f"Average Runtime: {avg_runtime:.3f}s")
            print(f"Average Improvement Iterations: {avg_improvement:.1f}")
            print(f"Feasibility Rate: {len(feasible_results)/len(results):.1%}")


if __name__ == "__main__":
    main()
