# gap_min.mod - Generalized Assignment Problem (Minimization)

set WORKERS;
set TASKS;

param cost{WORKERS, TASKS};      # Cost of assignment
param capacity{WORKERS};          # Capacity of each worker
param weight{WORKERS, TASKS};     # Weight/requirement of each task

var x{WORKERS, TASKS}, binary;   # 1 if worker i performs task j

minimize Total_Cost:
    sum {i in WORKERS, j in TASKS} cost[i, j] * x[i, j];

# Each worker cannot exceed its capacity
subject to CapacityConstraint {i in WORKERS}:
    sum {j in TASKS} weight[i, j] * x[i, j] <= capacity[i];

# Each task must be assigned to exactly one worker
subject to TaskAssignment {j in TASKS}:
    sum {i in WORKERS} x[i, j] = 1;
