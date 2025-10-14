# gap.mod - Generalized Assignment Problem

set WORKERS;
set TASKS;

param cost{WORKERS, TASKS};      # Custo de assignment
param capacity{WORKERS};          # Capacidade de cada worker
param weight{WORKERS, TASKS};     # Peso/requisito de cada task

var x{WORKERS, TASKS}, binary;   # 1 se worker i faz task j

maximize Total_Cost:
    sum {i in WORKERS, j in TASKS} cost[i, j] * x[i, j];

# Cada worker não pode exceder sua capacidade
subject to CapacityConstraint {i in WORKERS}:
    sum {j in TASKS} weight[i, j] * x[i, j] <= capacity[i];

# Cada task deve ser atribuída a exatamente um worker
subject to TaskAssignment {j in TASKS}:
    sum {i in WORKERS} x[i, j] = 1;
