# assignment.mod

set WORKERS;
set TASKS;

param cost{WORKERS, TASKS};

var x{WORKERS, TASKS}, binary;

minimize Total_Cost:
    sum {i in WORKERS, j in TASKS} cost[i, j] * x[i, j];

subject to OneTaskPerWorker {i in WORKERS}:
    sum {j in TASKS} x[i, j] = 1;

subject to OneWorkerPerTask {j in TASKS}:
    sum {i in WORKERS} x[i, j] = 1;
