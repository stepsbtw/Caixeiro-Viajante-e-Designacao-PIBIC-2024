set NODES;                       # Set of cities
param dist {NODES, NODES};       # Distance matrix

var x {i in NODES, j in NODES} binary;   # 1 if path goes from i to j
var u {i in NODES: i != 1} >= 2, <= card(NODES);  # Subtour elimination variable

minimize TotalDistance:
  sum {i in NODES, j in NODES: i != j} dist[i,j] * x[i,j];

# Each city must have exactly one outgoing arc
subject to OutDegree {i in NODES}:
  sum {j in NODES: j != i} x[i,j] = 1;

# Each city must have exactly one incoming arc
subject to InDegree {j in NODES}:
  sum {i in NODES: i != j} x[i,j] = 1;

# Subtour elimination constraints (MTZ)
subject to SubtourElim {i in NODES, j in NODES: i != j && i != 1 && j != 1}:
  u[i] - u[j] + card(NODES) * x[i,j] <= card(NODES) - 1;
