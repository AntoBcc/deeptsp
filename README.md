# Solving the Travelling Salesman Problem with Deep Learning and Belief Propagation

First part of research thesis - MSc DSBA 

Implementation of a belief propagation algorithm for minimum-weight perfect 2-matching, in order to use the resulting approximation within a *graph neural network* to solve the TSP. 

Convergence diagnostics and overlap with the TSP solutions available in `test/`.
Data generated using the **Concorde TSP solver**, using [this script](https://github.com/CarloLucibello/benchmarking-gnns/blob/ab/dev/data/TSP/generate_TSP.py).


## Instructions

To create and use the virtual environment (same needed for the GNN part of the project)
```
#if the environment has not been created yet
conda env create -f deeptsp.yml -n deeptsp

#once created
conda activate deeptsp
```

To run BP experiments on tsp30-50:

```
# At the root of the project
bash run.sh
```

The bash script relies on
```
src/run_BP.py
src/BP.py
```



