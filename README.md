# CountryCrab 🦀 
## Introduction
CountryCrab is a distributed simulator for physics-inspired optimization solvers. 
It utilize both multiprocessing and GPU parallelization to maximize the performance (i.e. flips/s).
CountryCrab was first used to benchmark Content Addressable Memories approaches to the solution of SAT solvers [[1]](#ref1) and [[2]](#ref2).

## Setting up the Jupyter Docker container

It is reccommended to use CountryCrab in the provided container.
The `dockerctl` script provides options to control building and running the image.
Running can be done in two modes:
- **lab**: (default) launches Jupyter Lab inside the container.
- **shell**: drops you into an interactive shell session without starting Jupyter.
Usage examples (from the `docker` directory):

```sh
# build an image tagged `countrycrab`
./dockerctl build countrycrab

# launch Jupyter Lab on port 8888 (default)
./dockerctl run countrycrab

# launch Jupyter Lab on a custom port (e.g. 9999)
./dockerctl run countrycrab 9999

# drop into a shell session instead of launching Jupyter Lab
./dockerctl run countrycrab <port> shell
```
After the container starts:

- In **lab** mode, you’ll be prompted to set a Jupyter Lab password, and the script will print connection URLs.
- In **shell** mode, you get a full-featured bash prompt inside the container. When you’re done, type `exit` (or press Ctrl-D) to end the session and automatically clean up the container.

## Installing CountryCrab

In the case the container is not used, to install the CountryCrab package run
```sh
pip install -e .
```

## Optional: Ising Machine CPU Solver
CountryCrab can be used with `ising-machine-cpu`, a CPU-based solver. To include it, add it as a submodule:
```sh
git submodule add https://github.com/lucashmorais/ising-machine-cpu submodules/ising-machine-cpu
git submodule update --init --recursive
```

## Basic usage
An example of basic usage for CountryCrab can be found in `tests/basic_usage.ipynb`.
The first step is to create a configuration and parameters for the experiment.
The only necessary field is `instance` in the configuration file which is the path to the instance to be solved.
If not other parameters are specified, defaults one will be used.

After creating a configuraiton and parameters the `countrycrab.solver` can be run with
```
p_vs_t, violated_constr_mat, inputs = solver.solve(config = config,params = params)
```
with `p_vs_t` a vector representing the solution probability as a function of iteration, 
`violated_constr_mat` the number of violated clauses as a funciton of iteration for each run,
and `inputs` the optimized input for each run.
Note that `countrycrab.solver` run on GPU(s) through `CuPy` calls, thus the available GPU(s) should be specified with
```
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
```

## Scheduler usage
The scheduler takes a configuration, which usually contains multiple instances and parameters, and parallelize them in multiple processes with `multiprocessing`.
`ray` is used to schedule the various experiments and `mlflow` to track the result.
An example of the scheduler usage is shown in `tests/scheduler_usage.ipynb`.

After creating a configuration file for the experiment, then the experiment is run with
```
python3 countrycrab/scheduler.py --tracking_uri=mlflow_tracking_uri --config=path_to_configuration_file
```

## Reprint results from [[2]](#ref2)
Code and data to reprint the results in [[2]](#ref2) can be found in `data/npj_uc_figures`.

---

## References

1. <a id="ref1"></a> **Pedretti, G., et al.** "Zeroth and higher-order logic with content addressable memories." *2023 International Electron Devices Meeting (IEDM)*. IEEE, 2023. [DOI](https://doi.org/10.1109/IEDM45741.2023.10413853)

2. <a id="ref2"></a> **Pedretti, G., et al.** "Solving Boolean satisfiability problems with resistive content addressable memories." *npj Unconventional Computing*, 2024. [DOI](https://doi.org/10.1038/s44335-025-00020-w)
