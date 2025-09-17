###
# Copyright (2024) Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###

import numpy as np
import pandas as pd
import os
import typing as t
import json

from inspect import getmembers, isfunction

from countrycrab.compiler import compile_MNSAT, compile_GNSAT,compile_memHNN
from countrycrab.analyze import vector_its, vector_tts
import countrycrab.heuristics
import warnings

import cupy as cp


def solve(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    # config contains parameters to optimize, params are fixed

    # Check GPUs are available.
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        warnings.warn(
            "No GPUs available. `CUDA_VISIBLE_DEVICES` is not set. Running on CPU."
        )


    # the compiler returns an architecuture
    # in the case of a single core that is just the selected problem mapped to in-memory computing arrays
    # in the case of multiple cores, the compiler returns a multidimensional array with the mapping of the problem to the cores

    compiler_name = config.get("compiler", 'compile_MNSAT')
    compilers_dict = {
        'compile_MNSAT': compile_MNSAT,
        'compile_GNSAT': compile_GNSAT,
        'compile_memHNN':compile_memHNN
    }
    compiler_function = compilers_dict.get(compiler_name)
    architecture, params = compiler_function(config, params)

    
    # hyperparemeters can be provided by the user
    if "hp_location" in params:
        with open(params["hp_location"], 'r') as f:  
            optimized_hp = json.load(f)
        if params['variables'] in optimized_hp['N_V']:
            # Get the index of N in 'N_V'
            index = optimized_hp['N_V'].index(params['variables'])
            # Retrieve the corresponding noise
            config['noise'] = optimized_hp['noise'][index]
            params['max_flips'] = int(optimized_hp['max_flips_median'][index])
        else:
            raise ValueError(f"Number of variables {params['variables']} not found in the hyperparameters file {params['hp_location']}")        

    
    # load the heuristic function from a separate file
    heuristic_name = config.get("heuristic", 'MNSAT')
    heuristics_dict = {
        name: fn
        for name, fn in getmembers(countrycrab.heuristics, isfunction)
        if name[0] != "_" # "private" members
    }
    heuristic_function = heuristics_dict.get(heuristic_name)
    if heuristic_function is None:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")

    # check if compiler and heuristic are compatible
    heuristic_to_compiler = {
        'MNSAT' : 'compile_MNSAT',
        'GNSAT' : 'compile_GNSAT',
        'walksat_skc' : 'compile_GNSAT',
        'walksat_b' : 'compile_GNSAT',
        'memHNN' : 'compile_memHNN',
        'pbits_ising' : 'compile_memHNN'
    }
    if compiler_name != heuristic_to_compiler.get(heuristic_name):
        raise ValueError(f"Compiler {compiler_name} is not compatible with heuristic {heuristic_name}")

    # call the heuristic function with the necessary arguments
    results = heuristic_function(architecture, config, params)
    metric_tracking_mat = results[0]
    n_iters = results[1]
    inputs = results[2]
    iterations_timepoints = results[3]
    if len(results) > 4:
        overall_best_solution = results[4]
        overall_best_metric = results[5]

    # METRIC FUNCTIONS
    # target_probability
    p_solve = params.get("p_solve", 0.99)
    # task is the type of task to be performed
    task = params.get("task", "debug")
    # max runs is the number of parallel initialization (different inputs)
    max_runs = params.get("max_runs", 100)
    # max_flips is the maximum number of iterations
    max_flips = params.get("max_flips", 1000)
    # metric to use
    metric = params.get("metric", "its")
    # metric approach
    metric_approach = params.get("metric_approach", "frequentist")

    # tts computation precision
    tts_deltatime = params.get("tts_deltatime", params.get("Tclk", 6e-9))

    if metric == "its":
        # probability of solving the problem as a function of the iterations
        p_vs_it = cp.sum(metric_tracking_mat[:, :n_iters] == 0, axis=0) / max_runs
        # probability of solving the problem as a function of the time
        # Notes:
        #   For continuous-time solvers, the runs may have different time between all flips, because they occur asynchronously
        #   We stored the violated_constr_mat per iteration (== flip), thus we must decouple the iteration and time per run
        ts = cp.arange(np.nanmin(iterations_timepoints), np.nanmax(iterations_timepoints) + tts_deltatime, tts_deltatime)
        time_to_iter = cp.apply_along_axis( # use apply_along_axis instead of vectorization to prevent needing literal TBs of (V)RAM
            lambda tps: cp.sum(ts[:, np.newaxis] >= tps, axis=1) - 1,
            arr = iterations_timepoints,
            axis = 1
        )
        p_vs_t = cp.mean(p_vs_it[time_to_iter], axis=0)
        # check if the problem was solved at least one
        solved = (cp.sum(p_vs_it) > 0).item()
        # get the arrays from the GPU to the CPU
        p_vs_it = cp.asnumpy(p_vs_it)
        p_vs_t = cp.asnumpy(p_vs_t)
        ts = cp.asnumpy(ts)
        iterations_timepoints = cp.asnumpy(iterations_timepoints)
        
        if metric_approach == "frequentist":
            # Compute iterations to solution for 99% of probability to solve the problem
            iteration_vector = np.arange(1, len(p_vs_it)+1)
            its = vector_its(iteration_vector, p_vs_it, p_target=p_solve)
            tts = vector_tts(ts, p_vs_t, p_target=p_solve)

            if task == 'hpo':
                if solved:
                    # return the best (minimum) its and the corresponding max_flips
                    argbest_its = np.argmin(its[its > 0])
                    best_max_flips = np.where(its == its[its > 0][argbest_its])
                    return {
                        "its": its[its > 0][argbest_its],
                        "max_flips_opt": best_max_flips[0][0],
                        "opt_tts": np.nanmin(tts)
                    }
                else:
                    return {
                        "its": np.nan,
                        "max_flips_opt": max_flips,
                        "opt_tts": np.nan,
                    }
            
            elif task == 'solve':
                if solved:
                    return {"its": its[-2], "opt_tts": tts[-2]}
                else:
                    return {"its": np.nan, "tts": np.nan}
                
            elif task == "debug":
                return p_vs_it, cp.asnumpy(metric_tracking_mat), cp.asnumpy(inputs), (ts, p_vs_t)
            else:
                raise ValueError(f"Unknown task: {task}")
        
        elif metric_approach == "bayesian":
            try:
                from countrycrab.metrics import vector_its_bayesian
                its, its_err = vector_its_bayesian(cp.asnumpy(metric_tracking_mat[:, 1 : n_iters + 1]), config)

            except ImportError:
                its = np.nan
                its_err = np.nan
            return {"its": its, "its_err": its_err}
        
        elif metric_approach == "diversity":
            # Here we want to study how many different solutions we get and what's the frequency and ITS of each one
            # We need to return a dictionary with the different frequency and ITS
            # violated_constr_mat is a matrix with the shape (max_runs, n_iters)
            # input is a matrix with the shape (max_runs, variables)
            if solved:
                # Step 1: Find where violated_constr_mat is zero for each run
                solved_runs = cp.where(metric_tracking_mat[:, 1:n_iters+1] == 0)
                solved_runs = solved_runs[0].get()  # Explicit conversion to NumPy array
                inputs = cp.asnumpy(inputs)
                # Step 2: Count the number of different solutions. 
                unique_solutions = np.unique(inputs[solved_runs], axis=0)
                num_solutions = len(unique_solutions)
            
                # Step 3: Compute the frequency of each solution
                # Step 3: Compute the frequency of each solution
                frequency = np.zeros(num_solutions)
                for i, solution in enumerate(unique_solutions):
                    frequency[i] = np.sum(np.all(inputs == solution, axis=1))
                
                # Step 4: Compute the ITS as usual
                iteration_vector = np.arange(1, len(p_vs_it)+1)
                its = vector_its(iteration_vector, p_vs_it, p_target=p_solve)

            if task == 'hpo':
                if solved:
                    # return the best (minimum) its and the corresponding max_flips
                    best_its = np.min(its[its > 0])
                    best_max_flips = np.where(its == its[its > 0][np.argmin(its[its > 0])])
                    return {"its": best_its, "max_flips_opt": best_max_flips[0][0], "unique_solutions": unique_solutions, "frequency": frequency}
                else:
                    return {"its": np.nan, "max_flips_opt": max_flips, "unique_solutions": np.nan, "frequency": np.nan}
            
            elif task == 'solve':
                if solved:
                    # return the its at the given max_flips
                    return {"its": its[-2], "unique_solutions": unique_solutions, "frequency": frequency}
                else:
                    return {"its": np.nan, "unique_solutions": np.nan, "frequency": np.nan}
    elif metric=='energy':
        if task =='hpo':
            # we don't care about the actual results, we just want to return the energy
            # return the best (minimum) energy and the corresponding max_flips
            mean_energy = np.mean(metric_tracking_mat, axis=0)
            argbest_energy = np.argmin(mean_energy)
            best_max_flips = np.where(mean_energy == mean_energy[argbest_energy])
            return {
                "energy": mean_energy[argbest_energy],
                "max_flips_opt": best_max_flips[0][0],
                "best_energy": np.min(metric_tracking_mat),
                "best_solution": overall_best_solution if 'overall_best_solution' in locals() else None,
                "best_metric": overall_best_metric if 'overall_best_metric' in locals() else None
            }
            

        
