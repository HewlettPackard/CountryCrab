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


from countrycrab.compiler import compile_walksat_m, compile_walksat_g
from countrycrab.analyze import vector_its
from countrycrab.heuristics import walksat_m, walksat_g, walksat_skc, walksat_b

import cupy as cp


def solve(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    # config contains parameters to optimize, params are fixed

    # Check GPUs are available.
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        raise RuntimeError(
            f"No GPUs available. Please, set `CUDA_VISIBLE_DEVICES` environment variable."
        )


    # the compiler returns an architecuture
    # in the case of a single core that is just the selected problem mapped to in-memory computing arrays
    # in the case of multiple cores, the compiler returns a multidimensional array with the mapping of the problem to the cores

    compiler_name = config.get("compiler", 'compile_walksat_m')
    compilers_dict = {
        'compile_walksat_m': compile_walksat_m,
        'compile_walksat_g': compile_walksat_g
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
    heuristic_name = config.get("heuristic", 'walksat_m')
    heuristics_dict = {
        'walksat_m': walksat_m,
        'walksat_g': walksat_g,
        'walksat_skc': walksat_skc,
        'walksat_b': walksat_b
    }
    heuristic_function = heuristics_dict.get(heuristic_name)
    if heuristic_function is None:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")

    # check if compiler and heuristic are compatible
    heuristic_to_compiler = {
        'walksat_m' : 'compile_walksat_m',
        'walksat_g' : 'compile_walksat_g',
        'walksat_skc' : 'compile_walksat_g',
        'walksat_b' : 'compile_walksat_g',
    }
    if compiler_name != heuristic_to_compiler.get(heuristic_name):
        raise ValueError(f"Compiler {compiler_name} is not compatible with heuristic {heuristic_name}")

    # call the heuristic function with the necessary arguments
    violated_constr_mat, n_iters, inputs = heuristic_function(architecture, config,params)

    # METRIC FUNCTIONS
    # target_probability
    p_solve = params.get("p_solve", 0.99)
    # task is the type of task to be performed
    task = params.get("task", "debug")
    # max runs is the number of parallel initialization (different inputs)
    max_runs = params.get("max_runs", 100)
    # max_flips is the maximum number of iterations
    max_flips = params.get("max_flips", 1000)

    # probability os solving the problem as a function of the iterations
    p_vs_t = cp.sum(violated_constr_mat[:, 1 : n_iters + 1] == 0, axis=0) / max_runs
    p_vs_t = cp.asnumpy(p_vs_t)
    # check if the problem was solved at least one
    solved = (np.sum(p_vs_t) > 0)

    # Compute iterations to solution for 99% of probability to solve the problem
    iteration_vector = np.arange(1, len(p_vs_t)+1)
    its = vector_its(iteration_vector, p_vs_t, p_target=p_solve)

    if task == 'hpo':
        if solved:
            # return the best (minimum) its and the corresponding max_flips
            best_its = np.min(its[its > 0])
            best_max_flips = np.where(its == its[its > 0][np.argmin(its[its > 0])])
            return {"its": best_its, "max_flips_opt": best_max_flips[0][0]}
        else:
            return {"its": np.nan, "max_flips_opt": max_flips}
    
    elif task == 'solve':
        if solved:
            # return the its at the given max_flips
            return {"its": its[-2]}
        else:
            return {"its": np.nan}
    elif task == "debug":
        inputs = cp.asnumpy(inputs)
        return p_vs_t, cp.asnumpy(violated_constr_mat), cp.asnumpy(inputs)
    else:
        raise ValueError(f"Unknown task: {task}")