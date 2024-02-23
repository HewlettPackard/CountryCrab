import numpy as np
import pandas as pd
import os
import typing as t
import math

from countrycrab.compiler import map_camsat
from countrycrab.analyze import vector_its

import campie
import cupy as cp


def camsat(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    # config contains parameters to optimize, params are fixed

    # Check GPUs are available.
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        raise RuntimeError(
            f"No GPUs available. Please, set `CUDA_VISIBLE_DEVICES` environment variable."
        )
    # get configuration. This is part of the scheduler search space
    instance_addr = config["instance"]
    # noise is the standard deviation of noise applied to the make_values
    noise = config.get("noise", 0.5)

    # load instance and map it to the CAMSAT arrays
    tcam_array, ram_array = map_camsat(instance_addr)
    clauses = tcam_array.shape[0]
    variables = tcam_array.shape[1]

    # number of clauses that can map to each core
    n_words = params.get("n_words", clauses)
    # total number of cores
    n_cores = params.get("n_cores", 1)
    
    # get parameters. This should be "fixed values"
    # max runs is the number of parallel initialization (different inputs)
    max_runs = params.get("max_runs", 1000)
    # max_flips is the maximum number of iterations
    max_flips = params.get("max_flips", 1000)
    # scheduling is the way the cores are used
    scheduling = params.get("scheduling", "fill_first")
    # noise profile
    noise_dist = params.get("noise_distribution",'normal')
    # target_probability
    p_solve = params.get("p_solve", 0.99)
    # task is the type of task to be performed
    task = params.get("task", "debug")



    # generate random inputs
    inputs = cp.random.randint(2, size=(max_runs, variables)).astype(cp.float32)

    tcam = cp.asarray(tcam_array, dtype=cp.float32)
    ram = cp.asarray(ram_array, dtype=cp.float32)
    
    # hyperparemeters can be provided by the user
    if "hp_location" in params:
        optimized_hp = pd.read_csv(params["hp_location"])
        # take the correct hp for the size of the problem
        filtered_df = optimized_hp[(optimized_hp["N_V"] == variables)]            
        noise = filtered_df["noise"].values[0]
        max_flips = int(filtered_df["max_flips_max"].values[0])



    if scheduling == "fill_first":
        needed_cores = math.ceil(tcam.shape[0] / n_words)
        if n_cores < needed_cores:
            raise ValueError(
                f"Not enough CAMSAT cores available for mapping the instance: clauses={clauses}, n_cores={n_cores}, n_words={n_words}, needed_cores={needed_cores}"
            )

        # potentially reduce the amount of cores used to the actually needed amount
        n_cores = needed_cores

        # extend tcam and ram so they can be divided by n_cores
        if clauses % n_cores != 0:
            padding = n_cores * n_words - tcam.shape[0]
            tcam = cp.concatenate(
                (tcam, cp.full((padding, variables), cp.nan)), dtype=cp.float32
            )
            ram = cp.concatenate(
                (ram, cp.full((padding, variables), 0)), dtype=cp.float32
            )

    elif scheduling == "round_robin":
        core_size = math.ceil(tcam.shape[0] / n_cores)

        # create potentialy uneven splits, that's why we need a python list
        tcam_list = cp.array_split(tcam, n_cores)
        ram_list = cp.array_split(ram, n_cores)

        # even out the sizes of each core via padding
        for i in range(len(tcam_list)):
            if tcam_list[i].shape[0] == core_size:
                continue

            padding = core_size - tcam_list[i].shape[0]
            tcam_list[i] = cp.concatenate(
                (tcam_list[i], cp.full((padding, variables), cp.nan)), dtype=cp.float32
            )
            ram_list[i] = cp.concatenate(
                (ram_list[i], cp.full((padding, variables), 0)), dtype=cp.float32
            )

        # finally, update the tcam and ram, with the interspersed padding now added
        tcam = cp.concatenate(tcam_list)
        ram = cp.concatenate(ram_list)

    else:
        raise ValueError(f"Unknown scheduling algorithm: {scheduling}")

    # split into cores
    tcam_cores = tcam.reshape((n_cores, -1, variables))
    ram_cores = ram.reshape((n_cores, -1, variables))

    # note, to speed up the code violated_constr_mat does not represent the violated constraints but the unsatisfied variables. It doesn't matter for the overall computation of p_vs_t
    violated_constr_mat = cp.full((max_runs, max_flips), cp.nan, dtype=cp.float32)

    # tracks the amount of iteratiosn that are actually completed
    n_iters = 0

    for it in range(max_flips - 1):
        n_iters += 1

        # global
        violated_clauses = campie.tcam_match(inputs, tcam)
        make_values = violated_clauses @ ram
        
        violated_constr = cp.sum(make_values > 0, axis=1)
        violated_constr_mat[:, it] = violated_constr

        # early stopping
        if cp.sum(violated_constr_mat[:, it]) == 0:
            break

        if n_cores == 1:
            # there is no difference between the global matches and the core matches (i.e. violated_clauses)
            # if there is only one core. we can just copy the global results and
            # and wrap a single core dimension around them
            violated_clauses, make_values, violated_constr = map(
                lambda x: x[cp.newaxis, :],
                [violated_clauses, make_values, violated_constr],
            )
        else:
            # otherwise, actually compute the matches (violated_clauses) for each core
            violated_clauses = campie.tcam_match(inputs, tcam_cores)
            make_values = violated_clauses @ ram_cores
            violated_constr = cp.sum(make_values > 0, axis=2)
        
        if noise_dist == 'normal':
            # add gaussian noise to the make values
            make_values += noise * cp.random.randn(*make_values.shape, dtype=make_values.dtype)  
        elif noise_dist == 'uniform':
            # add uniform noise. Note that the standard deviation is modulated by sqrt(3)
            make_values += cp.random.uniform(low=-noise*np.sqrt(3), high=noise*np.sqrt(3), size=make_values.shape, dtype=make_values.dtype) 
        elif noise_dist == 'intrinsic':
            # add noise considering automated annealing. Noise comes from memristor devices
            make_values += noise * cp.sqrt(make_values) * cp.random.randn(*make_values.shape, dtype=make_values.dtype)
        else:
            raise ValueError(f"Unknown noise distribution: {noise_dist}")

        # select highest values
        update = cp.argmax(make_values, axis=2)
        update[cp.where(violated_constr == 0)] = -1

        if n_cores == 1:
            # only one core, no need to do random picks
            update = update[0]
        else:
            # reduction -> randomly selecting one update
            update = update.T
            random_indices = cp.random.randint(0, update.shape[1], size=update.shape[0])
            update = update[cp.arange(update.shape[0]), random_indices]

        # update inputs
        campie.flip_indices(inputs, update[:, cp.newaxis])
    
    # probability os solving the problem as a function of the iterations
    p_vs_t = cp.sum(violated_constr_mat[:, 1 : n_iters + 1] == 0, axis=0) / max_runs
    p_vs_t = cp.asnumpy(p_vs_t)
    # check if the problem was solved at least one
    solved = (np.sum(p_vs_t) > 0)

    # Compute iterations to solution for 99% of probability to solve the problem
    iteration_vector = np.arange(1, n_iters + 1)
    its = vector_its(iteration_vector, p_vs_t, p_target=p_solve)

    if task == 'hpo':
        if solved:
            # return the best (minimum) its and the corresponding max_flips
            best_its = np.min(its[its > 0])
            best_max_flips = np.where(its == its[its > 0][np.argmin(its[its > 0])])
            return {"its_opt": best_its, "max_flips_opt": best_max_flips[0][0]}
        else:
            return {"its_opt": np.nan, "max_flips_opt": max_flips}
    
    elif task == 'solve':
        if solved:
            # return the its at the given max_flips
            return {"its": its[max_flips]}
        else:
            return {"its": np.nan}
    elif task == "debug":
        inputs = cp.asnumpy(inputs)
        return p_vs_t, violated_constr_mat, inputs
    else:
        raise ValueError(f"Unknown task: {task}")