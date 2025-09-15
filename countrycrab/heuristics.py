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
from typing import Dict, List, Tuple
from decimal import Decimal
import math
import csv

import campie
import cupy as cp

from numba import cuda, float32, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import warnings
import toml
import subprocess


def MNSAT(architecture, config, params):

    tcam = architecture[0]
    ram = architecture[1]
    tcam_cores = architecture[2]
    ram_cores = architecture[3]

    # get parameters. This should be "fixed values"
    # max runs is the number of parallel initialization (different inputs)
    max_runs = params.get("max_runs", 100)
    # max_flips is the maximum number of iterations
    max_flips = params.get("max_flips", 1000)
    # noise profile
    noise_dist = params.get("noise_distribution",'normal')
    # number of cores
    n_cores = params.get("n_cores", 1)
    # variables
    variables = tcam.shape[1]


    # get configuration. This is part of the scheduler search space
    # noise is the standard deviation of noise applied to the make_values
    noise = config.get('noise',0.8)

    
    # note, to speed up the code violated_constr_mat does not represent the violated constraints but the unsatisfied variables. It doesn't matter for the overall computation of p_vs_t
    violated_constr_mat = cp.full((max_runs, max_flips), cp.nan, dtype=cp.float32)

    # generate random inputs
    inputs = cp.random.randint(2, size=(max_runs, variables)).astype(cp.float32)
    # tracks the amount of iteratiosn that are actually completed
    n_iters = 0

    for it in range(max_flips):
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

    iterations_timepoints = cp.arange(1, n_iters + 1) * params.get("Tclk", 6e-9)
    return violated_constr_mat, n_iters, inputs, iterations_timepoints[np.newaxis, :]


def GNSAT(architecture, config, params):

    ramf_array = architecture[0]
    ramb_array = architecture[1]
    superTile_varIndices = architecture[2]
    num_superTiles = architecture[3]
    
    max_runs = params.get("max_runs", 100)
    max_flips = params.get("max_flips", 1000)
    n_cores = params.get("n_cores", 1)
    noise_dist = params.get("noise_distribution",'normal')

    clauses = ramf_array.shape[1]
    variables = int(ramf_array.shape[0]/2)
    literals = 2*variables

    noise = config.get('noise',0.8)

    var_inputs = cp.random.randint(2, size=(max_runs, variables)).astype(cp.float32)
    lit_inputs = cp.zeros((max_runs,literals)).astype(cp.float32)
    lit_inputs[:,2*cp.arange(0,variables,1)]=var_inputs
    lit_inputs[:,2*cp.arange(0,variables,1)+1]=cp.abs(var_inputs-1)

    ramf = cp.asarray(ramf_array, dtype=cp.float32)
    ramb = cp.asarray(ramb_array, dtype=cp.float32)
    
    violated_constr_mat = cp.full((max_runs, max_flips), cp.nan, dtype=cp.float32)

    n_iters = 0

    for it in range(max_flips - 1):
        n_iters += 1

        # global
        lit_inputs = cp.zeros((max_runs,literals)).astype(cp.float32)
        lit_inputs[:,2*cp.arange(0,variables,1)]=var_inputs
        lit_inputs[:,2*cp.arange(0,variables,1)+1]=cp.abs(var_inputs-1)
        f_val = lit_inputs @ ramf
        s_val = cp.zeros((max_runs,clauses))
        z_val = cp.zeros((max_runs,clauses))

        # s_val is the value of the unsatisfied clauses
        s_val[cp.where(f_val==0)] = 1
        # z_val is the value of satified clauses with only one literal = true
        z_val[cp.where(f_val==1)] = 1
        a = s_val @ ramb
        b = z_val @ ramb
        lit_one_indices = cp.where(lit_inputs==1)
        lit_zero_indices = cp.where(lit_inputs==0)
        # neg_lit_indices = 2*cp.arange(0,variables,1)+1 # do you need it?
        
        # compute the gain values
        mv_arr = cp.reshape(a[lit_zero_indices[0],lit_zero_indices[1]],(max_runs,variables))
        bv_arr = cp.reshape(b[lit_one_indices[0],lit_one_indices[1]],(max_runs,variables))
        g_arr = mv_arr - bv_arr
        y = g_arr
        violated_constr = cp.sum(mv_arr > 0, axis=1)
        violated_constr_mat[:, it] = violated_constr

        # early stopping
        if cp.sum(violated_constr_mat[:, it]) == 0:
            break 

        # add noise
        if noise_dist == 'bernoulli':
            if num_superTiles==1:
                all_variable_indices = cp.reshape(cp.arange(0,variables,1),(1,variables))
                all_variable_indices = cp.repeat(all_variable_indices,max_runs,axis=0)
                all_variable_indices[mv_arr < 1] = -1
                all_variable_indices_sorted = cp.sort(-all_variable_indices,axis=1)
                all_variable_indices_sorted = -all_variable_indices_sorted
        
                violated_constr_temp = cp.copy(violated_constr)
                violated_constr_temp[violated_constr == 0] = 1
            else:
                superTile_eligibility = cp.zeros((max_runs,num_superTiles))
                for i in range(num_superTiles):
                    superTile_eligibility[:,i] = cp.heaviside(cp.sum(mv_arr[:,superTile_varIndices[i]],axis=1),0)
        
                selectedSuperTile = cp.argmax(cp.multiply(cp.random.uniform(0.01,1,size=(max_runs,num_superTiles)),superTile_eligibility),axis=1)
                selected_VarIndices = [superTile_varIndices[int(i)] for i in list(selectedSuperTile)]
                
                dummy_var_index = cp.zeros((max_runs,variables))
                allIters = cp.arange(max_runs).reshape((max_runs,1))
                dummy_var_index[allIters,selected_VarIndices[:]] = 1
                
                y[mv_arr < 1] = -100
                y[dummy_var_index<1] = -100
        
                tiled_mv_arr = mv_arr[allIters,selected_VarIndices[:]]
                #all_variable_indices = cp.array(cp.copy(selected_VarIndices))
                all_variable_indices = cp.copy(cp.array(selected_VarIndices))
                all_variable_indices[tiled_mv_arr < 1] = -1
                all_variable_indices_sorted = cp.sort(-all_variable_indices,axis=1)
                all_variable_indices_sorted = -all_variable_indices_sorted
                
                violated_constr_temp = cp.sum(tiled_mv_arr > 0, axis=1)
                violated_constr_temp[violated_constr_temp == 0] = 1
                
            num_candidate_variables = cp.array(np.random.randint(0,cp.asnumpy(violated_constr_temp)))
            xy = cp.arange(0,max_runs,1)
            update2 = all_variable_indices_sorted[xy,num_candidate_variables]
            
            y, violated_constr = map(
                    lambda x: x[cp.newaxis, :],
                    [y, violated_constr],
                )
            update = cp.argmax(y, axis=2)
            update[cp.where(violated_constr == 0)] = -1
            update1 = update[0]
            
            ind1 = cp.random.uniform(0,1,max_runs)>noise
            ind2 = update1==-1
            ind3 = ind1 & ~ind2
            update1[ind3] = update2[ind3]
        else:
            if noise_dist == 'normal':
                # add gaussian noise to the make values
                y += noise * cp.random.randn(*y.shape, dtype=y.dtype)
            elif noise_dist == 'uniform':
                # add uniform noise. Note that the standard deviation is modulated by sqrt(3)
                y += cp.random.uniform(low=-noise*np.sqrt(3), high=noise*np.sqrt(3), size=y.shape, dtype=y.dtype) 
            elif noise_dist == 'intrinsic':
                # add noise considering automated annealing. Noise comes from memristor devices
                y += noise * cp.sqrt(y) * cp.random.randn(*y.shape, dtype=y.dtype)
            else:
                raise ValueError(f"Unknown noise distribution: {noise_dist}")
                
            if num_superTiles!=1:
                superTile_eligibility = cp.zeros((max_runs,num_superTiles))
                for i in range(num_superTiles):
                    superTile_eligibility[:,i] = cp.heaviside(cp.sum(mv_arr[:,superTile_varIndices[i]],axis=1),0)
        
                selectedSuperTile = cp.argmax(cp.multiply(cp.random.uniform(0.01,1,size=(max_runs,num_superTiles)),superTile_eligibility),axis=1)
                selected_VarIndices = [superTile_varIndices[int(i)] for i in list(selectedSuperTile)]
                
                dummy_var_index = cp.zeros((max_runs,variables))
                allIters = cp.arange(max_runs).reshape((max_runs,1))
                dummy_var_index[allIters,selected_VarIndices[:]] = 1
                y[dummy_var_index<1] = -100

            #y[mv_arr < 1] = -100
            y, violated_constr = map(
                    lambda x: x[cp.newaxis, :],
                    [y, violated_constr],
                )
            update = cp.argmax(y, axis=2)
            update[cp.where(violated_constr == 0)] = -1
            update1 = update[0]
                

        if n_cores == 1:
            update = update1
        else:
            # reduction -> randomly selecting one update
            update = update.T
            random_indices = cp.random.randint(0, update.shape[1], size=update.shape[0])
            update = update[cp.arange(update.shape[0]), random_indices]
        campie.flip_indices(var_inputs, update[:, cp.newaxis])
    
    iterations_timepoints = cp.arange(1, n_iters + 1) * params.get("Tclk", 6e-9)
    return violated_constr_mat, n_iters, var_inputs, iterations_timepoints[np.newaxis, :]

def walksat_skc(architecture, config, params):
    
    noise = config.get("noise", 2)
    
    ramf_array = architecture[0]
    ramb_array = architecture[1]

    max_runs = params.get("max_runs", 1000)
    max_flips = params.get("max_flips", 1000)
    n_cores = params.get("n_cores", 1)
    noise_dist = params.get("noise_distribution",'normal')    

    clauses = ramf_array.shape[1]
    variables = int(ramf_array.shape[0]/2)
    literals = 2*variables
    
    var_inputs = cp.random.randint(2, size=(max_runs, variables)).astype(cp.float32)
    lit_inputs = cp.zeros((max_runs,literals)).astype(cp.float32)
    lit_inputs[:,2*cp.arange(0,variables,1)]=var_inputs
    lit_inputs[:,2*cp.arange(0,variables,1)+1]=cp.abs(var_inputs-1)


    ramf = cp.asarray(ramf_array, dtype=cp.float32)
    ramb = cp.asarray(ramb_array, dtype=cp.float32)
    
    violated_constr_mat = cp.full((max_runs, max_flips), cp.nan, dtype=cp.float32)

    n_iters = 0

    for it in range(max_flips - 1):
        n_iters += 1

        lit_inputs = cp.zeros((max_runs,literals)).astype(cp.float32)
        lit_inputs[:,2*cp.arange(0,variables,1)]=var_inputs
        lit_inputs[:,2*cp.arange(0,variables,1)+1]=cp.abs(var_inputs-1)
        f_val = lit_inputs @ ramf
        s_val = cp.zeros((max_runs,clauses))
        z_val = cp.zeros((max_runs,clauses))
        s_val[cp.where(f_val==0)] = 1
        z_val[cp.where(f_val==1)] = 1
        a = s_val @ ramb
        b = z_val @ ramb
        lit_one_indices = cp.where(lit_inputs==1)
        lit_zero_indices = cp.where(lit_inputs==0)
        neg_lit_indices = 2*cp.arange(0,variables,1)+1
        mv_arr = cp.reshape(a[lit_zero_indices[0],lit_zero_indices[1]],(max_runs,variables))
        bv_arr = cp.reshape(b[lit_one_indices[0],lit_one_indices[1]],(max_runs,variables))
        g_arr = mv_arr - bv_arr
        y = g_arr
        violated_constr = cp.sum(mv_arr > 0, axis=1)
        violated_constr_mat[:, it] = violated_constr

        # early stopping
        if cp.sum(violated_constr_mat[:, it]) == 0:
            break 

        tmp1 = mv_arr<1
        tmp2 = bv_arr==0
        tmp3 = (~tmp1&tmp2).any(1)
        tmp4 = cp.repeat(cp.reshape(tmp3,(max_runs,1)),variables,axis=1)
        tmp5 = (tmp4 & (tmp1 | ~tmp2)) | (~tmp4 & tmp1)
        y[tmp5] = -1000
        if noise_dist == 'bernoulli':
            all_variable_indices = cp.reshape(cp.arange(0,variables,1),(1,variables))
            all_variable_indices = cp.repeat(all_variable_indices,max_runs,axis=0)
            all_variable_indices[tmp5] = -1
            all_variable_indices_sorted = cp.sort(-all_variable_indices,axis=1)
            all_variable_indices_sorted = -all_variable_indices_sorted
    
            tmp6 = cp.ones((max_runs,variables))
            tmp6[tmp5] = 0
            violated_constr_temp = cp.sum(tmp6,axis=1)
            violated_constr_temp[violated_constr_temp == 0] = 1
            num_candidate_variables = cp.array(np.random.randint(0,cp.asnumpy(violated_constr_temp)))
            xy = cp.arange(0,max_runs,1)
            update2 = all_variable_indices_sorted[xy,num_candidate_variables]
            
            y, violated_constr = map(
                    lambda x: x[cp.newaxis, :],
                    [y, violated_constr],
                )
            update = cp.argmax(y, axis=2)
            update[cp.where(violated_constr == 0)] = -1
            update1 = update[0]
            
            ind1 = cp.random.uniform(0,1,max_runs)>noise
            ind2 = update1==-1
            ind3 = ind1 & ~ind2
            update1[ind3] = update2[ind3]
        else:  
            if noise_dist == 'normal':
                # add gaussian noise to the make values
                y += noise * cp.random.randn(*y.shape, dtype=y.dtype)
            elif noise_dist == 'uniform':
                # add uniform noise. Note that the standard deviation is modulated by sqrt(3)
                y += cp.random.uniform(low=-noise*np.sqrt(3), high=noise*np.sqrt(3), size=y.shape, dtype=y.dtype) 
            elif noise_dist == 'intrinsic':
                # add noise considering automated annealing. Noise comes from memristor devices
                y += noise * cp.sqrt(y) * cp.random.randn(*y.shape, dtype=y.dtype)
            else:
                raise ValueError(f"Unknown noise distribution: {noise_dist}")

            y, violated_constr = map(
                    lambda x: x[cp.newaxis, :],
                    [y, violated_constr],
                )
            update = cp.argmax(y, axis=2)
            update[cp.where(violated_constr == 0)] = -1
            update1 = update[0]
            
        if n_cores == 1:
            update = update1
        else:
            # reduction -> randomly selecting one update
            update = update.T
            random_indices = cp.random.randint(0, update.shape[1], size=update.shape[0])
            update = update[cp.arange(update.shape[0]), random_indices]
        campie.flip_indices(var_inputs, update[:, cp.newaxis])
    
    iterations_timepoints = cp.arange(1, n_iters + 1) * params.get("Tclk", 6e-9)
    return violated_constr_mat, n_iters, var_inputs, iterations_timepoints[np.newaxis, :]

def walksat_b(architecture, config, params):
    
    noise = config.get("noise", 2)
    
    ramf_array = architecture[0]
    ramb_array = architecture[1]

    max_runs = params.get("max_runs", 1000)
    max_flips = params.get("max_flips", 1000)
    n_cores = params.get("n_cores", 1)
    noise_dist = params.get("noise_distribution",'normal')    
    
    clauses = ramf_array.shape[1]
    variables = int(ramf_array.shape[0]/2)
    literals = 2*variables
    
    var_inputs = cp.random.randint(2, size=(max_runs, variables)).astype(cp.float32)
    lit_inputs = cp.zeros((max_runs,literals)).astype(cp.float32)
    lit_inputs[:,2*cp.arange(0,variables,1)]=var_inputs
    lit_inputs[:,2*cp.arange(0,variables,1)+1]=cp.abs(var_inputs-1)


    ramf = cp.asarray(ramf_array, dtype=cp.float32)
    ramb = cp.asarray(ramb_array, dtype=cp.float32)
    
    violated_constr_mat = cp.full((max_runs, max_flips), cp.nan, dtype=cp.float32)

    n_iters = 0

    for it in range(max_flips - 1):
        n_iters += 1

        # global
        lit_inputs = cp.zeros((max_runs,literals)).astype(cp.float32)
        lit_inputs[:,2*cp.arange(0,variables,1)]=var_inputs
        lit_inputs[:,2*cp.arange(0,variables,1)+1]=cp.abs(var_inputs-1)
        f_val = lit_inputs @ ramf
        s_val = cp.zeros((max_runs,clauses))
        z_val = cp.zeros((max_runs,clauses))
        s_val[cp.where(f_val==0)] = 1
        z_val[cp.where(f_val==1)] = 1
        a = s_val @ ramb
        b = z_val @ ramb
        lit_one_indices = cp.where(lit_inputs==1)
        lit_zero_indices = cp.where(lit_inputs==0)
        neg_lit_indices = 2*cp.arange(0,variables,1)+1
        mv_arr = cp.reshape(a[lit_zero_indices[0],lit_zero_indices[1]],(max_runs,variables))
        bv_arr = cp.reshape(b[lit_one_indices[0],lit_one_indices[1]],(max_runs,variables))
        g_arr = mv_arr - bv_arr
        y = bv_arr
        violated_constr = cp.sum(mv_arr > 0, axis=1)
        violated_constr_mat[:, it] = violated_constr

        # early stopping
        if cp.sum(violated_constr_mat[:, it]) == 0:
            break 

        tmp1 = mv_arr<1
        tmp2 = bv_arr==0
        tmp3 = (~tmp1&tmp2).any(1)
        tmp4 = cp.repeat(cp.reshape(tmp3,(max_runs,1)),variables,axis=1)
        tmp5 = (tmp4 & (tmp1 | ~tmp2)) | (~tmp4 & tmp1)

        y[tmp5] = 1000

        if noise_dist == 'bernoulli':
            all_variable_indices = cp.reshape(cp.arange(0,variables,1),(1,variables))
            all_variable_indices = cp.repeat(all_variable_indices,max_runs,axis=0)
            all_variable_indices[tmp5] = -1
            all_variable_indices_sorted = cp.sort(-all_variable_indices,axis=1)
            all_variable_indices_sorted = -all_variable_indices_sorted
    
            tmp6 = cp.ones((max_runs,variables))
            tmp6[tmp5] = 0
            violated_constr_temp = cp.sum(tmp6,axis=1)
            violated_constr_temp[violated_constr_temp == 0] = 1
            num_candidate_variables = cp.array(np.random.randint(0,cp.asnumpy(violated_constr_temp)))
            xy = cp.arange(0,max_runs,1)
            update2 = all_variable_indices_sorted[xy,num_candidate_variables]
            
            y, violated_constr = map(
                    lambda x: x[cp.newaxis, :],
                    [y, violated_constr],
                )
            update = cp.argmin(y, axis=2)
            update[cp.where(violated_constr == 0)] = -1
            update1 = update[0]
            
            ind1 = cp.random.uniform(0,1,max_runs)>noise
            ind2 = update1==-1
            ind3 = ind1 & ~ind2 & ~tmp3
            update1[ind3] = update2[ind3]
        else:  
            if noise_dist == 'normal':
                # add gaussian noise to the make values
                y += noise * cp.random.randn(*y.shape, dtype=y.dtype)
            elif noise_dist == 'uniform':
                # add uniform noise. Note that the standard deviation is modulated by sqrt(3)
                y += cp.random.uniform(low=-noise*np.sqrt(3), high=noise*np.sqrt(3), size=y.shape, dtype=y.dtype) 
            elif noise_dist == 'intrinsic':
                # add noise considering automated annealing. Noise comes from memristor devices
                y += noise * cp.sqrt(y) * cp.random.randn(*y.shape, dtype=y.dtype)
            else:
                raise ValueError(f"Unknown noise distribution: {noise_dist}")

            y, violated_constr = map(
                    lambda x: x[cp.newaxis, :],
                    [y, violated_constr],
                )
            update = cp.argmin(y, axis=2)
            update[cp.where(violated_constr == 0)] = -1
            update1 = update[0]       
        
        if n_cores == 1:
            # only one core, no need to do random picks
            #update = update[0]
            update = update1
        else:
            # reduction -> randomly selecting one update
            update = update.T
            random_indices = cp.random.randint(0, update.shape[1], size=update.shape[0])
            update = update[cp.arange(update.shape[0]), random_indices]
        campie.flip_indices(var_inputs, update[:, cp.newaxis])
    
    iterations_timepoints = cp.arange(1, n_iters + 1) * params.get("Tclk", 6e-9)
    return violated_constr_mat, n_iters, var_inputs, iterations_timepoints[np.newaxis, :]



# ===============================================================
# Device utility: temperature/annealing policy (future: PT, ladders)
# ===============================================================
@cuda.jit(device=True)
def temperature_metaheuristic(initial_noise, final_noise, flip, max_flips):
    """Return sigma (noise/temperature) for this flip.
    Current policy: linear schedule from initial_noise to final_noise.
    Designed as a device function so we can later swap in parallel tempering or
    ladder-based schemes without touching the kernel body.
    """
    if max_flips > 1:
        progress = float32(flip) / float32(max_flips - 1)
    else:
        progress = float32(1.0)
    sigma = initial_noise - progress * (initial_noise - final_noise)
    if sigma < final_noise:
        sigma = final_noise
    return sigma


# ===============================================================
# Device kernel: runs the entire annealing schedule per run/thread
# ===============================================================
@cuda.jit
def _anneal_kernel(
    W, B, W_en, B_en, C_vec,    inputs,                 # (n_runs, n_vars) float32 in {0,1}
    metric,                 # (n_runs, max_flips) float32
    best_metric,            # (n_runs,) float32
    best_solution,          # (n_runs, n_vars) float32
    var_order,              # (max_flips, n_vars) int32
    initial_noise, final_noise, threshold,
    max_flips,
    rng_states
):
    r = cuda.grid(1)
    n_runs = inputs.shape[0]
    if r >= n_runs:
        return

    n_vars = inputs.shape[1]

    # Local best tracker for this run; use a large float32 sentinel
    best_e = float32(3.4028235e38)

    # Main annealing loop (GPU)
    for flip in range(max_flips):
        # Temperature policy (wrapped for future PT/ladders)
        sigma = temperature_metaheuristic(initial_noise, final_noise, flip, max_flips)

        # Update variables in the provided order for this flip
        for k in range(n_vars):
            i = var_order[flip, k]

            # Local field lf = sum_j W[i, j] * s_j + B[i]
            lf = float32(0.0)
            for j in range(n_vars):
                lf += W[i, j] * inputs[r, j]
            lf -= B[i]

            # Gaussian noise and thresholding
            if sigma > 0.0:
                eps = sigma * xoroshiro128p_normal_float32(rng_states, r)
            else:
                eps = float32(0.0)

            val = lf + eps
            inputs[r, i] = float32(1.0) if val >= threshold else float32(0.0)

        # Energy recomputation: E = -0.5 * s @ (W @ s) + B @ s
        acc = float32(0.0)  # will hold s^T (W s)
        for i in range(n_vars):
            si = inputs[r, i]
            if si != 0.0:
                inner = float32(0.0)
                for j in range(n_vars):
                    inner += inputs[r, j] * W[i, j]
                acc += si * inner

        lin = float32(0.0)  # will hold B^T s
        for i in range(n_vars):
            lin += inputs[r, i] * B[i]

        E = float32(-0.5) * acc + lin

        # Store metric for this flip directly (no helper kernel)
        metric[r, flip] = E

        # Track per-run best
        if E < best_e:
            best_e = E
            best_metric[r] = E
            for j in range(n_vars):
                best_solution[r, j] = inputs[r, j]


# =========================
# Host-side helpers (readable, faithful)
# =========================
def _as_numpy_f32(x):
    """Accept numpy or accidental CuPy (via .get()) and cast to float32 C-contiguous."""
    if hasattr(x, "get"):
        x = x.get()
    return np.asarray(x, dtype=np.float32, order="C")


def _max_gradient_like_yours(W_f32):
    """Your scaling: max over columns of (sum of positives, sum of abs(negatives))."""
    col_pos = (W_f32 * (W_f32 > 0)).sum(axis=0)
    col_neg = ((-W_f32) * (W_f32 < 0)).sum(axis=0)
    return np.maximum(col_pos, col_neg).max().astype(np.float32)


def _prep_energy_parts(W_f32):
    """Move diagonal into B_en, and zero the diagonal in W_en (float32)."""
    B_en = np.diag(W_f32).astype(np.float32)
    W_en = W_f32.copy()
    np.fill_diagonal(W_en, 0.0)
    return W_en, B_en


# =========================
# Public API (same signature/returns as original)
# =========================

def memHNN(architecture, config, params):
    """
    Numba-CUDA implementation with BOTH the per-flip annealing loop and the per-variable
    updates executed on the GPU. Notes:
      - local fields lf = W[i,:] @ s + B[i]
      - noise schedule and thresholding (as in memHNN paper)
      - energy with diag moved to linear term: W_en (zero diag), B_en = diag(W). This is MIMO specific

    Returns (matching original structure):
        metric_tracking_mat[:, :flip], flip, inputs, overall_best_solution, overall_best_metric
    """
    # Unpack architecture and coerce types
    W = _as_numpy_f32(architecture[0])  # (N, N)
    B = _as_numpy_f32(architecture[1])  # (N,)
    C_raw = architecture[2]

    # Params
    max_runs  = int(params.get("max_runs", 1000))
    max_flips = int(params.get("max_flips", 1000))

    # Config
    initial_noise = np.float32(config.get("noise", 0.8))
    final_noise   = np.float32(config.get("final_noise", 0.0))
    threshold     = np.float32(config.get("threshold", 0.0))
    update_mode   = config.get("update_mode", "sequential")  # "sequential" or "random"

    n_vars = int(W.shape[0])

    # Build per-run C vector (accept scalar, 0-d array, (1,), or (max_runs,))
    if np.isscalar(C_raw):
        C_vec_h = np.full((max_runs,), np.float32(C_raw), dtype=np.float32)
    else:
        C_arr = np.asarray(C_raw, dtype=np.float32)
        if C_arr.ndim == 0 or (C_arr.ndim == 1 and C_arr.size == 1):
            C_vec_h = np.full((max_runs,), np.float32(C_arr.reshape(())), dtype=np.float32)
        elif C_arr.ndim == 1 and C_arr.size == max_runs:
            C_vec_h = C_arr.astype(np.float32, copy=False)
        else:
            # Fallback: broadcast first element
            C_vec_h = np.full((max_runs,), np.float32(C_arr.ravel()[0]), dtype=np.float32)

    # Noise scaled by your exact max-column-gradient rule
    max_grad = _max_gradient_like_yours(W)
    initial_noise = np.float32(initial_noise * max_grad)

    # Initial states in {0,1}
    inputs_h = np.random.randint(2, size=(max_runs, n_vars)).astype(np.float32)

    # Metric tracking (prefilled with NaNs; kernel overwrites real columns)
    metric_h = np.full((max_runs, max_flips), np.float32(np.nan), dtype=np.float32)

    # Best trackers
    best_metric_h   = np.full(max_runs, np.float32(np.inf), dtype=np.float32)
    best_solution_h = inputs_h.copy()

    # Energy parts
    W_en_h, B_en_h = _prep_energy_parts(W)

    # Variable order matrix (one row per flip)
    if update_mode == "sequential":
        base = np.arange(n_vars, dtype=np.int32)
        var_order_h = np.tile(base, (max_flips, 1))
    elif update_mode == "random":
        var_order_h = np.empty((max_flips, n_vars), dtype=np.int32)
        for f in range(max_flips):
            var_order_h[f, :] = np.random.permutation(n_vars).astype(np.int32)
    else:
        base = np.arange(n_vars, dtype=np.int32)
        var_order_h = np.tile(base, (max_flips, 1))

    # ---- Move to device
    W_d       = cuda.to_device(W)
    B_d       = cuda.to_device(B)
    W_en_d    = cuda.to_device(W_en_h)
    B_en_d    = cuda.to_device(B_en_h)
    inputs_d  = cuda.to_device(inputs_h)
    metric_d  = cuda.to_device(metric_h)
    best_metric_d   = cuda.to_device(best_metric_h)
    best_solution_d = cuda.to_device(best_solution_h)
    var_order_d     = cuda.to_device(var_order_h)
    C_vec_d         = cuda.to_device(C_vec_h)

    # RNG states: one per run
    seed = np.random.randint(0, 2**31 - 1, dtype=np.int64)
    rng_states = create_xoroshiro128p_states(max_runs, seed=seed)

    # Launch config: ensure reasonable occupancy even for small max_runs
    TPB = 256
    blocks = (max_runs + TPB - 1) // TPB
    # Low-occupancy warnings are harmless; for small runs we still launch >= 1 block
    if blocks == 0:
        blocks = 1

    # ---- Single kernel: includes all loops and writes metric directly
    _anneal_kernel[blocks, TPB](
        W_d, B_d, W_en_d, B_en_d, C_vec_d,
        inputs_d, metric_d, best_metric_d, best_solution_d,
        var_order_d,
        np.float32(initial_noise), np.float32(final_noise), np.float32(threshold),
        np.int32(max_flips),
        rng_states,
    )

    # ---- Copy back
    inputs_h        = inputs_d.copy_to_host()
    metric_h        = metric_d.copy_to_host()
    best_metric_h   = best_metric_d.copy_to_host()
    best_solution_h = best_solution_d.copy_to_host()

    # Overall best (host reduction)
    overall_idx = int(np.argmin(best_metric_h))
    overall_best_metric  = np.float32(best_metric_h[overall_idx])
    overall_best_solution = best_solution_h[overall_idx]

    # Preserve original slicing convention
    flip = max_flips - 1
    return (
        metric_h[:, :flip],  # intentionally excludes the last column for fidelity
        flip,
        inputs_h,
        None, # No timing info in this version
        overall_best_solution,
        overall_best_metric,
    )


def pbits_ising(architecture, config, params):

    # Unpack architecture and coerce types
    J = _as_numpy_f32(architecture[0])  # (N, N)
    h = _as_numpy_f32(architecture[1])  # (N,)

    N = J.shape[0]
    neighbors = [[] for _ in range(N)]

    # Find non-zero elements in the upper triangle to define edges
    rows, cols = np.where(np.triu(J, k=1) != 0)

    for i, j in zip(rows, cols):
        neighbors[i].append(j)
        neighbors[j].append(i)

    # Sort each neighbor list for deterministic output
    for i in range(N):
        neighbors[i].sort()

    # Params
    max_runs  = int(params.get("max_runs", 1000))
    max_flips = int(params.get("max_flips", 1000))
    activation_fn = params.get("activation_fn", "linear")  # "tanh" or "step"
    num_threads = int(params.get("num_threads", 1))
    run_per_thread = int(max_runs / num_threads)

    # Config
    initial_noise = np.float32(config.get("noise", 0.8))
    final_noise   = np.float32(config.get("final_noise", 0.0))

    # Define paths
    # Assuming the script is run from the root of the CountryCrab project
    if os.getcwd().endswith("CountryCrab"):
        module_path = os.getcwd()
    else:
        module_path = os.path.abspath(os.path.join(".."))
    binary_dir = os.path.join(module_path,"submodules/ising-machine-cpu")
    binary_path = os.path.join(binary_dir, "target", "release", "ising_sa")
    
    # Create paths for w, h, and config files inside the binary directory. Use pid to avoid conflicts.
    j_path = os.path.join(binary_dir, f"J_{os.getpid()}.csv")
    h_path = os.path.join(binary_dir, f"h_{os.getpid()}.csv")
    neighbors_path = os.path.join(binary_dir, f"neighbors_{os.getpid()}.csv")
    config_path = os.path.join(binary_dir, f"pbits_config_{os.getpid()}.toml")

    # Remove the diagonal 0s from J
    J_no_diag = J[~np.eye(N, dtype=bool)].reshape(N, N - 1)

    np.savetxt(j_path, J_no_diag, delimiter=",")
    np.savetxt(h_path, h, delimiter=",")

    max_deg = max((len(row) for row in neighbors), default=0)
    with open(neighbors_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in neighbors:
            padded = row + [-1] * (max_deg - len(row))
            writer.writerow(padded)

    pbits_config = {
        'steps_per_temp': 1,
        'num_temps': max_flips,
        'initial_temp': float(initial_noise),
        'final_temp': float(final_noise),
        'solutions_per_thread': run_per_thread,
        'prob_method': activation_fn,
        'num_threads': num_threads,
        'J_values_file': j_path,  # Relative to binary_dir
        'h_values_file': h_path,  # Relative to binary_dir
        'neighbors_file': neighbors_path,  # Relative to binary_dir
        'generate_solutions_csv': True,
        'uid':str(os.getpid())
    }

    with open(config_path, 'w') as f:
        toml.dump(pbits_config, f)

    # Run the binary
    command = [
        os.path.abspath(binary_path),
        os.path.basename(config_path)
    ]
    
    result = subprocess.run(
        command,
        cwd=binary_dir,
        capture_output=True,
        text=True,
        check=True
    )

    # Read the output file
    metric = np.loadtxt(os.path.join(binary_dir, f"all_energy_{os.getpid()}.csv"), delimiter=",")
    overall_best_solution = np.loadtxt(os.path.join(binary_dir, f"solutions_{os.getpid()}.csv"), delimiter=",")

    # delete the temporary files
    os.remove(j_path)
    os.remove(h_path)
    os.remove(neighbors_path)
    os.remove(config_path)

    # NOTE: The rest of this function seems incomplete. 
    # Returning placeholder values.
    return metric, None, None, None, None, None