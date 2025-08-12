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
import math

import campie
import cupy as cp
import warnings


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


def memHNN(architecture, config, params):
    """
    Memristive Hopfield Neural Network implementation for solving QUBO problems.
    Uses a thresholding mechanism on noisy local fields to update spins.
    Supports simulated annealing with multiple update modalities.
    
    Args:
        architecture: List containing [W, B, C] for QUBO mode or [W, B, C, tcam, ram] for k-SAT mode
        config: Configuration dict with noise, threshold, annealing parameters
        params: Execution parameters (max_runs, max_flips, etc.)
    
    Returns:
        violated_constr_mat: Matrix of violated constraints over iterations
        n_iters: Number of completed iterations
        inputs: Final state of variables
        iterations_timepoints: Timing information
    """    
    # Get operation mode
    mode = config.get("mode", "QUBO")
    
    # Get representation mode (binary or spin)
    representation = config.get("representation", "binary")  # Choose between "binary" (0,1) or "spin" (-1,1)
    
    # Get QUBO parameters from architecture - ensure all are CuPy arrays
    W = cp.asarray(architecture[0], dtype=cp.float32)  # Weight matrix
    B = cp.asarray(architecture[1], dtype=cp.float32)  # Linear terms
    C = float(architecture[2])  # Constant term - convert to float scalar for CuPy compatibility
    
    # Get TCAM/RAM matrices for k-SAT mode
    if mode == "k-SAT":
        tcam = cp.asarray(architecture[3], dtype=cp.float32)
        ram = cp.asarray(architecture[4], dtype=cp.float32)
        n_vars = tcam.shape[1]  # Number "true", not auxiliary variables
    
    # Get execution parameters
    max_runs = params.get("max_runs", 1000)
    max_flips = params.get("max_flips", 1000)
    n_cores = params.get("n_cores", 1)
    noise_dist = params.get("noise_distribution", 'normal')
    
    # Get configuration parameters
    initial_noise = config.get("noise", 0.8)  # Initial noise level
    # Initial noise should be multiplied by the maximum gradient value which is equal to the maximum sum of positive or absolute negative values of each column
    col_pos = cp.sum(cp.where(W > 0, W, 0), axis=0)
    col_neg = cp.sum(cp.where(W < 0, -W, 0), axis=0)
    max_gradient = cp.max(cp.maximum(col_pos, col_neg))
    initial_noise *= max_gradient
    final_noise = config.get("final_noise", 0.0)  # Final noise level
    threshold = config.get("threshold", 0.0)  # Decision threshold
    
    # Update mode settings
    update_mode = config.get("update_mode", "sequential")  # sequential, random, stochastic_group
    num_groups = config.get("num_groups", 10)  # Number of groups for stochastic group mode
    annealing_schedule = config.get("annealing_schedule", "linear")
    cooling_rate = config.get("cooling_rate", 0.95)  # For geometric schedule
    
    # Problem dimensions
    variables = W.shape[0]
    
    # Initialize inputs based on chosen representation
    if representation == "spin":
        # Initialize random spins (-1 or +1)
        inputs = 2 * cp.random.randint(2, size=(max_runs, variables)).astype(cp.float32) - 1
    else:  # binary representation
        # Initialize random binary values (0 or 1)
        inputs = cp.random.randint(2, size=(max_runs, variables)).astype(cp.float32)
    
    # Track violated constraints
    metric_tracking_mat = cp.full((max_runs, max_flips), cp.nan, dtype=cp.float32)

    # Track the current best solution
    best_solution = cp.copy(inputs)
    best_violated_constr = cp.full(max_runs, cp.inf, dtype=cp.float32)

    n_iters = 0
    current_iter = 0
    
    # Calculate initial energy based on representation
    if representation == "spin":
        energy = -0.5 * cp.sum(inputs * (inputs @ W), axis=1) - cp.sum(B * inputs, axis=1) - C
    else:  # binary representation
        energy = -0.5 * cp.sum(inputs * (inputs @ W), axis=1) - cp.sum(B * inputs, axis=1) - C
    
    # Calculate number of temperature steps based on update mode
    if update_mode == "sequential" or update_mode == "random":
        temp_steps = max(1, max_flips // variables)
    elif update_mode == "stochastic_group":
        temp_steps = max(1, max_flips // num_groups)
    
    # Main annealing loop
    for temp_step in range(temp_steps):
        if current_iter >= max_flips:
            break
        
        # Calculate current noise level based on annealing schedule
        progress = temp_step / (temp_steps - 1) if temp_steps > 1 else 1.0
        
        if annealing_schedule == "linear":
            current_noise = initial_noise - progress * (initial_noise - final_noise)
        elif annealing_schedule == "exponential":
            current_noise = initial_noise * cp.exp(-5.0 * progress)
        elif annealing_schedule == "geometric":
            current_noise = initial_noise * (cooling_rate ** temp_step)
        elif annealing_schedule == "logarithmic":
            current_noise = initial_noise / cp.log(temp_step + 2)
        else:
            current_noise = initial_noise  # Constant noise (no annealing)
            
        # Ensure noise doesn't drop below final_noise
        current_noise = max(current_noise, final_noise)
        
        # Apply update based on selected mode
        if update_mode == "sequential":
            # Update variables one by one in sequential order
            for var_idx in range(variables):
                
                # Efficiently compute local field for just this variable
                local_field = cp.zeros((max_runs,), dtype=cp.float32)
                for j in range(variables):
                    local_field += W[var_idx, j] * inputs[:, j]
                local_field += B[var_idx]
                
                # Add noise based on selected distribution
                if noise_dist == 'normal':
                    noisy_field = local_field + current_noise * cp.random.randn(*local_field.shape)
                elif noise_dist == 'uniform':
                    noisy_field = local_field + cp.random.uniform(-current_noise*cp.sqrt(3), 
                                                                 current_noise*cp.sqrt(3), 
                                                                 size=local_field.shape)
                elif noise_dist == 'intrinsic':
                    noisy_field = local_field + current_noise * cp.sqrt(cp.abs(local_field)) * cp.random.randn(*local_field.shape)
                else:
                    noisy_field = local_field
                
                # Compute new value based on threshold and representation
                if representation == "spin":
                    new_value = 2 * (noisy_field >= threshold).astype(cp.float32) - 1
                else:  # binary representation
                    new_value = (noisy_field >= threshold).astype(cp.float32)
                
                # Update variable
                inputs[:, var_idx] = new_value
                
                
        elif update_mode == "random":
            # Update variables one by one in random order
            var_indices = cp.random.permutation(variables)
            
            for var_idx in var_indices:
                
                # Efficiently compute local field for just this variable
                local_field = cp.zeros((max_runs,), dtype=cp.float32)
                for j in range(variables):
                    local_field += W[var_idx, j] * inputs[:, j]
                local_field += B[var_idx]
                
                # Add noise based on selected distribution
                if noise_dist == 'normal':
                    noisy_field = local_field + current_noise * cp.random.randn(*local_field.shape)
                elif noise_dist == 'uniform':
                    noisy_field = local_field + cp.random.uniform(-current_noise*cp.sqrt(3), 
                                                                 current_noise*cp.sqrt(3), 
                                                                 size=local_field.shape)
                elif noise_dist == 'intrinsic':
                    noisy_field = local_field + current_noise * cp.sqrt(cp.abs(local_field)) * cp.random.randn(*local_field.shape)
                else:
                    noisy_field = local_field
                
                # Compute new value based on threshold and representation
                if representation == "spin":
                    new_value = 2 * (noisy_field >= threshold).astype(cp.float32) - 1
                else:  # binary representation
                    new_value = (noisy_field >= threshold).astype(cp.float32)
                
                # Update variable
                inputs[:, var_idx] = new_value
                

        elif update_mode == "stochastic_group":
            # Randomly assign each variable to one of num_groups groups
            group_assignments = cp.random.randint(0, num_groups, size=variables)
            
            # Process each group
            for group_id in range(num_groups):
                # Get indices of variables assigned to this group
                group_indices = cp.where(group_assignments == group_id)[0]
                
                if len(group_indices) == 0:
                    continue  # Skip empty groups
                
                # Compute local fields for all variables in this group
                local_fields = cp.zeros((max_runs, len(group_indices)), dtype=cp.float32)
                
                # Efficient batch computation of local fields for the group
                for idx, var_idx in enumerate(group_indices):
                    local_fields[:, idx] = cp.sum(W[var_idx, :] * inputs, axis=1) + B[var_idx]

                # Add noise based on selected distribution
                if noise_dist == 'normal':
                    noisy_fields = local_fields + current_noise * cp.random.randn(*local_fields.shape)
                elif noise_dist == 'uniform':
                    noisy_fields = local_fields + cp.random.uniform(-current_noise*cp.sqrt(3), 
                                                                  current_noise*cp.sqrt(3), 
                                                                  size=local_fields.shape)
                elif noise_dist == 'intrinsic':
                    noisy_fields = local_fields + current_noise * cp.sqrt(cp.abs(local_fields)) * cp.random.randn(*local_fields.shape)
                else:
                    noisy_fields = local_fields
                
                # Compute new values based on threshold and representation
                if representation == "spin":
                    new_values = 2 * (noisy_fields >= threshold).astype(cp.float32) - 1
                else:  # binary representation
                    new_values = (noisy_fields >= threshold).astype(cp.float32)
                
                # Update values for all variables in this group
                for idx, var_idx in enumerate(group_indices):
                    inputs[:, var_idx] = new_values[:, idx]
                
        # Calculate violated constraints based on mode
        if mode == "k-SAT":
            # Use TCAM matching to compute violated constraints
            if representation == "spin":
                # Convert to binary for TCAM matching if using spin representation
                binary_inputs = (inputs + 1) / 2
                violated_clauses = campie.tcam_match(1-binary_inputs[:,0:n_vars], tcam)
            else:  # binary representation
                violated_clauses = campie.tcam_match(1-inputs[:,0:n_vars], tcam)
            
            make_values = violated_clauses @ ram
            violated_constr = cp.sum(make_values > 0, axis=1)
            metric_tracking_mat[:, current_iter] = violated_constr

            if cp.all(violated_constr == 0):
                break
        else:  # QUBO or energy mode
            # Calculate energy using QUBO formulation
            energy = -0.5 * cp.sum(inputs * (inputs @ W), axis=1) - cp.sum(B * inputs, axis=1) - C
            metric_tracking_mat[:, current_iter] = (energy).astype(cp.int32)
            # keep track of best solution

        current_iter += 1
        n_iters += 1
    
    # If returning from spin representation and we need binary outputs for consistency
    if representation == "spin" and config.get("return_binary", False):
        inputs = (inputs + 1) / 2
    
    # Calculate timing information
    iterations_timepoints = cp.arange(1, n_iters + 1) * params.get("Tclk", 6e-9)

    # get overall best solution (when violated constraints are minimal)
    overall_best_violated_constr = cp.min(best_violated_constr)
    overall_best_solution = best_solution[cp.where(best_violated_constr == overall_best_violated_constr)[0]]

    # Return results
    return metric_tracking_mat[:, :n_iters], n_iters, inputs, iterations_timepoints[cp.newaxis, :]