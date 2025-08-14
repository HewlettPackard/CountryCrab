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
from pysat.solvers import Minisat22
from pysat.formula import CNF
import typing as t
import math
import itertools
import cupy as cp
import os

def load_clauses_from_cnf(file_path: str) -> t.List[t.List[int]]:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        clauses = []
        for line in lines:
            if line.startswith('c') or line.startswith('p'):
                continue
            elif line.startswith('%'):
                break
            clause = [int(x) for x in line.strip().split() if x != '0']
            clauses.append(clause)
    # clean empty clauses
    clauses = [l for l in clauses if l]
    return clauses

def count_variables(list_of_lists):
    # Flatten the list of lists
    flattened_list = [abs(item) for sublist in list_of_lists for item in sublist]
    # Get largest integer in the list
    largest_integer = max(flattened_list)
    # Return the largest integer. This in the feature can be changed to the actual number of variables
    return largest_integer

def compile_MNSAT(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    
    instance_name = config["instance"]
    clauses_list = load_clauses_from_cnf(instance_name)
    
    clauses = len(clauses_list)
    variables = count_variables(clauses_list)
    # map clauses to TCAM
    tcam_array = np.zeros([clauses, variables])    
    tcam_array[:] = np.nan
    for i in range(len(clauses_list)):
        tcam_array[i,abs(np.array(clauses_list[i]))-1]=clauses_list[i]
    tcam_array[tcam_array>0] = 1
    tcam_array[tcam_array<0] = 0
    # map clauses to RAM
    ram_array = tcam_array*1
    ram_array[ram_array==0]=1
    ram_array[np.isnan(ram_array)]=0


    tcam = cp.asarray(tcam_array, dtype=cp.float32)
    ram = cp.asarray(ram_array, dtype=cp.float32)
    # number of clauses that can map to each core
    n_words = params.get("n_words", clauses)
    n_cores = params.get("n_cores", 1)
    scheduling = params.get("scheduling", "fill_first")

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

    # rewrite the parameters
    params['n_cores'] = n_cores
    params['variables'] = variables
    params['clauses'] = clauses
    
    architecture = [tcam, ram, tcam_cores, ram_cores, n_cores]
    return architecture, params


def compile_GNSAT(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    instance_name = config["instance"]
    vpr_netlist_loc = config.get("netlist_dir","~/CountryCrab/data/vpr_netlist/")
    clauses_list = load_clauses_from_cnf(instance_name)

    clauses = len(clauses_list)
    variables = count_variables(clauses_list)

    num_wta_inputs = config.get("num_wta_inputs",variables)
    scheduling = params.get("scheduling", "fill_first")

    # map clauses to TCAM
    ramf_array = np.zeros([2*variables,clauses])
    ramb_array = np.zeros([clauses,2*variables])
    
    for i,clause in enumerate(clauses_list):
        for literal in clause:
            if literal>0:
                ramf_array[2*(literal-1),i]=1
                ramb_array[i,2*(literal-1)]=1
            else:
                ramf_array[2*(-literal-1)+1,i]=1
                ramb_array[i,2*(-literal-1)+1]=1    
    
    params['variables'] = variables
    params['clauses'] = clauses

    n_cores = params.get("n_cores", 1)

    if n_cores>1:
        if scheduling == "fill_first":
            var_list = np.arange(variables)
            num_superTiles = math.ceil(variables/num_wta_inputs)
            superTile_varIndices = []
            last_index = 0
            for i in range(num_superTiles):
                if variables%num_wta_inputs==0:
                    superTile_varIndices.append(list(var_list[last_index:last_index+num_wta_inputs]))
                    last_index = last_index+num_wta_inputs
                else:
                    if i<num_superTiles-1:
                        superTile_varIndices.append(list(var_list[last_index:last_index+num_wta_inputs]))
                        last_index = last_index+num_wta_inputs
                    else:
                        superTile_varIndices.append(list(var_list[last_index:]))
                        last_var = var_list[variables-1]
                        superTile_varIndices[i].extend([last_var]*(num_wta_inputs-int(variables%num_wta_inputs)))
        elif scheduling == "vpr":
            net_filename = vpr_netlist_loc+os.path.basename(instance_name).split(".cnf")[0]+'.net'
            superTile_varIndices, num_superTiles = read_netlist(net_filename,variables,clauses,num_wta_inputs)
            for i in range(num_superTiles):
                if len(superTile_varIndices[i])<num_wta_inputs:
                    diff = num_wta_inputs - len(superTile_varIndices[i])
                    superTile_varIndices[i].extend([superTile_varIndices[i][-1]]*diff)
                    
        architecture = [ramf_array, ramb_array, superTile_varIndices, num_superTiles]
    else:
        superTile_varIndices = list(np.arange(0,variables,1))
        num_superTiles = 1
        architecture = [ramf_array, ramb_array, superTile_varIndices, num_superTiles]
    return architecture, params

def read_netlist(net_filename,nvar,nclause,num_wta_inputs):
    variables = nvar
    file = open(net_filename, 'r')
    lines = file.readlines()
     
    count = 0
    vpr_tile_inputs = []
    vpr_tile_outputs = []
    for i,line in enumerate(lines):
        if 'clb[' in line:
            tile_inputs = lines[i+2]
            tile_inputs = [int(j) for j in tile_inputs.split('>')[1].split('<')[0].replace("open","").replace("_x","").split()]
            tile_outputs_temp = lines[i+5]
            tile_outputs_temp = tile_outputs_temp.split('>')[1].split('<')[0].replace("open","").replace(" ","").split('-&gt;clbouts1')
            num_outputs = sum([1 for _ in tile_outputs_temp])-1
            curr_index = i+10
            output_lines_read = 0
            tile_outputs = []
            while output_lines_read < num_outputs:
                if "mode=\"default\"" in lines[curr_index] and "instance=\"ble[" in lines[curr_index]:
                    tile_outputs.append(int(lines[curr_index].split()[1].split("\"")[1].split("_l")[1]))
                    output_lines_read = output_lines_read + 1
               
                curr_index = curr_index + 1
               
            vpr_tile_inputs.append(tile_inputs)
            vpr_tile_outputs.append(tile_outputs)
           
    
    num_clbs = len(vpr_tile_outputs)
    vpr_var_list = []
    for i in range(num_clbs):
        output_temp = np.array(vpr_tile_outputs[i])
        vpr_var_list.append(list(output_temp[np.where(output_temp<nvar+1)[0]]-1))
    
    counter = 0
    for i in range(num_clbs):
        counter = counter + len(vpr_var_list[i])
    
    if counter!=nvar:
        print("Variables Missing in Post-Packing Netlist!!!")
       
    num_superTiles = math.ceil(variables/num_wta_inputs)
    superTile_varIndices = []
    vpr_var_list_copy = vpr_var_list.copy()
    for i in range(num_superTiles):
        pointer = 0
        superTile_varIndices.append(vpr_var_list_copy[pointer])
        vpr_var_list_copy.remove(vpr_var_list_copy[pointer])
        curr_len = len(superTile_varIndices[i])
        if curr_len < num_wta_inputs:
            done_flag = 0
            pointer = 0
            while done_flag==0 and len(vpr_var_list_copy)!=0:
                if curr_len +len(vpr_var_list_copy[pointer]) <= num_wta_inputs:
                    superTile_varIndices[i].extend(vpr_var_list_copy[pointer])
                    vpr_var_list_copy.remove(vpr_var_list_copy[pointer])
                    curr_len = len(superTile_varIndices[i])
                    if len(vpr_var_list_copy)==0:
                        break
                    if pointer>=len(vpr_var_list_copy)-1 or len(vpr_var_list_copy)==0 or curr_len==num_wta_inputs:
                        done_flag = 1
                else:
                    pointer = pointer + 1
                    if pointer==len(vpr_var_list_copy) or curr_len==num_wta_inputs:
                        done_flag = 1
                   
                       
        if len(vpr_var_list_copy)==0:
            break
    
    return superTile_varIndices, num_superTiles


# QUBO mappings available

# Returns W, B, C of the QUBO energy from 3/4-SAT cnf (k = 3/4 SAT only so far)
# W is returned as a symmetric matrix, i.e. E = x^T*W*x/2 + Bx + C
    
# mapping type 1: "clause_wise" mapping introduces an auxiliary variable for each clause of the cnf (Total QUBO size N = N_var + N_clauses)
# mapping type 2: "shared" mapping introduces each auxiliary varibale for multiple clauses by solving an optimization 
#     problem of finding x_ix_j = y substitutions minimizing the total number of aux variables (Total QUBO size N < N_var + N_clauses)
#
# mapping name 1: "Rosenberg" penalty (observed to be better for parallel updates on uf 3-SAT)
# mapping name 2: "KZFD" penalty      (observed to be better for single updates on uf 3-SAT)

def qubo_sat_map(config: t.Dict) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    instance_name = config["instance"]
    clauses = load_clauses_from_cnf(instance_name)

    num_vars = np.max(abs(np.concatenate(clauses, axis=0)))

    clauses_1 = []
    clauses_2 = []
    clauses_34 = []
    use_4sat = False

    for c in clauses:
        if len(c) > 4:
            raise RuntimeError("max 4 sat only!")

        elif len(c) == 4:
            use_4sat = True
            clauses_34.append(c)

        elif len(c) == 3:
            clauses_34.append(c)

        elif len(c) == 2:
            clauses_2.append(c)

        elif len(c) == 1:
            clauses_1.append(c)

    mapping_type = config["mapping_type"]   #clause_wise or shared   
    if not (mapping_type in ["clause_wise", "shared"]):
        raise RuntimeError("wrong mapping type!")
    
    mapping_name = config["mapping_name"]   #Rosenberg or KZFD    
    if not (mapping_name in ["Rosenberg", "KZFD"]):
        raise RuntimeError("wrong mapping name!") 
    
    if use_4sat:
        W, B, C = qubo_4sat_map(num_vars, clauses_34, mapping_name, mapping_type)
    else:
        if mapping_type == "clause_wise":
            W, B, C = clause_wise_qubo_3sat_map(num_vars, clauses_34, mapping_name)
        else:
            W, B, C = shared_qubo_3sat_map(num_vars, clauses_34, mapping_name)
    
    for c in clauses_1:
        if c[0] < 0:
            B[-c[0]-1] += 1
        else:
            B[c[0]-1] += -1
            C += 1

    for c in clauses_2:
        if c[0] < 0 and c[1] < 0:
            W[-c[0]-1, -c[1]-1] += 1
            W[-c[1]-1, -c[0]-1] += 1

        elif c[0] > 0 and c[1] < 0:
            W[c[0]-1, -c[1]-1] += -1
            W[-c[1]-1, c[0]-1] += -1
            B[-c[1]-1] += 1

        elif c[0] < 0 and c[1] > 0:
            W[-c[0]-1, c[1]-1] += -1
            W[c[1]-1, -c[0]-1] += -1
            B[-c[0]-1] += 1

        else:
            W[c[0]-1, c[1]-1] += 1
            W[c[1]-1, c[0]-1] += 1

            B[c[0]-1] += -1
            B[c[1]-1] += -1

            C += 1
    
    return W, B, C


def shared_qubo_3sat_map(num_vars: int, 
                         clauses: t.List[t.List[int]], mapping_name) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    num_clauses = len(clauses)

    cl = -np.array(clauses)            #invert the clauses to map to energy
    cl_idx =  np.abs(cl) - np.full_like(cl, 1) #count the variables from 0

    W1 = np.zeros((num_vars, num_vars))
    B1 = np.zeros(num_vars)
    C = np.zeros(1)
    
    for i in range(num_clauses):
        if cl[i, 0] > 0: 
            if cl[i, 1] > 0: 
                if cl[i, 2] > 0:
                    pass
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] += 1

            else:
                if cl[i, 2] > 0:
                    W1[cl_idx[i,0], cl_idx[i,2]] += 1
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] -= 1
                    W1[cl_idx[i,0], cl_idx[i,2]] -= 1

                    B1[cl_idx[i,0]] += 1
            
        else:
            if cl[i,  1] > 0: 
                if cl[i, 2] > 0: 
                    W1[cl_idx[i,1], cl_idx[i,2]] += 1
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] -= 1
                    W1[cl_idx[i,1], cl_idx[i,2]] -= 1

                    B1[cl_idx[i,1]] += 1
                
            else:
                if cl[i,2] > 0: 
                    W1[cl_idx[i,0], cl_idx[i,2]] -= 1
                    W1[cl_idx[i,1], cl_idx[i,2]] -= 1

                    B1[cl_idx[i,2]] += 1
                else:
                    W1[cl_idx[i,0], cl_idx[i,1]] += 1
                    W1[cl_idx[i,0], cl_idx[i,2]] += 1
                    W1[cl_idx[i,1], cl_idx[i,2]] += 1

                    B1[cl_idx[i,0]] -= 1
                    B1[cl_idx[i,1]] -= 1
                    B1[cl_idx[i,2]] -= 1

                    C += 1 
    
    count3 = np.full(num_clauses, 1)

    #take into account repeating 3rd order terms
    for i in range(num_clauses):
        if count3[i] == 0:
            continue
        c1 = np.sort(cl_idx[i, :])

        first_match = True
        for j in range(i+1, num_clauses):
            if count3[j] == 0:
                continue
            c2 = np.sort(cl_idx[j, :])
            if (c1 == c2).all():
                if first_match:
                    if np.prod(cl[i, :]) > 0:
                        count3[i] = 1
                    else:
                        count3[i] = -1
                    first_match = False
                
                if np.prod(cl[j, :]) > 0:
                    count3[i] += 1
                else:
                    count3[i] -= 1
                
                count3[j] = 0

    pairs_matrix = np.zeros((num_vars, num_vars), dtype=int)
    for i in range(num_clauses):
        if count3[i] != 0:
            pairs_matrix[cl_idx[i, 0], cl_idx[i, 1]] += 1
            pairs_matrix[cl_idx[i, 0], cl_idx[i, 2]] += 1
            pairs_matrix[cl_idx[i, 1], cl_idx[i, 2]] += 1
    
    pairs_matrix += np.transpose(pairs_matrix)

    pair_count = []
    for i in range(num_vars):
        for j in range(i+1, num_vars):
            if pairs_matrix[i, j] > 0:
                pair_count.append([np.array([i, j]), pairs_matrix[i, j]])

    pair_count = sorted(pair_count, key = lambda p: p[1], reverse=True)

    shared_pairs = []
    track_clauses = np.full(num_clauses, 1)

    for i in range(num_clauses):
        if i == 45:
            pass
        if track_clauses[i] == 0 or count3[i] == 0:
            continue
        
        c01 = np.sort([cl_idx[i, 0], cl_idx[i, 1]])
        c02 = np.sort([cl_idx[i, 0], cl_idx[i, 2]])
        c12 = np.sort([cl_idx[i, 1], cl_idx[i, 2]])

        for k in range(len(pair_count)):
            save_pair = False

            if (c01 == pair_count[k][0]).all():
                save_pair = True
                for l in range(len(pair_count)):
                    if((c02 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                        pair_count[l][1] -= 1

            if (c02 == pair_count[k][0]).all():
                save_pair = True
                for l in range(len(pair_count)):
                    if((c01 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                        pair_count[l][1] -= 1
                        
            if (c12 == pair_count[k][0]).all():
                save_pair = True
                for l in range(len(pair_count)):
                    if((c01 == pair_count[l][0]).all() or (c02 == pair_count[l][0]).all()):
                        pair_count[l][1] -= 1

            if save_pair:
                pair_count[k][1] -= 1
                
                for j in range(i+1, num_clauses):
                    if count3[j] == 0 or track_clauses[j] == 0:
                        continue

                    c01 = np.sort([cl_idx[j, 0], cl_idx[j, 1]])
                    c02 = np.sort([cl_idx[j, 0], cl_idx[j, 2]])
                    c12 = np.sort([cl_idx[j, 1], cl_idx[j, 2]])

                    if (c01 == pair_count[k][0]).all():
                        for l in range(len(pair_count)):
                            if((c02 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                                pair_count[l][1] -= 1
                        pair_count[k][1] -= 1
                        track_clauses[j] = 0

                    if (c02 == pair_count[k][0]).all():
                        for l in range(len(pair_count)):
                            if((c01 == pair_count[l][0]).all() or (c12 == pair_count[l][0]).all()):
                                pair_count[l][1] -= 1
                        pair_count[k][1] -= 1
                        track_clauses[j] = 0

                    if (c12 == pair_count[k][0]).all():
                        for l in range(len(pair_count)):
                            if((c01 == pair_count[l][0]).all() or (c02 == pair_count[l][0]).all()):
                                pair_count[l][1] -= 1
                        pair_count[k][1] -= 1
                        track_clauses[j] = 0

                shared_pairs.append(pair_count[k][0])
                if pair_count[k][1] != 0:
                    raise RuntimeError("Error")
                
                pair_count = sorted(pair_count, key = lambda p: p[1], reverse=True)

                break
    
    N = num_vars + len(shared_pairs)

    W = np.zeros((N, N))
    B = np.zeros(N)

    W[:num_vars, :num_vars] = W1
    B[:num_vars] = B1

    if mapping_name == "Rosenberg":
        penalty_lb = np.zeros((len(shared_pairs), 3))

    check = 0
    for i in range(num_clauses):
        c01 = np.sort([cl_idx[i, 0], cl_idx[i, 1]])
        c02 = np.sort([cl_idx[i, 0], cl_idx[i, 2]])
        c12 = np.sort([cl_idx[i, 1], cl_idx[i, 2]])
    
        found_pair = False
        for k in range(len(shared_pairs)):
            if (c01 == shared_pairs[k]).all():
                found_pair = True
                if np.prod(cl[i, :]) > 0:
                    W[cl_idx[i, 2], num_vars + k] += 1
                else:
                    W[cl_idx[i, 2], num_vars + k] -= 1

            if (c02 == shared_pairs[k]).all():
                found_pair = True
                if np.prod(cl[i, :]) > 0:
                    W[cl_idx[i, 1], num_vars + k] += 1
                else:
                    W[cl_idx[i, 1], num_vars + k] -= 1

            if (c12 == shared_pairs[k]).all():
                found_pair = True
                if np.prod(cl[i, :]) > 0:
                    W[cl_idx[i, 0], num_vars + k] += 1
                else:
                    W[cl_idx[i, 0], num_vars + k] -= 1
    
            if found_pair:
                check += 1
                if mapping_name == "Rosenberg":
                    if np.prod(cl[i, :]) > 0:
                        penalty_lb[k, 0] += 1
                    else:
                        penalty_lb[k, 1] += 1
                    
                    if penalty_lb[k, 2] <  np.max(penalty_lb[k, :]):
                        W[shared_pairs[k][0], shared_pairs[k][1]] += 1
                        W[shared_pairs[k][0], num_vars + k] -= 2
                        W[shared_pairs[k][1], num_vars + k] -= 2
                        B[num_vars + k] += 3

                        penalty_lb[k, 2] += 1

                elif mapping_name == "KZFD":
                    W[shared_pairs[k][0], shared_pairs[k][1]] += 1
                    W[shared_pairs[k][0], num_vars + k] -= 1
                    W[shared_pairs[k][1], num_vars + k] -= 1
                    B[num_vars + k] += 1
                    
                    if np.prod(cl[i, :]) < 0:
                        B[num_vars + k] += 1
                        W[shared_pairs[k][0], shared_pairs[k][1]] -= 1

                else:
                    raise RuntimeError("Unknown mapping!")
                
                break
    # print(f"substituted {check} clauses")
    W += np.transpose(W)

    return W, B, C


def clause_wise_qubo_3sat_map(num_vars: int, 
                              clauses: t.List[t.List[int]], mapping_name) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_clauses = len(clauses)

    N = num_vars + num_clauses

    cl = -np.array(clauses)            #invert the clauses to map to energy
    cl_idx =  np.abs(cl) - np.full_like(cl, 1) #count the variables from 0

    W = np.zeros((N, N))
    B = np.zeros(N)
    C = np.zeros(1)

    for i in range(num_clauses):
      if cl[i, 2] > 0:
        W[num_vars + i, cl_idx[i, 2]] = 1
      else:
        B[num_vars + i] = 1
        W[num_vars + i, cl_idx[i, 2]] = -1

    for i in range(num_clauses):
      if mapping_name == "Rosenberg":
          B[num_vars + i] += 3
      else:
          B[num_vars + i] += 1

      if cl[i, 0] > 0 and cl[i, 1] > 0:
          W[cl_idx[i, 0], cl_idx[i, 1]] += 1

          if mapping_name == "Rosenberg":
              W[num_vars + i, cl_idx[i, 0]] -= 2
              W[num_vars + i, cl_idx[i, 1]] -= 2
          else:
              W[num_vars + i, cl_idx[i, 0]] -= 1
              W[num_vars + i, cl_idx[i, 1]] -= 1

      elif cl[i, 0] > 0 and cl[i, 1] < 0:
          W[cl_idx[i, 0], cl_idx[i, 1]] -= 1
          B[cl_idx[i, 0]] += 1

          if mapping_name == "Rosenberg":
              W[num_vars + i, cl_idx[i, 0]] -= 2
              W[num_vars + i, cl_idx[i, 1]] += 2
              B[num_vars + i] -= 2
          else:
              W[num_vars + i, cl_idx[i, 0]] -= 1
              W[num_vars + i, cl_idx[i, 1]] += 1
              B[num_vars + i] -= 1

      elif cl[i, 0] < 0 and cl[i, 1] > 0:
          W[cl_idx[i, 0], cl_idx[i, 1]] -= 1
          B[cl_idx[i, 1]] += 1

          if mapping_name == "Rosenberg":
              W[num_vars + i, cl_idx[i, 1]] -= 2
              W[num_vars + i, cl_idx[i, 0]] += 2
              B[num_vars + i] -= 2
          else:
              W[num_vars + i, cl_idx[i, 1]] -= 1
              W[num_vars + i, cl_idx[i, 0]] += 1
              B[num_vars + i] -= 1

      else:
          W[cl_idx[i, 0], cl_idx[i, 1]] += 1
          B[cl_idx[i, 0]] -= 1
          B[cl_idx[i, 1]] -= 1
          C += 1  # add the constant

          if mapping_name == "Rosenberg":
              W[num_vars + i, cl_idx[i, 0]] += 2
              W[num_vars + i, cl_idx[i, 1]] += 2
              B[num_vars + i] -= 4
          else:
              W[num_vars + i, cl_idx[i, 0]] += 1
              W[num_vars + i, cl_idx[i, 1]] += 1
              B[num_vars + i] -= 2

    W += W.transpose()   

    return W, B, C

# Reduces the 4sat problem to 3sat: xa*xb*xc*xd -> xa*xb*y + penalty and 
# performs quadratization according to the schemes above
def qubo_4sat_map(num_vars: int, 
                  clauses: t.List[t.List[int]], 
                  mapping_name, 
                  mapping_type) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:

    cl_3sat = []
    cl_4sat = []

    n_aux = num_vars
    for c in clauses:
        if len(c) == 4:
            n_aux += 1
            cl_4sat.append(-np.array(c))
            cl_3sat.append(np.array([c[0], c[1], -n_aux]))
        else:
            cl_3sat.append(-np.array(c))
    
    N = num_vars + len(cl_4sat)

    W0 = np.zeros((N, N))
    B0 = np.zeros(N)
    C0 = np.zeros(1)

    cl_4idx = np.abs(cl_4sat) - np.full_like(cl_4sat, 1) 

    for i in range(len(cl_4sat)):
      if mapping_name == "Rosenberg":
          B0[num_vars + i] += 3
      else:
          B0[num_vars + i] += 1

      if cl_4sat[i][2] > 0 and cl_4sat[i][3] > 0:
          W0[cl_4idx[i, 2], cl_4idx[i, 3]] += 1

          if mapping_name == "Rosenberg":
              W0[num_vars + i, cl_4idx[i, 2]] -= 2
              W0[num_vars + i, cl_4idx[i, 3]] -= 2
          else:
              W0[num_vars + i, cl_4idx[i, 2]] -= 1
              W0[num_vars + i, cl_4idx[i, 3]] -= 1

      elif cl_4sat[i][2] > 0 and cl_4sat[i][3] < 0:
          W0[cl_4idx[i, 2], cl_4idx[i, 3]] -= 1
          B0[cl_4idx[i, 2]] += 1

          if mapping_name == "Rosenberg":
              W0[num_vars + i, cl_4idx[i, 2]] -= 2
              W0[num_vars + i, cl_4idx[i, 3]] += 2
              B0[num_vars + i] -= 2
          else:
              W0[num_vars + i, cl_4idx[i, 2]] -= 1
              W0[num_vars + i, cl_4idx[i, 3]] += 1
              B0[num_vars + i] -= 1

      elif cl_4sat[i][2] < 0 and cl_4sat[i][3] > 0:
          W0[cl_4idx[i, 2], cl_4idx[i, 3]] -= 1
          B0[cl_4idx[i, 3]] += 1

          if mapping_name == "Rosenberg":
              W0[num_vars + i, cl_4idx[i, 3]] -= 2
              W0[num_vars + i, cl_4idx[i, 2]] += 2
              B0[num_vars + i] -= 2
          else:
              W0[num_vars + i, cl_4idx[i, 3]] -= 1
              W0[num_vars + i, cl_4idx[i, 2]] += 1
              B0[num_vars + i] -= 1

      else:
          W0[cl_4idx[i, 2], cl_4idx[i, 3]] += 1
          B0[cl_4idx[i, 2]] -= 1
          B0[cl_4idx[i, 3]] -= 1
          C0 += 1

          if mapping_name == "Rosenberg":
              W0[num_vars + i, cl_4idx[i, 2]] += 2
              W0[num_vars + i, cl_4idx[i, 3]] += 2
              B0[num_vars + i] -= 4
          else:
              W0[num_vars + i, cl_4idx[i, 2]] += 1
              W0[num_vars + i, cl_4idx[i, 3]] += 1
              B0[num_vars + i] -= 2

    W0 += W0.transpose()
    
    if mapping_type == "clause_wise":
        W, B, C = clause_wise_qubo_3sat_map(N, cl_3sat, mapping_name)
    else:
        W, B, C = shared_qubo_3sat_map(N, cl_3sat, mapping_name)

    W[:N, :N] += W0
    B[:N] += B0
    C += C0

    return W, B, C


def compile_memHNN(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    """
    Compiler function for memHNN heuristic.
    Extracts QUBO mapping from the CNF instance and also TCAM/RAM for k-SAT mode.
    
    Args:
        config: Configuration dictionary with instance path and mapping settings
        params: Execution parameters
        
    Returns:
        architecture: List containing [W, B, C] for QUBO/energy mode or 
                     [W, B, C, tcam, ram] for k-SAT mode
        params: Updated parameters
    """
    
    # Get operation mode
    mode = config.get("mode", "QUBO")
    # TODO, currently the way to get the violated constraints matrix and the mapping are both passed as "mode" but these are two different things
    if mode == 'k-SAT':
    # Get QUBO mapping parameters from config
        mapping_type = config.get("mapping_type", "clause_wise")
        mapping_name = config.get("mapping_name", "Rosenberg")
        
        # Update config with mapping parameters if not already present
        if "mapping_type" not in config:
            config["mapping_type"] = mapping_type
        if "mapping_name" not in config:
            config["mapping_name"] = mapping_name
        
        # Extract QUBO coefficients from the CNF file
        W, B, C = qubo_sat_map(config)
    

    
    if mode == "k-SAT":
        # Also compile TCAM/RAM matrices like in MNSAT
        mnsat_config = config.copy()
        mnsat_config["mode"] = "MNSAT"  # Ensure we get the right format
        mnsat_architecture, _ = compile_MNSAT(mnsat_config, params.copy())
        
        # Extract TCAM/RAM matrices
        tcam = mnsat_architecture[0]
        ram = mnsat_architecture[1]
        
        architecture = [-W, B, C, tcam, ram]
    elif mode == 'MIMO':
        # QUBO or energy mode
        W, transmitted_bits = qubo_mimo_map(config["instance"])
        params['transmitted_bits'] = transmitted_bits
        # this works not sure why
        B = np.zeros(W.shape[0])
        C = np.zeros(1)        
        architecture = [-W, B, C]

        # remove this if it doesn't work
        # B = np.diag(W)
        # np.fill_diagonal(W, 0)
        # architecture = [-W, B, C]
        
    else:
        raise ValueError("Unsupported mode. Choose either 'k-SAT' or 'MIMO'.")
    
    # Store problem dimensions in params
    variables = W.shape[0]
    params['variables'] = variables
    return architecture, params



def qubo_mimo_map(file_path):
    """
    Extract the coupling matrix W from a MIMO QUBO file.
    
    Parameters:
    file_path (str): Path to the QUBO file
    
    Returns:
    numpy.ndarray: The coupling matrix W
    int: Number of variables
    str: Transmitted bits
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Get the number of variables
    num_variables = int(lines[0].strip())
    
    # Get the transmitted bits
    transmitted_bits = lines[1].strip()
    
    # Initialize the coupling matrix with zeros
    W = np.zeros((num_variables, num_variables))
    
    # Process the QUBO terms (starting from line 3)
    for line in lines[2:]:
        # Split the line into tokens and convert to appropriate types
        tokens = line.strip().split()
        
        if len(tokens) == 2:
            # Diagonal term: i value
            i = int(tokens[0])
            value = float(tokens[1])
            W[i, i] = value
        elif len(tokens) == 3:
            # Off-diagonal term: i j value
            i = int(tokens[0])
            j = int(tokens[1])
            value = float(tokens[2])
            W[i, j] = value
            W[j, i] = value  # Ensure symmetry for QUBO matrix
    
    return W, transmitted_bits