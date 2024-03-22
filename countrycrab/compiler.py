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
    return clauses

def count_variables(list_of_lists):
    # Flatten the list of lists
    flattened_list = [abs(item) for sublist in list_of_lists for item in sublist]
    # Get largest integer in the list
    largest_integer = max(flattened_list)
    # Return the largest integer. This in the feature can be changed to the actual number of variables
    return largest_integer

def compile_walksat_m(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    
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


def compile_walksat_g(config: t.Dict, params: t.Dict) -> t.Union[t.Dict, t.Tuple]:
    instance_name = config["instance"]

    instance_name = config["instance"]
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
        net_filename = '/home/bhattact/CountryCrab/data/vpr_netlist/'+os.path.basename(instance_name).split(".cnf")[0]+'.net'
        superTile_varIndices, num_superTiles = read_netlist(net_filename,variables,clauses,num_wta_inputs)
        for i in range(num_superTiles):
            if len(superTile_varIndices[i])<num_wta_inputs:
                diff = num_wta_inputs - len(superTile_varIndices[i])
                superTile_varIndices[i].extend([superTile_varIndices[i][-1]]*diff)
                
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
