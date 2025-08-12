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

import os
import re

def natural_keys(text):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('(\d+)', text)]

def split_hpo_test(instances_path_list, hpo_test = 0.2, max_instances = 100,extension = '.cnf'):
    # this function load the isntances in the instances_path
    # split them in two dictionaries, one for hyperparameters optimization and one for testing

    instances_hpo_list = []
    instances_test_list = []
    for instances_path in instances_path_list:
        if not os.path.exists(instances_path):
            raise FileNotFoundError(f"{instances_path} does not exist.")
        
        # Load instances
        all_files = sorted(os.listdir(instances_path))
        
        
        # Filter out files with a the given extension extension 
        cnf_files = [file for file in all_files if file.endswith(extension)]

        # sort them alphabetically
        cnf_files.sort(key=natural_keys)
        
        # limit to max_instances
        cnf_files = cnf_files[:max_instances]

        # Use hpo_test of them for hpo
        number_of_hpo_instances = int(len(cnf_files) * hpo_test)
        
        # Get the first 20% of the files
        instances_hpo = cnf_files[:number_of_hpo_instances]
        instances_hpo = [instances_path + filename for filename in instances_hpo]
        instances_test = cnf_files[number_of_hpo_instances:]
        instances_test = [instances_path + filename for filename in instances_test]

        instances_hpo_list.append(instances_hpo)
        instances_test_list.append(instances_test)
    
    instances_hpo_list = [item for sublist in instances_hpo_list for item in sublist]
    instances_test_list = [item for sublist in instances_test_list for item in sublist]

    return instances_hpo_list, instances_test_list