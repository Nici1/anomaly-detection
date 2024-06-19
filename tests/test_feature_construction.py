import argparse
import importlib
import inspect
import pytest
import os
import sys
import inspect
import typing
import numpy as np



@pytest.fixture(scope = "session")
def set_script(request):


    script_path = os.path.realpath(__file__)

    # Get the directory containing the script
    script_dir = os.path.dirname(script_path)

    # Go up one level to get the parent directory
    parent_dir = os.path.dirname(script_dir)

    algorithms_dir = os.path.join(parent_dir, "src", "algorithms", 'kmeans.py')

    abs_file_path = os.path.abspath(algorithms_dir)
    if os.path.exists(abs_file_path):
        print(f"File '{abs_file_path}' exists at the specified location.")
    else:
        print(f"File '{abs_file_path}' does not exist at the specified location.")

    module_name, _ = os.path.splitext(os.path.basename(abs_file_path))

    # Dynamically importing the module
    spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, obj in module.__dict__.items():
        
        # We want to check if the class in our package has the message_insert method and not its parents' classes
        
        if hasattr(obj, '__module__') and obj.__module__.split('.')[0] == module.__name__ :
                      
            return obj
        

def construct_instance(obj):
     
    # Constructs an instance of the class
    conf =  {"filtering": "None", "train_data": "data/ads-1.csv", "input_vector_size": 1, "n_clusters": 4, "treshold": 4, "output": [], "output_conf": [{}]}

    instance = obj(conf = conf)
    return instance




def test_average_construct(set_script):
    obj = set_script


    instance = construct_instance(obj)
    instance.averages = [[1, 2, 3]]
    instance.memory = [[[-0.4], 0.3100000000000001], [[-0.6], 0.4100000000000001], [[-0.8], 0.5100000000000001]]
    average_values = instance.average_construction()

    correctness = []
    buffer = []
    for idx, i in enumerate(instance.memory[::-1]):
        buffer.append(i[0][0])
        for j in instance.averages[0]:
            if idx+1 == j:
                result = (np.round(np.mean(buffer),decimals =3) == np.round(average_values[idx],decimals = 3))
                correctness.append(result)


    assert all(correctness), "Something is wrong with the average construction"




def test_periodic_averages(set_script):
    obj = set_script


    instance = construct_instance(obj)
    instance.periodic_averages = [[[2,[1, 2, 3]]]]
    instance.memory = [[[-0.6], 251.329], [[-0.2], 251.339], [[-0.1], 251.349], [[-1], 251.359], [[-1.2], 251.369], [[-1.4], 251.379], [[-0.8], 251.389]]
    instance.memory_size = len(instance.memory)
    average_values = instance.periodic_average_construction()

    #print(instance.periodic_averages[0][0])

    correctness = [True]
    buffer = []

    for idx, i in enumerate(instance.memory[::-1][::instance.periodic_averages[0][0][0]]):
        buffer.append(i[0][0])
        for j in instance.periodic_averages[0][0][1]:
            if idx+1 == j:
                result = (np.round(np.mean(buffer),decimals =3) == np.round(average_values[idx],decimals = 3))
                correctness.append(result)

    #print(average_values)
   
    
    assert all(correctness), "Something is wrong with the periodic average construction"



def test_shifts(set_script):
    obj = set_script


    instance = construct_instance(obj)
    instance.shifts = [[1,2,3]]
    instance.memory = [[[-0.6], 251.329], [[-0.2], 251.339], [[-0.1], 251.349], [[-1], 251.359], [[-1.2], 251.369], [[-1.4], 251.379], [[-0.8], 251.389]]
    values = instance.shift_construction()

    correctness = []
    buffer = []
    for idx, i in enumerate(instance.memory[::-1]):
        for j in instance.shifts[0]:
            if idx+1 == j:
                buffer.append(i[0][0])
    result = (buffer == values)
    correctness.append(result)

    assert all(correctness), "Something is wrong with the shift construction"