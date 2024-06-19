import argparse
import importlib
import inspect
import pytest
import os
import sys
import inspect
import typing




@pytest.fixture(params = [('kmeans.py',
                           { "filtering": "None", "train_data": "data/ads-1.csv", "input_vector_size": 1, "n_clusters": 4, "treshold": 4, "output": [], "output_conf": [{}]}), 
                           ('isolation_forest.py',
                            {"filtering": "None", "train_data": "data/ads-1.csv", "input_vector_size": 1,  "max_samples": 100, "contamination":0.05, "max_features":1, "model_name": "sensor-cleaning-data1-test", "output": [], "output_conf": [{}]})], 
                            scope = "session")
def set_script(request):


    script_path = os.path.realpath(__file__)

    # Get the directory containing the script
    script_dir = os.path.dirname(script_path)

    # Go up one level to get the parent directory
    parent_dir = os.path.dirname(script_dir)

    algorithms_dir = os.path.join(parent_dir, "src", "algorithms", request.param[0])

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
                      
            return [obj, request.param[1]]



def method_presence(obj, method):
    
    found_method = False

    if hasattr(obj, method) and callable(getattr(obj, method)):
        print(f"The {method} method is present in the class.")
        found_method = True
        
    #assert found_method, "The 'message_insert' method is not present in any class in the script."
    return found_method





def test_message_insert_params(set_script):

    obj = set_script[0]
    signature = inspect.signature(obj.message_insert)
    parameter = list(signature.parameters.values())[1]

    # Get the type of the parameter
    parameter_type = parameter.annotation
    assert typing.get_origin(parameter_type) == dict, "The type of the parameter must be a dictionary"




def test_configure_params(set_script):

    obj = set_script[0]
    results = []
    signature = inspect.signature(obj.configure)
    parameters = list(signature.parameters.values())[1:]

    # Get the type of the parameter

    results.append(typing.get_origin(parameters[0].annotation) == dict)
    results.append(parameters[1].annotation == str)
    results.append(parameters[2].annotation == int)

    assert all(results), f"The parameter types must be typing.Dict, str, int"




def test_method_presence(set_script):

    obj = set_script[0]
    methods = ['__init__', 'configure', 'message_insert']
    results = []

    for m in methods:
        results.append(method_presence(obj, m))
    
    assert all(results)



def construct_instance(obj, arguments):
     
    # Constructs an instance of the class
 
    #arguments = { "filtering": "None", "train_data": "../data/ads-1.csv", "input_vector_size": 1, "n_clusters": 4, "treshold": 4, "output": [], "output_conf": [{}]}
    instance = obj(conf = arguments)
    return instance
           


def test_message_insert_return(set_script):

    # Checking if the return of the message_insert function is an integer
    obj = set_script[0]

    instance = construct_instance(obj, set_script[1])

    result = instance.message_insert({'timestamp': 0.3100000000000001, 'ftr_vector': [-0.103370997772]})
    assert isinstance(result, int), 'The return of message_insert must be an int'




def test_message_insert_return(set_script):

    # Checking if the return of the message_insert function is an integer
    obj = set_script[0]

    instance = construct_instance(obj, set_script[1])

    result = instance.message_insert({'timestamp': 0.3100000000000001, 'ftr_vector': [-0.103370997772]})
    assert isinstance(result, int), 'The return of message_insert must be an int'