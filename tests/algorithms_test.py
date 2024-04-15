import argparse
import importlib
import inspect
import pytest
import os


@pytest.fixture
def script_existence_check(name):
    
    abs_file_path = os.path.abspath(name)
    if os.path.exists(abs_file_path):
        print(f"File '{abs_file_path}' exists at the specified location.")
    else:
        print(f"File '{abs_file_path}' does not exist at the specified location.")

    module_name, _ = os.path.splitext(os.path.basename(abs_file_path))

    # Dynamically importing the module
    spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.fixture
def arguments_existence_check(arguments):
    return arguments
    


def test_message_insert_presence(script_existence_check):


    module = script_existence_check

    # Checking for the presence of the method inside any class in the module
    found_method = False
    for name, obj in module.__dict__.items():
    
        
        # We want to check if the class in our package has the message_insert method and not its parents' classes
        
        if hasattr(obj, '__module__') and obj.__module__.split('.')[0] == module.__name__:

            if hasattr(obj, 'message_insert') and callable(getattr(obj, 'message_insert')):

                print(f"The 'message_insert' method is present in the class '{name}'.")
                found_method = True
           
    if not found_method:
        print("The 'message_insert' method is not present in any class in the script.")




def test_message_insert_return(arguments_existence_check,script_existence_check):

    # Checking if the return of the message_insert function is an integer

    arguments = arguments_existence_check
    module = script_existence_check

    
    for name, obj in module.__dict__.items():

        if hasattr(obj, '__module__') and obj.__module__.split('.')[0] == module.__name__:

            instance = obj()
            instance.configure(conf = arguments)

            result = instance.message_insert({'timestamp': 0.3100000000000001, 'ftr_vector': [-0.103370997772]})

            if isinstance(result, int):
                assert True


