'''

import ast
import argparse



def pytest_addoption(parser):
    parser.addoption("--name", action="store", default="default name")
    parser.addoption("--arguments", action="store", default="default argument")



def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.name
    if 'name' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("name", [option_value])

    option_arguments = metafunc.config.option.arguments
    if 'arguments' in metafunc.fixturenames and option_arguments is not None:
        try:
            # Parse the arguments string into a dictionary
            arguments_dict = ast.literal_eval(option_arguments)
            # Parametrize the 'arguments' fixture with the parsed dictionary
            metafunc.parametrize("arguments", [arguments_dict])
        except (SyntaxError, ValueError):
            raise ValueError("Invalid dictionary format provided via command line")

'''
   



    

