import sys
import os
import importlib.util

def locate_file(file_path):
    abs_file_path = os.path.abspath(file_path)
    if os.path.exists(abs_file_path):
        print(f"File '{abs_file_path}' exists at the specified location.")
    else:
        print(f"File '{abs_file_path}' does not exist at the specified location.")

    # Extracting the module name from the file path
    module_name, _ = os.path.splitext(os.path.basename(abs_file_path))
    
    # Dynamically importing the module
    spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Checking for the presence of the method inside any class in the module
    found_method = False
    for name, obj in module.__dict__.items():
        if hasattr(obj, 'message_insert') and callable(getattr(obj, 'message_insert')):
            print(f"The 'message_insert' method is present in the class '{name}'.")
            found_method = True
    
    if not found_method:
        print("The 'message_insert' method is not present in any class in the script.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    locate_file(file_path)
