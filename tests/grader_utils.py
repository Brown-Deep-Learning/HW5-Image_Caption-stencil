import os
import sys
import traceback

hw = "hw5"
default_paths = ['student', './student', f'student/{hw}', f'./student/{hw}']
code_path = ''

def file_exists(filename, get_path=False, other_paths=[code_path], base_paths=default_paths):
    paths = other_paths + base_paths
    files = [f'{p}/{filename}' for p in paths]
    found = False
    found_files = []
    for file in files:
        if os.path.exists(file):
            found = True
            found_files += [file]
    return (found, found_files) if get_path else found

def file_paths_correct(paths: list):
    for path in paths:
        if not os.path.exists(path):
            return False
    return True


def raise_file_paths_incorrect_error(get_error=False):
    sys.tracebacklimit = 0
    error_msg = (
        "\nFile paths may not be correct\n"
        "\t- Please check that your files are of the form hw5/code/...\n"
        "\t- Ex: hw5/code/rnn.py\n"
        "Imports may not be correct\n"
        "\t- Please check that you don't have any extra imports in your files\n"
        "\t- Ex: tkinter, tensorflow (also not allowed for this assignment)\n"
        "Calling functions and not wrapping them in if __name__=='__main__'"
    )
    if get_error: return AssertionError(error_msg)
    raise AssertionError(error_msg)


def raise_grad_only_error(e, is_grad, raise_anyways=False):
    if is_grad:
        raise RuntimeError(f"\nAdvanced model failed to construct: {e}")
    print("Student advanced model doesn't work; Not '2470student', so ok!")
    if raise_anyways:
        raise e
    print(e)
