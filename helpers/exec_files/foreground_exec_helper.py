import subprocess
import json
import os
import platform
from subprocess import PIPE

def create_shell_strings(filename):
    parameters = json.load(open('../parameters.json', 'r'))
    project_directory_path = os.path.normpath(parameters["project_directory_path"])
    assets_handler_dir = os.path.join(project_directory_path,'classification-barebones','helpers')
    assets_handler_file_name = filename
    venv_path = os.path.normpath(parameters['venv_path'])

    if platform.system() == 'Windows':
        if assets_handler_dir[0] == "X":
            shell_string_cd = "X: && cd " + assets_handler_dir
        else:
            shell_string_cd = " cd " + assets_handler_dir
    else:
        shell_string_cd = "cd " + assets_handler_dir

    shell_string_assets = venv_path + " -u " + assets_handler_file_name
    if platform.system() == 'Windows':
        shell_string = shell_string_cd + " && " + shell_string_assets
    else:
        shell_string = shell_string_cd + " ; " + shell_string_assets

    shell_string = shell_string + " 2>&1 "
    return shell_string


def run_handler(filename):
    shell_string = create_shell_strings(filename)
    print("Running ", shell_string)
    p2 = subprocess.Popen(shell_string, shell=True, stdout=PIPE, stderr=PIPE)
    for line in p2.stdout:
        print(line.decode(), end='')

