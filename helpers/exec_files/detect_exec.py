import subprocess
import json
import os
from subprocess import PIPE
import platform


def create_shell_strings():
    parameters = json.load(open('../parameters.json', 'r'))
    source = os.path.normpath(parameters['detect']['source'])
    output = os.path.normpath(parameters['detect']['output'])
    img_height = parameters['train']['height']
    img_width = parameters['train']['width']
    model_type = parameters['model_type']
    padding_kind = parameters['train']['padding_kind']
    augment = parameters['detect']['augment']
    pretrained = parameters['train']['pretrained']
    normalization = parameters['train']['normalization']

    project_directory_path = os.path.normpath(parameters["project_directory_path"])
    assets_directory_path = os.path.join(project_directory_path,"assets")
    detect_dir = os.path.join(project_directory_path,"classification-barebones")
    venv_path = os.path.normpath(parameters['venv_path'])
    weights_file_name = os.path.normpath(parameters['detect']['weights_file_name'])
    weights_file_path = os.path.join(project_directory_path,"exports","weights",weights_file_name)
    classes_dir = os.path.join(assets_directory_path,"dataset","train")


    if platform.system() == 'Windows':
        if detect_dir[0] == "X":
            shell_string_cd = "X: && cd " + detect_dir
        else:
            shell_string_cd = " cd " + detect_dir
    else:
        shell_string_cd = " cd " + detect_dir
    #2 is std error,1 is stdout ;2>&1 tells to redirect all error to stdout
    shell_string_train = venv_path + \
                         ' -u ' + \
                         " detect.py  " \
                         " --source " + str(source) + \
                         " --output " + str(output) + \
                         " --height " + str(img_height) + \
                         " --width " + str(img_width) +\
                         " --weights " + str(weights_file_path) + \
                        " --model-type " + str(model_type) +\
                        " --classes-dir " + str(classes_dir)  +\
                        " --padding-kind " + str(padding_kind)

    if pretrained:
        shell_string_train += " --pretrained "

    if normalization:
        shell_string_train += " --normalization "
    if augment:
        shell_string_train += " --augment "

    shell_string_train += " 2>&1 "


    return shell_string_cd, shell_string_train


def run_detection():
    shell_string_cd, shell_string_detect = create_shell_strings()
    if platform.system() == 'Windows':
        shell_string = shell_string_cd + " && " + shell_string_detect
    else:
        shell_string = shell_string_cd + " ; " + shell_string_detect

    print("Running ", shell_string)
    p2 = subprocess.Popen(shell_string, shell=True, stdout=PIPE, stderr=PIPE)
    for line in p2.stdout:
        print(line.decode(), end='')


if __name__ == "__main__":
    run_detection()