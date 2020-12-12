import subprocess
import json
import os
from subprocess import PIPE
import platform


def create_shell_strings(background=False):
    parameters = json.load(open('../parameters.json', 'r'))

    # network param
    model_type = parameters['model_type']


    # train param
    batch_size = parameters['train']['batch_size']
    epochs = parameters['train']['epochs']
    resume = parameters['train']['resume']
    weights = parameters['train']['weights']
    device = parameters['train']['device']
    adam = parameters['train']['adam']
    lr = parameters['train']['lr']
    name = parameters['train']['name']
    image_height = parameters['train']['height']
    image_width = parameters['train']['width']
    padding_kind = parameters['train']['padding_kind']
    pretrained = parameters['train']['pretrained']
    decay = parameters['train']['decay']
    normalization = parameters['train']['normalization']
    subdataset = parameters['train']['subdataset']
    test_on_train = parameters['train']['test_on_train']




    project_directory_path = os.path.normpath(parameters["project_directory_path"])
    assets_directory_path = os.path.join(project_directory_path, "assets")
    train_data_dir = os.path.join(assets_directory_path,"dataset","train")
    valid_data_dir = os.path.join(assets_directory_path,"dataset","valid")
    train_dir = os.path.join(project_directory_path,"classification-barebones")
    venv_path = os.path.normpath(parameters['venv_path'])
    export_directory_path = os.path.join(project_directory_path,"exports")
    weights_dir = os.path.join(export_directory_path, "weights")



    if os.path.exists(export_directory_path):
        raise Exception("Export directory already exists.Please manually delete it to continue.")
    else:
        os.mkdir(export_directory_path)
        os.mkdir(weights_dir)


    if platform.system() == 'Windows':
        if train_dir[0] == "X":
            shell_string_cd = "X: && cd " + train_dir
        else:
            shell_string_cd = " cd " + train_dir
    else:
        shell_string_cd = " cd " + train_dir

    if background:
        shell_string_train = 'nohup '
    else:
        shell_string_train = ''

    shell_string_train += venv_path + \
                         ' -u ' + \
                         " train.py  " \
                         " --epochs " + str(epochs) + \
                         " --batch-size " + str(batch_size) + \
                         " --height " + str(image_height) + \
                         " --width " + str(image_width) +\
                         " --weights " + str(weights) + \
                         " --weights-dir " + str(weights_dir)  +\
                         " --lr " + str(lr) +\
                         " --model-type " + str(model_type) +\
                        " --valid-data-dir " + str(valid_data_dir) +\
                        " --train-data-dir " + str(train_data_dir) + \
                        " --padding-kind " + str(padding_kind) +\
                        " --decay " + str(decay)




    if resume == True:
        shell_string_train += " --resume "

    if adam:
        shell_string_train += " --adam "

    if device == 'cpu':
        shell_string_train += " --device cpu "

    if name != '':
        shell_string_train += " --name " + str(name) + " "

    if pretrained:
        shell_string_train += " --pretrained "

    if normalization:
        shell_string_train += "  --normalization "

    if test_on_train:
        shell_string_train += " --test-on-train "

    if subdataset:
        shell_string_train += " --subdataset "



    if background:
        shell_string_train += " >nohup.out 2>&1 &"
    else:
        shell_string_train +=  " 2>&1"



    return shell_string_cd, shell_string_train


def run_training():
    shell_string_cd, shell_string_train = create_shell_strings(background=True)
    if platform.system() == 'Windows':
        shell_string = shell_string_cd + " && " + shell_string_train
    else:
        shell_string = shell_string_cd + " ; " + shell_string_train

    print("Running ", shell_string)
    p2 = subprocess.Popen(shell_string, shell=True, stdout=PIPE, stderr=PIPE)
    for line in p2.stdout:
        print(line.decode(), end='')


if __name__ == "__main__":
    run_training()