import subprocess
from subprocess import PIPE
import platform
from train_background_exec import create_shell_strings



def run_training():
    shell_string_cd,shell_string_train = create_shell_strings()
    if platform.system() == 'Windows':
        shell_string = shell_string_cd + " && " + shell_string_train
    else:
        shell_string = shell_string_cd + " ; " + shell_string_train

    print("Running ",shell_string)
    p2 = subprocess.Popen(shell_string,shell=True,stdout=PIPE, stderr=PIPE)
    for line in p2.stdout:
        print(line.decode(),end='')


if __name__ == "__main__":
    run_training()