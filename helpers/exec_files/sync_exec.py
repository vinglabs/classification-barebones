import json
import os
import subprocess
from subprocess import PIPE

def create_shell_strings():
    parameters = json.load(open('../parameters.json', 'r'))
    project_directory_path = parameters["project_directory_path"]
    runs_directory = os.path.join(project_directory_path, "classification-barebones", "runs")
    destination_bucket = parameters['sync']['bucket_name']
    destination_key = parameters['sync']['destination_key']
    destination_url = "s3://{}/{}".format(destination_bucket, destination_key)
    sync_command = "aws s3 sync {} {}".format(runs_directory, destination_url)
    sleep_time = 30
    while_command = "while true;do {} ; sleep {};done".format(sync_command,sleep_time)

    return "nohup bash -c '{}' > sync.out 2>&1 &".format(while_command)

def run_sync():
    shell_string = create_shell_strings()
    print("Running ", shell_string)
    p2 = subprocess.Popen(shell_string, shell=True, stdout=PIPE, stderr=PIPE)
    for line in p2.stdout:
       print(line.decode(), end='')


if __name__ == "__main__":
    run_sync()


