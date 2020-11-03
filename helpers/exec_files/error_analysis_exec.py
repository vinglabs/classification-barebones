import subprocess
import json
import os
import platform
from subprocess import PIPE
from foreground_exec_helper import run_handler


def run_error_analysis_handler(filename):
    run_handler(filename)


if __name__ == "__main__":
    run_error_analysis_handler('error_analysis_handler.py')