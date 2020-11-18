import os
from helper_utils.file_utils import copy_file, zip_file
from helper_utils.s3_helpers import upload_to_s3
import json
import shutil


def read_parameter_json():
    parameters = json.load(open('parameters.json', 'r'))
    return parameters


def export():
    parameters = read_parameter_json()
    project_directory_path = os.path.normpath(parameters["project_directory_path"])
    destination_directory = parameters['export']["destination_directory"]
    export_bucket_name = parameters['export']['bucket_name']
    export_directory_path = os.path.join(project_directory_path, 'exports')


    print("Copying parameters.json to export folder...")
    # copy parameters.json to export folder
    parameters_file_path = os.path.join(project_directory_path, 'classification-barebones', 'helpers', 'parameters.json')
    export_directory_parameter_file_path = os.path.join(export_directory_path, 'parameters.json')
    copy_file(parameters_file_path, export_directory_parameter_file_path)

    print("Copying runs folder to export folder...")
    runs_directory_path = os.path.join(project_directory_path, 'classification-barebones', "runs")
    export_directory_runs_directory = os.path.join(export_directory_path, "runs")
    shutil.copytree(runs_directory_path, export_directory_runs_directory)


    print("Zipping export folder...")
    # zip export folder
    # do not include .zip in destination file as shutil itself does it
    export_zip_path = os.path.join(project_directory_path, 'exports')
    zip_file(export_directory_path, export_zip_path)

    print("Uploading to S3...")
    # upload zipped to s3
    destination_key_zip = destination_directory + "export.zip"
    export_zip_path = export_zip_path + ".zip"
    upload_to_s3(export_bucket_name, destination_key_zip, export_zip_path)


if __name__ == '__main__':
    export()


