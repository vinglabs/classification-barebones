import os
from helper_utils.file_utils import create_directory,unzip_file
from helper_utils.s3_helpers import download_from_s3,download_folder_contents_from_s3
import json


def read_parameter_json():
    parameters = json.load(open('parameters.json', 'r'))
    return parameters


def create_assets_directory():

    print("Reading parameter file...")
    parameters = read_parameter_json()
    project_directory_path = os.path.normpath(parameters["project_directory_path"])
    assets_directory_path = os.path.join(project_directory_path, "assets")
    dataset_s3_key = parameters['dataset_s3_info']['dataset_key']
    dataset_s3_bucket = parameters['dataset_s3_info']['bucket_name']
    dataset_destination_filename = parameters['dataset_s3_info']['destination_filename']



    print("Creating assets directory...")
    # creatw assets directory
    assets_directory_path = os.path.normpath(assets_directory_path)
    create_directory(assets_directory_path)

    print("Downloading dataset from ", dataset_s3_key, "...")
    # download dataset from s3
    dataset_destination_path = os.path.join(assets_directory_path, dataset_destination_filename)
    download_from_s3(dataset_s3_bucket, dataset_s3_key, dataset_destination_path)

    print("Unzipping data...")
    # unzip dataset
    unzip_file(source=dataset_destination_path, destination=assets_directory_path)



if __name__=="__main__":
    create_assets_directory()
