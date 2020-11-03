import os
from helper_utils.file_utils import copy_file,unzip_file,create_directory
from helper_utils.s3_helpers import download_from_s3
import json


def read_parameter_json(filename):
    parameters = json.load(open(filename, 'r'))
    return parameters

def import_project():
    parameters = read_parameter_json('parameters.json')
    project_directory_path = os.path.normpath(parameters["project_directory_path"])
    export_zip_path = os.path.join(project_directory_path,'exports.zip')
    venv_path = os.path.normpath(parameters["venv_path"])
    import_bucket_name = parameters['import']['bucket_name']
    import_source_key = parameters['import']['source_key']

    print("Downloading exports...")
    #download export.zip from s3
    download_from_s3(import_bucket_name,import_source_key,export_zip_path)

    print("Creating destination directory...")
    #create exports folder
    exports_directory_path = os.path.join(project_directory_path,'exports')
    create_directory(exports_directory_path)

    print("Unzipping export...")
    #unzip export.zip
    unzip_file(export_zip_path,exports_directory_path)

    print("Altering parameters.json...")
    #Reading parameter file from exports folder
    imported_parameter_path = os.path.join(project_directory_path,'exports','parameters.json')
    imported_parameter_path_alternate = os.path.join(project_directory_path, 'exports', 'parameter.json')
    try:
        imported_parameters = read_parameter_json(imported_parameter_path)
    except FileNotFoundError:
        imported_parameters = read_parameter_json(imported_parameter_path_alternate)

    imported_parameters['venv_path'] = venv_path
    imported_parameters['project_directory_path'] = project_directory_path

    print("Overwriting parameters.json...")
    #save imported parameter.json
    json.dump(imported_parameters,open("parameters.json",'w'),indent=4)

    print("The model has been imported in exports directory.Edit parameters.json to proceed...")






if __name__ == '__main__':
    import_project()


