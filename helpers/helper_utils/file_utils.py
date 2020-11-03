import os
import shutil
import zipfile



def create_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

def unzip_file(source,destination):
    with zipfile.ZipFile(source) as zf:
        zf.extractall(destination)

def copy_file(source,destination):
    shutil.copyfile(source,destination)

def zip_file(source,destination):
    shutil.make_archive(destination,'zip',source)
