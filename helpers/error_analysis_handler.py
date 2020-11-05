import json
import os
import pickle
from helper_utils.error_analysis_tasks import get_data_urls,get_prediction_stats
from helper_utils.error_analysis_utils import format_stats,format_file_stats
import subprocess
from helper_utils.file_utils import create_directory
from helper_utils.s3_helpers import sync_to_S3_command


def error_analysis():
    parameters = json.load(open("parameters.json",'r'))
    project_directory_path = parameters['project_directory_path']
    train_directory = os.path.join(project_directory_path,"assets","dataset","train")
    train_labels = pickle.load(open(os.path.join(train_directory,"one_not.p"),'rb'))
    classes = train_labels['classes']
    input_directory = parameters['detect']['source']
    input_labels = pickle.load(open(os.path.join(input_directory,"one_not.p"),'rb'))
    prediction_directory = parameters['detect']['output']
    serve_port = parameters['error_analysis']['image_port']
    error_analysis_sync_bucket = parameters['error_analysis']['bucket_name']
    error_analysis_sync_key = parameters['error_analysis']['destination_key']

    print("Creating error analysis directory...")
    error_analysis_directory = os.path.join(project_directory_path, "classification-barebones", "error_analysis_data")
    create_directory(error_analysis_directory)
    if 'name' not in parameters['detect'].keys():
        name = "test"
    else:
        name = parameters['detect']['name']
    error_analysis_file_path = os.path.join(error_analysis_directory, name + ".p")

    print("Getting training urls")
    #class wise training data urls
    training_urls = get_data_urls(train_directory,train_labels,serve_port)

    print("Getting input urls")
    #class wise input data urls
    input_urls = get_data_urls(input_directory,input_labels,serve_port)

    print("Getting prediction urls")
    #class wise predictions(acc to gt) with class plotted on them
    prediction_urls = get_data_urls(prediction_directory,input_labels,serve_port)

    print("Getting Stats")
    #correctly predicted
    #incorrectly predicted
    #stats(f1,p,r)
    stats,file_stats = get_prediction_stats(prediction_directory,input_labels,serve_port)

    #convert file stats from {class_id:{"tp":[],"fp":[],"fn":[]}} to {"tp":{"class_id":[]},"fp":{"class_id":[]},...}
    file_stats = format_file_stats(file_stats)

    number_stats = {
        'training_urls': {class_index: len(stat) for class_index, stat in training_urls.items()},
        'input_urls': {class_index: len(stat) for class_index, stat in prediction_urls.items()},
        'prediction_urls': {class_index: len(stat) for class_index, stat in prediction_urls.items()},
        'tp_urls': {class_index: len(stat) for class_index, stat in file_stats['tp'].items()},
        'fp_urls': {class_index: len(stat) for class_index, stat in file_stats['fp'].items()},
        'fn_urls': {class_index: len(stat) for class_index, stat in file_stats['fn'].items()}

    }

    stats, metrics = format_stats(stats, classes)

    # json_data = {"training_urls": training_urls,
    #              "input_data_urls":input_urls,
    #              "prediction_urls":prediction_urls,
    #              "stats":stats,
    #              "file_stats":file_stats,
    #              "number_stats":number_stats,
    #              "metrics":metrics,
    #              "classes":classes}
    json_data = {
        "urls":{
            "training_urls":training_urls,
            "input_urls":input_urls,
            "prediction_urls":prediction_urls,
            "tp_urls": file_stats['tp'],
            "fp_urls": file_stats['fp'],
            "fn_urls":file_stats['fn'],
        },
        "number_stats":number_stats,
        "confusion":{
            "stats":stats,
            "metrics":metrics
        },
        "classes":classes

    }


    print("Saving pickle file...")
    pickle.dump(json_data,open(error_analysis_file_path,'wb'))

    print("Syncing results to s3...")
    command = sync_to_S3_command(error_analysis_directory,error_analysis_sync_bucket,error_analysis_sync_key)
    print("Running ",command)
    p2 = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for line in p2.stdout:
        print(line.decode(), end='')









if __name__ == "__main__":
    error_analysis()


