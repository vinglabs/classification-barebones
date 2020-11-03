import os
from glob import glob
from .error_analysis_utils import get_url
import pickle

def get_data_urls(data_directory,data_labels,port):
    images = glob(os.path.join(data_directory,"*.jpg"))
    classes = data_labels['classes']
    url_class_dict = {}
    for i in range(len(classes)):
        url_class_dict[i] = []

    for image in images:
        one_hot_label = data_labels[os.path.splitext(os.path.split(image)[1])[0]]
        class_index = max(enumerate(one_hot_label),key=lambda  x: x[1])[0]
        url_class_dict[class_index].append(get_url(image,port))

    return url_class_dict


def get_prediction_stats(prediction_directory,input_labels,port):

    images = glob(os.path.join(prediction_directory,"*.jpg"))
    classes = input_labels['classes']
    # predicted_labels = glob(os.path.join(prediction_directory,"*.p"))

    file_stats = {}
    stats = {}

    for i in range(len(classes)):
        file_stats[i] = {}
        file_stats[i]['tp'] = []
        file_stats[i]['fp'] = []
        file_stats[i]['fn'] = []
        stats[i] = {}
        stats[i]['tp'] = 0
        stats[i]['fp'] = 0
        stats[i]['fn'] = 0

    for image in images:
        filename = os.path.splitext(os.path.split(image)[1])[0]
        predicted_label = pickle.load(open(os.path.join(prediction_directory,filename+".p"),"rb"))
        one_hot_input_label = input_labels[filename]
        gt = max(enumerate(one_hot_input_label), key=lambda x: x[1])[0]
        pd = predicted_label[filename][0][0]

        if gt == pd:
            file_stats[gt]['tp'].append(get_url(image,port))
            stats[gt]['tp'] += 1

        else:
            file_stats[pd]['fp'].append(get_url(image,port))
            file_stats[gt]['fn'].append(get_url(image,port))
            stats[gt]['fn'] += 1
            stats[pd]['fp'] += 1

    for i in range(len(classes)):
        stats[i]['p'] = round(stats[i]['tp'] / (stats[i]['fp'] + stats[i]['tp'] + 1e-13), 2)
        stats[i]['r'] = round(stats[i]['tp'] / (stats[i]['fn'] + stats[i]['tp'] + 1e-13), 2)
        stats[i]['f1'] = round(2 * stats[i]['p'] * stats[i]['r'] / (stats[i]['p'] + stats[i]['r'] + 1e-13), 2)

    return stats,file_stats














