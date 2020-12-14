import pickle
import os
from glob import glob
import numpy as np
import torch


def calculate_class_weights(train_dir):
    img_filenames = glob(os.path.join(train_dir,"*.jpg"))
    one_not = pickle.load(open(glob(os.path.join(train_dir,"*.p"))[0], 'rb'))
    num_classes = len(one_not['classes'])
    class_freq = [0]*num_classes
    for img_filename in img_filenames:
        filename = os.path.split(img_filename)[1].split(".")[0]
        label = np.argmax(one_not[filename])
        class_freq[label] += 1

    print("class frequency ",class_freq)
    min_freq = min(class_freq)
    max_freq = max(class_freq)
    fraction_variation = (max_freq-min_freq)/max_freq
    if fraction_variation > 0.2:

        total_examples = sum(class_freq)
        weights = [round(total_examples/class_freq_indi,4) for class_freq_indi in class_freq]
        weights = torch.tensor(weights,dtype=torch.float32)

    else:

        weights = torch.tensor([1]*num_classes,dtype=torch.float32)


    return weights

def xavier_initialization(model):
    for m in model.modules():
        t = type(m)
        if t is torch.nn.Conv2d:
            torch.nn.init.xavier_normal_(m.weight)







