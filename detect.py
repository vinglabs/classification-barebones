from dataset import LoadImages
from torch.utils.data import DataLoader
import torch
import cv2
import os
import numpy as np
import argparse
from models import get_model
import pickle
import shutil
from tqdm import tqdm
from timeit import default_timer
import torchvision

def detect_output():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = opt.weights
    images_dir = opt.source
    model_type = opt.model_type
    input_width = opt.width
    input_height = opt.height
    classes_dir = opt.classes_dir
    output_dir = opt.output
    padding_kind = opt.padding_kind
    augment = opt.augment
    pretrained = opt.pretrained
    normalization=opt.normalization


    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)



    checkpoint = torch.load(weights)
    train_one_not = pickle.load(open(os.path.join(classes_dir,'one_not.p'),'rb'))
    classes = train_one_not['classes']
    model,optimizer = get_model(model_type,len(classes),pretrained=pretrained)
    model.load_state_dict(checkpoint['model'])
    # detectloader = DataLoader(LoadImages(transform=Pad(input_height,input_width,'letterbox'),image_files_dir=images_dir),batch_size=32)

    if augment:
        if os.path.exists("augment.p"):
            augment_props = pickle.load(open("augment.p","rb"))
        else:
            raise("augment.p does not exists")
    else:
        augment_props = {}

    if not pretrained and normalization:
        try:
            params = pickle.load(open("normalization_parameters.p","rb"))
            mean = params['mean']
            std = params['std']
        except FileNotFoundError:
            print("normalization_parameters.p not found,reading from training set")
            try:
                mean,std = train_one_not['normalization_parameters']
            except:
                #Fallback for past detect
                print("normalization parameters not found in train set.Falling back to imagenst")
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

    elif pretrained and normalization:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    else:
        print("Normalization disabled.Using mean=0,stddev=1")
        mean = [0,0,0]
        std = [1,1,1]

    detectloader = DataLoader(LoadImages(image_files_dir=images_dir,padding_kind=padding_kind,
                                         padded_image_shape=(input_width,input_height),augment=augment_props,normalization_params=(mean,std)),batch_size=32)

    with torch.no_grad():
        for j,(filenames,imgs) in enumerate(tqdm(detectloader,desc="Running")):
            model.eval()
            imgs = imgs.to(device)
            model = model.to(device)
            torch.cuda.synchronize()
            t_start = default_timer()
            output = model(imgs)
            torch.cuda.synchronize()
            print("\n Time ",str(round(default_timer()-t_start,2))+"s")
            probs = torch.nn.functional.softmax(output)
            probs, preds = torch.topk(probs, 2 ,1)
            for i,pred in enumerate(preds):
                first_pred_index = pred[0].item()
                second_pred_index = pred[1].item()
                first_prediction = classes[first_pred_index]
                second_prediction = classes[second_pred_index]
                first_prob = probs[i][0].item()
                second_prob = probs[i][1].item()
                img = imgs[i].cpu().numpy()
                #std and mean are RGB
                img = img.transpose(1,2,0)*std + mean
                img = img[...,::-1]
                img = np.ascontiguousarray(img)
                # img = std * img + mean
                img = cv2.putText(img, first_prediction + "(" + str(round(first_prob,2)) + ")," + second_prediction + "(" + str(round(second_prob,2)) + ")", (10,10),
                                   fontFace= 1,fontScale=1,color=(255,255,255))
                filename = os.path.split(filenames[i])[1]
                savepath = os.path.join(output_dir,filename)
                cv2.imwrite(savepath,img)

                pickle_savepath = savepath.replace("jpg","p")
                pickle.dump({os.path.splitext(filename)[0]:[(first_pred_index,first_prob),(second_pred_index,second_prob)]},open(pickle_savepath,"wb"))






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--weights',type=str,default="best.pt")
    parser.add_argument('--width',type=int,default=256)
    parser.add_argument('--height',type=int,default=256)
    parser.add_argument('--model-type',type=str,default="resnet18")
    parser.add_argument('--classes-dir',type=str,default='')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument("--padding-kind",type=str,help="whole/letterbox/nopad")
    parser.add_argument("--augment",action='store_true', help='augment during detection')
    parser.add_argument("--pretrained",action="store_true",help="use pretrained base network")
    parser.add_argument("--normalization",action="store_true",help="normalization enable")



    opt = parser.parse_args()
    print(opt)
    print("starting to detect")
    detect_output()



