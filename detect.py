from dataset import LoadImages,Pad
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

def detect_output():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = opt.weights
    images_dir = opt.source
    model_type = opt.model_type
    input_width = opt.width
    input_height = opt.height
    classes_dir = opt.classes_dir
    output_dir = opt.output


    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)



    checkpoint = torch.load(weights)
    classes = pickle.load(open(os.path.join(classes_dir,'one_not.p'),'rb'))['classes']
    model,optimizer = get_model(model_type,len(classes))
    model.load_state_dict(checkpoint['model'])
    detectloader = DataLoader(LoadImages(transform=Pad(input_height,input_width,'letterbox'),image_files_dir=images_dir),batch_size=32)


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
            img = img.transpose(1, 2, 0)[...,::-1]
            img = np.ascontiguousarray(img)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
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


    opt = parser.parse_args()
    print(opt)
    print("starting to detect")
    detect_output()



