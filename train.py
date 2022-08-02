from torch.utils.data import DataLoader
from dataset import LoadImagesAndLabels
from utils import calculate_class_weights,xavier_initialization
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from valid import calculate_class_wise_precision_recall_f1
import math
from torch.utils.tensorboard import SummaryWriter
from models import get_model,get_batch_norm_parameters
import argparse
import pickle
import torch.optim as optim
import os
from tqdm import tqdm
import torchvision
import random
import torch.backends.cudnn as cudnn
import numpy
import wandb
import json

def train_model():

    wandb.login(key='924764f1e5cac1fa896fada3c8d64b39a0926275')

    

    train_batch_size = opt.batch_size
    train_dir = opt.train_data_dir
    valid_dir = opt.valid_data_dir
    model_type = opt.model_type
    num_epochs = opt.epochs
    resume = opt.resume
    input_width = opt.width
    input_height = opt.height
    name = opt.name
    lr = opt.lr
    adam = opt.adam
    device = opt.device
    weights = opt.weights
    wdir = opt.weights_dir
    padding_kind = opt.padding_kind
    pretrained = opt.pretrained
    decay = opt.decay
    normalization = opt.normalization
    subdataset = opt.subdataset
    test_on_train = opt.test_on_train

        
    #setting seed
    seed=0
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    # Reduce randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True

    #device selection
    if device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            print("GPU not available.Switching to CPU")
            device = 'cpu'
    elif device == 'cpu':
        device = 'cpu'
    else:
        raise("The selected device is not available")

    print("Using ",device)

    if os.path.exists("augment.p"):
        print("Reading augmentation specs from augment.p")
        augment = pickle.load(open("augment.p","rb"))
        print("Using augment = ",augment)
    else:
        augment = {}
    
    

    train_one_not = pickle.load(open(os.path.join(train_dir,'one_not.p'),'rb'))
    classes = train_one_not['classes']

    config = {'train_batch_size': train_batch_size,
                "model_type":model_type,
                "num_epochs":num_epochs,
                "resume":resume,
                "input_width":input_width,
                "name":name,
                "input_height":input_height,
                "lr":lr,
                "adam":adam,
                "weights":weights,
                "wdir":wdir,
                "padding_kind":padding_kind,
                "pretrained":pretrained,
                "decay":decay,
                "normalization":normalization,
                "subdataset":subdataset,
                "test_on_train":test_on_train,
                "augment":augment,
                "classes":classes}

    print("Calculating Normalization Parameters...")
    if not pretrained and normalization:
        #mean,std = calculate_normalization_parameters(train_dir)
        #pickle.dump({"mean":mean,"std":std},open("normalization_parameters.p","wb"))
        mean,std = train_one_not['normalization_parameters']
        print("Using NP from training set")
        print("norm params ",mean,std)
    elif pretrained and normalization:
        print("Using pretrained NP")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        print("norm params ",mean,std)

    else:
        print("Normalization disabled.Using mean=0 and stddev=1 as norm params")
        mean = [0, 0, 0]
        std = [1,1,1]
        print("norm params ",mean,std)



    train_dataset = LoadImagesAndLabels(image_files_dir=train_dir,labels_file_dir=train_dir,
                                        padding_kind=padding_kind,padded_image_shape=(input_width,input_height),
                                        augment=augment,normalization_params = (mean,std),subdataset=subdataset)

    test_dataset = LoadImagesAndLabels(image_files_dir=valid_dir if not test_on_train else train_dir,labels_file_dir=valid_dir if not test_on_train else train_dir,
                                        padding_kind=padding_kind,padded_image_shape=(input_width,input_height),augment={},normalization_params = (mean,std),
                                       subdataset=subdataset)

    class_weights = calculate_class_weights(train_dir,device)
    print("class weights ",class_weights)

    nw = min([os.cpu_count(), train_batch_size if train_batch_size > 1 else 0, 8])  # number of workers
    print("Num workers ",str(nw))
    trainloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,num_workers=nw)
    testloader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False,num_workers=nw)

    writer = SummaryWriter(comment=name)
    model, trainable_params = get_model(model_type,len(classes),pretrained)
    if model_type in  ['colornet_avg', 'colornetlite_avg' , 'colornet' , 'colornetlite']:
        print("Initializaing xavier")
        xavier_initialization(model)

    # as weight decay is not applied to batch norm parameters,
    # creating param groups
    pg0,pg1,pg2 = [],[],[]
    batch_norm_params = get_batch_norm_parameters(model)
    for k,v in dict(model.named_parameters()).items():
        if v.requires_grad:
            if k not in batch_norm_params and  ".bias" in k:
                #biases except bn
                pg2 += [v]
            elif k not in batch_norm_params and '.weight' in k:
                #weights except bn
                pg1 += [v]
            else:
                #rest
                pg0 += [v]

    print("Biases(except bn) = {},Weights(except bn) = {},Remaining = {}".
          format(str(sum([x.numel() for x in pg0])),
                 str(sum([x.numel() for x in pg1])),
                 str(sum([x.numel() for x in pg2]))))

    if adam:
        optimizer = optim.Adam(pg0 + pg2,lr=lr)
    else:
        optimizer = optim.SGD(pg0 + pg2, lr=lr, momentum= 0.937, nesterov=True)

    #weight decay for weights parameter
    print("Adding decay = ",decay)
    optimizer.add_param_group({'params':pg1,'weight_decay':decay})
    # #no weight decay for biases
    # optimizer.add_param_group({'params':pg2})

    del pg0,pg1,pg2

    model = model.to(device)
    nl = len(list(model.parameters()))
    np = sum(x.numel() for x in model.parameters())
    ng = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print("Number of Layers : {},"
          "Number of Parameters: {},"
          "Number of trainable parameters: {}".format(str(nl),str(np),str(ng)))


    if resume:
        checkpoint = torch.load(os.path.join(wdir,weights))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0


    criterion = nn.CrossEntropyLoss(weight=class_weights)


    run = wandb.init(project='alpla-classification',config=config,job_type='train',name=name)
    print("Running wandb run ",wandb.run.name)

    wandb.define_metric("train/loss",summary="min")
    wandb.define_metric("test/acc",summary="max")
    wandb.define_metric("test/loss",summary="min")
    wandb.define_metric("F1avg",summary="max")
    wandb.define_metric("Pavg",summary="max")
    wandb.define_metric("Ravg",summary="max")



    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_f1 = - math.inf
    best_validation_error = math.inf
    for epoch in range(start_epoch,start_epoch + num_epochs):
        model.train()
        test_labels = []
        test_predictions = []
        running_training_loss = 0.0
        running_test_loss = 0.0
        pbar = tqdm(enumerate(trainloader),total=len(trainloader))

        for i,(imgs,labels,_) in pbar:
            imgs = imgs.to(device)

            labels = labels.to(device)
            optimizer.zero_grad()

            output = model(imgs)

            if model_type == 'googlenet' and not pretrained:
                output = output.logits

            training_loss = criterion(output,labels)

            training_loss.backward()
            optimizer.step()
            running_training_loss  = running_training_loss + training_loss.item()*imgs.size(0)
            print("\nRunning Training Loss")
            pbar.set_description(str(round(running_training_loss,2)))

        # pbar = tqdm(enumerate(testloader), total=len(testloader))
        with torch.no_grad():
            for i,(imgs,labels,_) in enumerate(tqdm(testloader)):
                model.eval()
                imgs = imgs.to(device)
                labels = labels.to(device)

                output = model(imgs)
                output_probs = torch.nn.functional.softmax(output)

                _, preds = torch.max(output_probs, 1)

                test_labels.append(labels)
                test_predictions.append(preds)

                test_loss = criterion(output, labels)

                running_test_loss = running_test_loss + test_loss.item()*imgs.size(0)
                print("\nRunning Test Loss ",round(running_test_loss,2))

        # scheduler.step()
        stat_dict = calculate_class_wise_precision_recall_f1(test_predictions,test_labels,classes)
        epoch_training_loss = running_training_loss/len(train_dataset)
        epoch_test_loss = running_test_loss/len(test_dataset)

        writer.add_scalar("Loss/train",epoch_training_loss,epoch)
        writer.add_scalar("Loss/test",epoch_test_loss,epoch)
        writer.add_scalar("Accuracy/test",stat_dict['accuracy'],epoch)
        wandb.log({"train/loss":epoch_training_loss,"test/loss":epoch_test_loss,"test/acc":stat_dict['accuracy'],"epoch":epoch},step=epoch)



        f1s = []
        ps = []
        rs = []
        for class_name in classes:
            writer.add_scalar("Precision/"+class_name,stat_dict[class_name]['precision'],epoch)
            writer.add_scalar("Recall/"+class_name,stat_dict[class_name]['recall'],epoch)
            writer.add_scalar("F1/"+class_name,stat_dict[class_name]['f1'],epoch)
            wandb.log({"{}/precision".format(class_name):stat_dict[class_name]['precision'],
                       "{}/recall".format(class_name):stat_dict[class_name]['recall'],
                       "{}/f1".format(class_name):stat_dict[class_name]['f1']},step=epoch)
            f1s.append(stat_dict[class_name]['f1'])
            ps.append(stat_dict[class_name]['precision'])
            rs.append(stat_dict[class_name]['recall'])


        f1avg = sum(f1s)/len(f1s)
        rsavg = sum(rs)/len(rs)
        psavg = sum(ps)/len(ps)

        writer.add_scalar("F1avg",f1avg,epoch)
        writer.add_scalar("Pavg",psavg,epoch)
        writer.add_scalar("Ravg",rsavg,epoch)
        wandb.log({"F1avg":f1avg,"Pavg":psavg,"Ravg":rsavg,"epoch":epoch},step=epoch)





        chkpt = {'epoch': epoch,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()
                 }

        torch.save(chkpt,os.path.join(wdir,"last.pt"))

        if f1avg > best_f1:
            best_f1 = f1avg
            torch.save(chkpt,os.path.join(wdir,"best.pt"))
        elif best_f1 == f1avg:
            if best_validation_error > epoch_test_loss:
                print("Saving due to same f1 but better loss")
                torch.save(chkpt,os.path.join(wdir,"best.pt"))
                best_validation_error = epoch_test_loss

        if epoch_test_loss < best_validation_error:
            best_validation_error = epoch_test_loss

        print("Epoch ", epoch)
        print("Training Loss ", round(epoch_training_loss,4))
        print("Validation Loss ", round(epoch_test_loss,4))
        print("Mean P", round(psavg,4))
        print("Mean R ", round(rsavg,4))
        print("Mean F1 ", round(f1avg,4))

    trained_model_artifact = wandb.Artifact(name="trained_model",
                                                type="model",
                                                metadata=dict(config))
    trained_model_artifact.add_dir(wdir,name="weights")
    run.log_artifact(trained_model_artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--train-data-dir',type=str,default="")
    parser.add_argument('--valid-data-dir',type=str,default="")
    parser.add_argument('--model-type',type=str,default="resnet18")
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--width',type=int,default=256)
    parser.add_argument('--height',type=int,default=256)
    parser.add_argument('--name', default='', help='name of run')
    parser.add_argument('--device', default='gpu', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--lr',type=float,default=0.001,help="learning rate of optimizer")
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--weights',type=str,help='weights to be used for resumption',default='best.pt')
    parser.add_argument('--weights-dir',type=str,help='dir to save weights')
    parser.add_argument("--padding-kind",type=str,help="whole/letterbox/nopad")
    parser.add_argument("--pretrained",action="store_true",help="use pretrained base network")
    parser.add_argument("--decay",type=float,default=0.0,help="weight decay")
    parser.add_argument("--normalization",action="store_true",help="normalization enable")
    parser.add_argument("--subdataset",action="store_true",help="normalization enable")
    parser.add_argument("--test-on-train",action="store_true",help="normalization enable")



    opt = parser.parse_args()
    print(opt)
    print("starting to train")
    train_model()
