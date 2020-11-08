from torch.utils.data import DataLoader
from dataset import LoadImagesAndLabels
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from valid import calculate_class_wise_precision_recall_f1
import math
from torch.utils.tensorboard import SummaryWriter
from models import get_model
import argparse
import pickle
import torch.optim as optim
import os
from tqdm import tqdm


def train_model():


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

    classes = pickle.load(open(os.path.join(train_dir,'one_not.p'),'rb'))['classes']

    # train_dataset = LoadImagesAndLabels(transform=Pad(input_height, input_width, 'letterbox'),
    #                                     image_files_dir=train_dir, labels_file_dir=train_dir)
    # test_dataset = LoadImagesAndLabels(transform=Pad(input_height, input_width, 'letterbox'),
    #                                    image_files_dir=valid_dir, labels_file_dir=valid_dir)
    
    train_dataset = LoadImagesAndLabels(image_files_dir=train_dir,labels_file_dir=train_dir,
                                        padding_kind=padding_kind,padded_image_shape=(input_width,input_height),
                                        augment=augment)

    test_dataset = LoadImagesAndLabels(image_files_dir=valid_dir,labels_file_dir=valid_dir,
                                        padding_kind=padding_kind,padded_image_shape=(input_width,input_height),augment={})


    nw = min([os.cpu_count(), train_batch_size if train_batch_size > 1 else 0, 8])  # number of workers
    print("Num workers ",str(nw))
    trainloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,num_workers=nw)
    testloader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False,num_workers=nw)

    writer = SummaryWriter(comment=name)
    model, trainable_params = get_model(model_type,len(classes))

    if adam:
        optimizer = optim.Adam(trainable_params,lr=lr)
    else:
        optimizer = optim.SGD(trainable_params, lr=lr, momentum= 0.937, nesterov=True)



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


    criterion = nn.CrossEntropyLoss()

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_f1 = - math.inf
    best_validation_error = math.inf
    for epoch in range(start_epoch,start_epoch + num_epochs):
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
            # output_probs = torch.nn.functional.softmax(output)
            #
            # _,preds = torch.max(output_probs,1)

            training_loss = criterion(output,labels)

            training_loss.backward()
            optimizer.step()
            running_training_loss  = running_training_loss + training_loss.item()*imgs.size(0)
            print("\nRunning Training Loss")
            pbar.set_description(str(round(running_training_loss,2)))

        # pbar = tqdm(enumerate(testloader), total=len(testloader))
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


        f1s = []
        ps = []
        rs = []
        for class_name in classes:
            writer.add_scalar("Precision/"+class_name,stat_dict[class_name]['precision'],epoch)
            writer.add_scalar("Recall/"+class_name,stat_dict[class_name]['recall'],epoch)
            writer.add_scalar("F1/"+class_name,stat_dict[class_name]['f1'],epoch)
            f1s.append(stat_dict[class_name]['f1'])
            ps.append(stat_dict[class_name]['precision'])
            rs.append(stat_dict[class_name]['recall'])


        f1avg = sum(f1s)/len(f1s)
        rsavg = sum(rs)/len(rs)
        psavg = sum(ps)/len(ps)

        writer.add_scalar("F1avg",f1avg,epoch)
        writer.add_scalar("Pavg",psavg,epoch)
        writer.add_scalar("Ravg",rsavg,epoch)




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

    opt = parser.parse_args()
    print(opt)
    print("starting to train")
    train_model()
