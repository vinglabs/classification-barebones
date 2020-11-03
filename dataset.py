from torch.utils.data import Dataset,DataLoader
import torch
from glob import glob
import cv2
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision

class LoadImagesAndLabels(Dataset):
    def __init__(self,transform,image_files_dir,labels_file_dir):

        self.filenames = glob(os.path.join(image_files_dir , "*.jpg"))
        self.label_filenames = glob(os.path.join(labels_file_dir , "*.p"))
        self.transform=transform

    def __getitem__(self, index):
        img = cv2.imread(self.filenames[index])
        img_filename = os.path.split(self.filenames[index])[1].split(".")[0]
        label = np.argmax(pickle.load(open(self.label_filenames[0],'rb'))[img_filename])
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.tensor(img,dtype=torch.float32)
        label = torch.tensor(label)
        if self.transform:
            img = self.transform(img)
            img = torchvision.transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False)
        return img,label,self.filenames[index]

    def __len__(self):
        return len(self.filenames)

class LoadImages(Dataset):
    def __init__(self,transform,image_files_dir):

        self.filenames = glob(os.path.join(image_files_dir , "*.jpg"))
        self.transform=transform

    def __getitem__(self, index):
        # img = Image.open(self.filenames[index])
        img = cv2.imread(self.filenames[index])
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.tensor(img,dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
            img = torchvision.transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=False)
        return self.filenames[index] ,img

    def __len__(self):
        return len(self.filenames)

class Pad:
    def __init__(self,padded_image_width,padded_image_height,kind):
        self.padded_image_width = padded_image_width
        self.padded_image_height = padded_image_height
        self.kind = kind
    def __call__(self,img):
        rows = img.size()[1]
        columns = img.size()[2]
        if self.kind == 'whole':
            padding_top = (self.padded_image_height - rows)//2
            padding_bottom = (self.padded_image_height - rows)//2 if (self.padded_image_height - rows)%2 ==0 else (self.padded_image_height - rows)//2 + 1
            padding_left = (self.padded_image_width - columns) // 2
            padding_right = (self.padded_image_width - columns) // 2 if (self.padded_image_width - columns) % 2 == 0 else (self.padded_image_width - columns) // 2 + 1
            return torch.nn.functional.pad(img,(padding_left,padding_right,padding_top,padding_bottom))
        elif self.kind == 'letterbox':
            ar = columns/rows
            if ar > 1:

                new_width = self.padded_image_width
                new_height = int(new_width/ar)
                padding_top = (self.padded_image_height - new_height) // 2
                padding_bottom = (self.padded_image_height - new_height) // 2 if (self.padded_image_height - new_height) % 2 == 0 else (self.padded_image_height - new_height) // 2 + 1
                img = torch.nn.functional.interpolate(img.unsqueeze(0),(new_height,new_width))
                img = img.squeeze(0)
                return torch.nn.functional.pad(img,(0,0,padding_top,padding_bottom))

            else:
                new_height = self.padded_image_height
                new_width = int(new_height*ar)
                padding_left = (self.padded_image_width - new_width) // 2
                padding_right = (self.padded_image_width - new_width) // 2 if (self.padded_image_width - new_width) % 2 == 0 else (self.padded_image_width - new_width) // 2 + 1
                img = torch.nn.functional.interpolate(img.unsqueeze(0), (new_height, new_width))
                img = img.squeeze(0)
                return torch.nn.functional.pad(img, (padding_right, padding_left, 0, 0))




#
# dataset = LoadImages(transform=Pad(300,300,'letterbox'),image_files_dir="train_images\\")
# dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
# img = next(iter(dataloader))[1]
# print(img.size())
# plt.imshow(img.squeeze(0).numpy().transpose(1,2,0).astype(np.uint8))
# plt.show()