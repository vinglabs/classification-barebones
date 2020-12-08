from torch.utils.data import Dataset,DataLoader
import torch
from glob import glob
import cv2
import pickle
import os
import numpy as np
from aug_engine import rotate,flip_lr,flip_ud,translate,hsv,cutout,rotate_90,logo_blur
import torchvision


class LoadImagesAndLabels(Dataset):
    def __init__(self,image_files_dir,labels_file_dir,padding_kind,padded_image_shape,augment,normalization_params):

        self.filenames = glob(os.path.join(image_files_dir , "*.jpg"))
        self.label_filenames = glob(os.path.join(labels_file_dir , "*.p"))
        self.padding_kind=padding_kind
        self.augment = augment
        self.padded_image_shape = padded_image_shape
        self.normalization_params = normalization_params

    def __getitem__(self, index):
        mean,std = self.normalization_params
        img = cv2.imread(self.filenames[index])
        img_filename = os.path.split(self.filenames[index])[1].split(".")[0]
        one_not = pickle.load(open(self.label_filenames[0],'rb'))
        label = np.argmax(one_not[img_filename])
        if ("cutout" in self.augment.keys() and self.augment['cutout']) or ("logo_blur" in self.augment.keys() and self.augment['logo_blur']):
            yolo_label = one_not[img_filename + "_yolo"]
        img = pad(img,self.padded_image_shape,self.padding_kind)

        if bool(self.augment):
            if "cutout" in self.augment.keys() and self.augment['cutout']:
                img = cutout(img,yolo_label)

            if "logo_blur" in self.augment.keys() and self.augment['logo_blur']:
                img = logo_blur(img,yolo_label)

            if "rotate" in self.augment.keys():
                img = rotate(img,self.augment['rotate']['angle_range'],mode=self.augment['rotate']['mode'])

            if "flip_lr" in self.augment.keys() and self.augment['flip_lr']:
                img = flip_lr(img)

            if "flip_ud" in self.augment.keys() and self.augment['flip_ud']:
                img = flip_ud(img)

            if "translate" in self.augment.keys():
                img = translate(img,self.augment['translate']['translate_factor'],self.augment['translate']['mode'])

            if "hsv" in self.augment.keys():
                img = hsv(img,self.augment['hsv']['hgain'],self.augment['hsv']['sgain'],self.augment['hsv']['vgain'])

            if "rotate90" in self.augment.keys() and self.augment['rotate90']:
                img = rotate_90(img)



        save_img = False
        if save_img and index < 100:
            if not os.path.exists("data-samples"):
                os.mkdir("data-samples")

            cv2.imwrite(os.path.join("data-samples",img_filename+".jpg"),img)

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.tensor(img,dtype=torch.float32)
        label = torch.tensor(label)
        img = torchvision.transforms.functional.normalize(img, mean, std, inplace=False)

        return img,label,self.filenames[index]

    def __len__(self):
        return len(self.filenames)

class LoadImages(Dataset):
    def __init__(self,image_files_dir,padding_kind,padded_image_shape,augment,normalization_params):

        self.filenames = glob(os.path.join(image_files_dir , "*.jpg"))
        self.padding_kind = padding_kind
        self.padded_image_shape = padded_image_shape
        self.augment = augment
        self.normalization_params = normalization_params

    def __getitem__(self, index):
        mean,std = self.normalization_params
        img = cv2.imread(self.filenames[index])
        img = pad(img, self.padded_image_shape, self.padding_kind)

        if bool(self.augment):
            if "rotate" in self.augment.keys():
                img = rotate(img,self.augment['rotate']['angle_range'],mode=self.augment['rotate']['mode'])

            if "flip_lr" in self.augment.keys() and self.augment['flip_lr']:
                img = flip_lr(img)

            if "flip_ud" in self.augment.keys() and self.augment['flip_ud']:
                img = flip_ud(img)

            if "translate" in self.augment.keys():
                img = translate(img,self.augment['translate']['translate_factor'],self.augment['translate']['mode'])

            if "hsv" in self.augment.keys():
                img = hsv(img,self.augment['hsv']['hgain'],self.augment['hsv']['sgain'],self.augment['hsv']['vgain'])


        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.tensor(img,dtype=torch.float32)

        img = torchvision.transforms.functional.normalize(img, mean, std, inplace=False)

        return self.filenames[index] ,img

    def __len__(self):
        return len(self.filenames)


def pad(img,final_dims,kind):
    height,width = img.shape[0:2]
    padded_image_height,padded_image_width = final_dims
    if kind == 'whole':
        padding_top = (padded_image_height - height) // 2
        padding_bottom = (padded_image_height - height) // 2 if (padded_image_height - height) % 2 == 0 else (padded_image_height - height) // 2 + 1
        padding_left = (padded_image_width - width) // 2
        padding_right = (padded_image_width - width) // 2 if (padded_image_width - width) % 2 == 0 else (padded_image_width - width) // 2 + 1
        img = cv2.copyMakeBorder(img,padding_top,padding_bottom,padding_left,padding_right,borderType=cv2.BORDER_CONSTANT)

        return img
    elif kind == 'letterbox':
        ar = width / height
        if ar > 1:

            new_width = padded_image_width
            new_height = int(new_width / ar)
            padding_top = (padded_image_height - new_height) // 2
            padding_bottom = (padded_image_height - new_height) // 2 if (padded_image_height - new_height) % 2 == 0 else (padded_image_height - new_height) // 2 + 1
            img = cv2.resize(img,(new_width,new_height),interpolation=cv2.INTER_AREA)
            img = cv2.copyMakeBorder(img,padding_top,padding_bottom,0,0,cv2.BORDER_CONSTANT)
            return img

        else:
            new_height = padded_image_height
            new_width = int(new_height * ar)
            padding_left = (padded_image_width - new_width) // 2
            padding_right = (padded_image_width - new_width) // 2 if (padded_image_width - new_width) % 2 == 0 else (padded_image_width - new_width) // 2 + 1
            img = cv2.resize(img,(new_width,new_height),interpolation=cv2.INTER_AREA)
            img = cv2.copyMakeBorder(img,0,0,padding_left,padding_right,cv2.BORDER_CONSTANT)
            return img

    elif kind == 'nopad':

        img = cv2.resize(img,(padded_image_width,padded_image_height),interpolation=cv2.INTER_AREA)

        return img

    elif kind == 'none':
        return img



def calculate_normalization_parameters(train_images_dir):
    filenames = glob(os.path.join(train_images_dir,"*.jpg"))
    m = np.zeros((1, 3))
    s = np.zeros((1, 3))
    for filename in filenames:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img / 255
        m += np.mean(img, axis=tuple(range(img.ndim - 1)))
        s += np.std(img, axis=tuple(range(img.ndim - 1)))

    mean = m / len(filenames)
    std = s / len(filenames)

    return mean[0],std[0]





# img = cv2.imread("X:\\python_projects\\vinglabs\\augmentation-engine\\images\\ariel\\20-10-20-09-54-48_0_0.jpg")
# print(img.shape)
# # img = cv2.resize(img,(300,400))
# img = pad(img,(500,500),"nopad")
# cv2.imshow("win",img)
# cv2.waitKey(0)
# # dataset = LoadImages(transform=Pad(300,300,'letterbox'),image_files_dir="train_images\\")
# # dataloader = DataLoader(dataset,batch_size=1,shuffle=True)
# # img = next(iter(dataloader))[1]
# # print(img.size())
# # plt.imshow(img.squeeze(0).numpy().transpose(1,2,0).astype(np.uint8))
# # plt.show()
# dataset = LoadImagesAndLabels(image_files_dir="../assets/dataset/train/",labels_file_dir="../assets/dataset/train/",
#                               padding_kind="nopad",padded_image_shape=(500,500),
#                               augment=
#                               {"rotate":
#                                    {"angle_range":(0,90),"mode":"no-crop"},
#                                 "translate":
#                                     {"translate_factor":0.1,"mode":"no-crop"},
#                                 "flip_ud":True,
#                                 "flip_lr":True,
#                                 "hsv":{"hgain":0,"sgain":0,"vgain":0.2}
#                               })
# dataloader = DataLoader(dataset,batch_size=1)
# for img,label,filename in dataloader:
#     print(filename)