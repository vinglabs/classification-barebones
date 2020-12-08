import random
import numpy as np
import cv2
from yolo_label_utils import convert_label_to_xyxy,find_coordinates_with_max_area_for_empty_cut,\
    find_crop_dimensions_for_empty_cut,get_roi,set_roi,blur_roi



def rotate(img,angle_range,mode="crop"):
    if angle_range == (0,0):
        return img
    min_angle,max_angle = angle_range
    angle = round(random.uniform(min_angle,max_angle),2)
    rmat = cv2.getRotationMatrix2D(center=(img.shape[1]/2,img.shape[0]/2),angle=angle,scale=1)
    if mode == "no-crop":
        #sin,cos from rotation matrix
        cos = abs(rmat[0, 0])
        sin = abs(rmat[0, 1])

        #new width = h*sin + w*cos
        #new_height = h*cos + b*sin
        new_width = int(img.shape[1] * cos + img.shape[0] * sin)
        new_height = int(img.shape[1] * sin + img.shape[0] * cos)

        #moving center to new_width/2,new_height/2
        new_center_x = new_width / 2 - img.shape[1] / 2
        new_center_y = new_height / 2 - img.shape[0] / 2

        #modifying rot mat to reflect above movement
        rmat[0, 2] += new_center_x
        rmat[1, 2] += new_center_y

        rot_img = cv2.warpAffine(img, rmat, (new_width, new_height))

        #resizing back to initial size
        rot_img = cv2.resize(rot_img,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_AREA)

    elif mode == "crop":
        rot_img = cv2.warpAffine(img, rmat,(img.shape[1],img.shape[0]))
        
    else:
        raise("Incorrect rotation mode specified!")

    return rot_img


def flip_lr(img):
    if random.random() < 0.5:
        img = np.fliplr(img)
    return img

def flip_ud(img):
    if random.random() < 0.5:
        img = np.flipud(img)

    return img


def hsv(img,hgain=0.5,sgain=0.5,vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.int16)

    #lut=> look up table
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    #cv2.LUT(img,lut) => img = [0,1,2,3] lut=>[5,6,7,8] ===> output[i] = lut[img[i]] i.e. looks up for values in lut for each image value
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img


def translate(img,translate_factor=0.5,mode="crop"):
    if translate_factor == 0:
        return img
    translation_coeff = np.random.uniform(0,translate_factor)
    translate_x = translation_coeff*img.shape[1]
    translate_y = translation_coeff*img.shape[0]

    translation_matrix = np.array([[1,0,translate_x],[0,1,translate_y]])
    height,width = img.shape[0:2]
    if mode == "no-crop":
        new_width = int(img.shape[1] + abs(translate_x))
        new_height = int(img.shape[0] + abs(translate_y))
        img = cv2.warpAffine(img,translation_matrix,(new_width,new_height))
        img = cv2.resize(img,(width,height),interpolation=cv2.INTER_AREA)
    elif mode == "crop":
        img = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
    else:
        raise("Incorrect Mode specified!")

    return img


def cutout(img,yolo_label):
    img_dims = img.shape[0:2]
    labels = convert_label_to_xyxy(yolo_label[1:],img_dims)
    coords = find_coordinates_with_max_area_for_empty_cut(labels,img_dims)
    crop_dims = find_crop_dimensions_for_empty_cut(coords)
    img = cv2.rectangle(img,crop_dims[0],crop_dims[1],color=(0,0,0),thickness=-1)
    return img




def rotate_90(img):
    if random.random() < 0.5:
        if random.random() <= 0.5:
            if random.random() <= 0.5:
                return cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
            else:
                return cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return cv2.rotate(img,cv2.ROTATE_180)
    else:
        return img

def blur(img,kernel_size):
    return cv2.GaussianBlur(img,kernel_size,0)

def logo_blur(img,yolo_label):
    if random.random() < 0.5:
        img_dims = img.shape[0:2]
        label_xyxy = convert_label_to_xyxy(yolo_label[1:], img_dims)
        roi_img, area = get_roi(img, label_xyxy)

        if area < 4000:
            roi_img = blur_roi(roi_img, (7, 7))
        else:
            roi_img = blur_roi(roi_img, (13, 13))

        img = set_roi(roi_img, img, label_xyxy)


    return img





