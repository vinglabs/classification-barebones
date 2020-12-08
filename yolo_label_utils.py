import cv2
import numpy as np

def convert_label_to_xyxy(xywh,img_dims):
    height,width = img_dims
    x_center_rel, y_center_rel, w_rel, h_rel = xywh
    x_min = max(0,int(x_center_rel * width - w_rel * width / 2))
    y_min = max(0,int(y_center_rel * height - h_rel * height / 2))
    x_max = min(width,int(x_center_rel * width + width * w_rel / 2))
    y_max = min(height,int(y_center_rel * height + height * h_rel / 2))

    return (x_min,y_min),(x_max,y_max)


def plot_area(img,rect_coordinates):
    (x_min, y_min), (x_max, y_max)  = rect_coordinates
    img= cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0))
    return img


def find_coordinates_with_max_area_for_empty_cut(label,img_dims):
    (x_min, y_min), (x_max, y_max) = label
    height, width = img_dims
    #label completely in second quadrant
    if x_min < width//2 and x_max <= width//2 and y_max <= height//2 and y_min < height//2:
        width_right = width - x_max
        area_right = width_right * height
        height_bottom = height - y_max
        area_bottom = width * height_bottom
        if area_right > area_bottom:
            return (x_max,0),(width,height)
        else:
            return (0,y_max),(width,height)

    # label completely in first quadrant
    elif x_min >= width//2 and x_max > width//2 and y_max <= height//2 and y_min < height//2:
        width_left = width - x_min
        area_left = width_left * height
        height_bottom = height - y_max
        area_bottom = width * height_bottom
        if area_left > area_bottom:
            return (0, 0), (x_min, height)
        else:
            return (0, y_max), (width, height)

        # label completely in third quadrant
    elif x_min < width // 2 and x_max <= width // 2 and y_max > height // 2 and y_min >= height // 2:
        width_right = width - x_max
        area_right = width_right * height
        height_top = height - y_min
        area_top = width * height_top
        if area_right > area_top:
            return (x_max, 0), (width, height)
        else:
            return (0, 0), (width, y_min)

    #label completely in 4th quad
    elif x_min >= width//2 and x_max > width//2 and y_max > height//2 and y_min >= height//2:
        width_left = width - x_min
        area_left = width_left * height
        height_top = height - y_min
        area_bottom = width * height_top
        if area_left > area_bottom:
            return (0, 0), (x_min, height)
        else:
            return (0, 0), (width, y_min)

    #label lies in 1st and second quadrsnt pick bottom region
    elif x_min < width//2 and x_max > width//2 and y_min < height//2 and y_max <= height//2:
        return (0,y_max),(width,height)

    # label lies in 1st and fourth quadrsnt pick left region
    elif x_min >= width // 2 and x_max > width // 2 and y_min < height // 2 and y_max > height // 2:
        return (0, 0), (x_min,height)

    # label lies in 3rd and fourth quadrsnt pick top region
    elif x_min < width // 2 and x_max > width // 2 and y_min >= height // 2 and y_max > height // 2:
        return (0, 0), (width,y_min)

    # label lies in 2nd and 3rd quadrant pick right region
    elif x_min < width // 2 and x_max <= width // 2 and y_min < height // 2 and y_max > height // 2:
        return (x_max, 0), (width, height)

    #label lies in all quadrants
    elif x_min < width // 2 and x_max > width // 2 and y_min < height // 2 and y_max > height // 2:
        area_top = width*y_min
        area_left = x_min*height
        area_bottom = width*(height-y_max)
        area_right = (width-x_max)*height

        areas = [area_top,area_left,area_bottom,area_right]
        max_area = max(areas)
        max_area_index =areas.index(max_area)
        if max_area_index == 0:
            return (0,0),(width,y_min)
        elif max_area_index == 1:
            return (0,0),(x_min,height)
        elif max_area_index == 2:
            return (0,y_max),(width,height)
        elif max_area_index == 3:
            return (x_max,0),(width,height)
    else:
        print("Case not handled")
        return (None,None),(None,None)

def find_crop_dimensions_for_empty_cut(max_area_coords):
    (x_min, y_min), (x_max, y_max) = max_area_coords
    max_area_center_x = (x_min + x_max)//2
    max_area_center_y = (y_min + y_max)//2
    max_area_crop_width = x_max - x_min
    max_area_crop_height = y_max - y_min
    rand_width = np.random.randint(max_area_crop_width//2,max_area_crop_width)
    rand_height = np.random.randint(max_area_crop_height//2,max_area_crop_height)
    return (max_area_center_x - rand_width//2,max_area_center_y-rand_height//2),(max_area_center_x + rand_width//2,max_area_center_y + rand_height//2)


def get_roi(img,label_xyxy):
    (x_min, y_min), (x_max, y_max) = label_xyxy
    label_width = x_max - x_min
    label_height = y_max - y_min
    area = label_height*label_width
    roi_img = img[y_min:y_min + label_height,x_min:x_min+label_width,:]
    return roi_img,area

def blur_roi(roi_img,ksize):
    return cv2.GaussianBlur(roi_img,ksize,0)

def set_roi(roi_img,img,label_xyxy):
    (x_min, y_min), (x_max, y_max) = label_xyxy
    label_width = x_max - x_min
    label_height = y_max - y_min
    img[y_min:y_min + label_height, x_min:x_min + label_width, :] = roi_img
    return img
