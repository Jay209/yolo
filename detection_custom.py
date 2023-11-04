#================================================================
#
#   File name   : detection_custom.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================



import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
import time
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.configs import *
#from keras.models import load_model
#from yolov3.dataset import random_horizontal_flip,random_crop, random_translate
image_dir   = "./Dataset/test/test/"
pred_dir   = "./Dataset/predicted/predicted/"
video_path   = "./IMAGES/test.mp4"

yolo = Load_Yolo_model()

#classification_model = load_model(CLASSIFICATION_MODEL)
#image_path = 'C:/Users/jj0276/Desktop/DL/YOLO-Object-Detection/Dataset/test/Inspection02122021_1517572.bmp'
#pred_path = 'C:/Users/jj0276/Desktop/DL/YOLO-Object-Detection/Dataset/predicted/Inspection02122021_1517572.bmp'
start = time.time()
for path in os.listdir(image_dir):
        image_path = image_dir + path
        pred_path = pred_dir + path
        detect_image(yolo, None, image_path, pred_path , input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
print(time.time() - start)
#print(nothing)