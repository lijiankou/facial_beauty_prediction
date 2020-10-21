import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pandas as pd
import log
import train

def Predict(model_path, img_width, img_height):
    model = load_model(model_path)
    img = load_img("test.png")
    #img = smart_resize(img, size = (img_height, img_width))
    img = img.resize((img_height, img_width))
    img = train.img_to_array(img).reshape(img_height, img_width, channels)
    test = img / 255.0
    test = test.reshape((1,) + test.shape)
    res = model.predict(test)
    logger.info(res)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logger = log.GetLogger(log.logging.INFO)
    img_width, img_height, channels = 350, 350, 3
    input_shape = (img_width, img_height, channels)
    path = "beauty_model"
    Predict(path, img_width, img_height)
