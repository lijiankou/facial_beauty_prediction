from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense
import os
import numpy as np
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import pandas as pd
import log
def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x



def GetLabel(num):
    ratings = pd.read_excel('data/SCUT-FBP5500/All_Ratings.xlsx')
    filenames = ratings.groupby('Filename').size().index.tolist()
    labels = []
    for index,filename in enumerate(filenames):
        if index > num:
            break
        df = ratings[ratings['Filename'] == filename]
        score = round(df['Rating'].mean(), 2)
        labels.append({'Filename': filename, 'score': score})
    labels_df = pd.DataFrame(labels)
    return labels_df

def GetData(num, input_shape, label_df):
    sample_dir = 'data/SCUT-FBP5500/Images/'
    nb_samples = len(os.listdir(sample_dir))

    feat = np.empty((nb_samples, img_width, img_height, channels), dtype=np.float32)
    #label = np.empty((nb_samples, 1), dtype=np.float32)
    label = np.empty((nb_samples, 1), dtype=np.float32)

    for i, fn in enumerate(os.listdir(sample_dir)):
        if i > num:
            break
        img = load_img('%s/%s' % (sample_dir, fn))
        x = img_to_array(img).reshape(img_height, img_width, channels)
        x = x.astype('float32') / 255.
        y = label_df[label_df.Filename == fn].score.values
        y = y.astype('float32')
        feat[i] = x
        if len(y) == 0:
            label[i] = 0
        else:
            label[i] = y

    seed = 42
    x_train, x_test, y_train, y_test = train_test_split(feat, label, test_size=0.2, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
    return x_train,x_val, x_test, y_train,y_val,y_test

def GetModel(input_shape):
    resnet = ResNet50(include_top=False, pooling='avg', input_shape=input_shape)
    model = Sequential()
    model.add(resnet)
    model.add(Dense(1))
    model.layers[0].trainable = False
    model.compile(loss='mse', optimizer='adam')
    return model

if __name__ == '__main__':
    logger = log.GetLogger(log.logging.INFO)
    logger.warn('hello world')
    img_width, img_height, channels = 350, 350, 3
    input_shape = (img_width, img_height, channels)
    num = 100
    label_df = GetLabel(num)
    print(label_df)
    x_train,x_val,x_test,y_train,y_val,y_test = GetData(num, input_shape,label_df)
    logger.info(x_train)
    logger.info(y_train)
    model = GetModel(input_shape)
    #model.fit(batch_size=32, x=x_train, y=y_train, epochs=30)
    model.fit(batch_size=128, x=x_train, y=y_train, epochs=10)
