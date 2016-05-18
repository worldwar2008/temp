# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import permutation

np.random.seed(2016)

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss
import h5py
from keras.optimizers import SGD, Adadelta

pp = '../'
color_type = 3
mean_pixel = [103.939, 116.779, 123.68]

def get_im(path):
    """图片像素调整,采用三通道的数据"""
    # Load as grayscale
    img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (224, 224))
    # resized = resized.transpose((2,0,1))
    # resized = np.expand_dims(resized, axis=0)
    return resized


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # 注释掉全连接层,因为我们不需要这一层,后面的权重系数导入的时候,也是避开了这一层
    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    # if weights_path:
    #     model.load_weights(weights_path)
    assert os.path.exists(weights_path), "Model weights file not found (see 'weights_path' variable in script)"
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        print "k", k
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        print "全连接层之前的：", k
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        model.layers[k].trainable = False

    f.close()
    print 'model loaded.'
    return model


def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        # 路径名
        path = os.path.join(pp + 'rawdata/train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_train.append(img)
            y_train.append(j)

    return X_train, y_train


def load_test():
    print('Read test images')
    path = os.path.join(pp + 'rawdata/test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    # 小于目标值的最大整数的浮点数
    # 分批读取测试文件,因为test数据量特别大
    thr = math.floor(len(files) / 10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir(pp + 'cache'):
        os.mkdir(pp + 'cache')
    open(os.path.join(pp + 'cache', 'architecture.json'), 'w').write(json_string)
    model.save_weights(os.path.join(pp + 'cache', 'model_weights.h5'), overwrite=True)


def read_model():
    model = model_from_json(open(os.path.join(pp + 'cache', 'architecture.json')).read())
    model.load_weights(os.path.join(pp + 'cache', 'model_weights.h5'))
    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def split_validation_set_with_hold_out(train, target, test_size):
    random_state = 51
    train, X_test, target, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    X_train, X_holdout, y_train, y_holdout = train_test_split(train, target, test_size=test_size,
                                                              random_state=random_state)
    return X_train, X_test, X_holdout, y_train, y_test, y_holdout


def create_submission(predictions, test_id, loss):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = str(round(loss, 6)) + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


# The same as log_loss
def mlogloss(target, pred):
    score = 0.0
    for i in range(len(pred)):
        pp = pred[i]
        for j in range(len(pp)):
            prob = pp[j]
            if prob < 1e-15:
                prob = 1e-15
            score += target[i][j] * math.log(prob)
    return -score / len(pred)


def validate_holdout(model, holdout, target):
    predictions = model.predict(holdout, batch_size=128, verbose=1)
    score = log_loss(target, predictions)
    print('Score log_loss: ', score)
    # score = model.evaluate(holdout, target, show_accuracy=True, verbose=0)
    # print('Score holdout: ', score)
    # score = mlogloss(target, predictions)
    # print('Score : mlogloss', score)
    return score


cache_path = os.path.join(pp + 'cache', 'train-3.dat')

if not os.path.isfile(cache_path):
    train_data, train_target = load_train()
    cache_data((train_data, train_target), cache_path)
else:
    print('Restore train from cache!')
    (train_data, train_target) = restore_data(cache_path)

batch_size = 64
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 224, 224
# number of convolutional filters to use
nb_filters = 32

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_conv = 3

train_data = np.array(train_data, dtype=np.uint8)
train_target = np.array(train_target, dtype=np.uint8)
if color_type == 1:
    train_data = train_data.reshape(train_data.shape[0], color_type, img_rows, img_cols)
elif color_type == 3:
    train_data = train_data.transpose((0, 3, 1, 2))
train_target = np_utils.to_categorical(train_target, nb_classes)
train_data = train_data.astype('float32')

# for c in range(3):
#     train_data[:, c, :, :] = train_data[:, c, :, :] - mean_pixel[c]
train_data /= 255

perm = permutation(len(train_target))
train_data = train_data[perm]

print('Train shape:', train_data.shape)
print(train_data.shape[0], 'train samples')

X_train, X_test, X_holdout, Y_train, Y_test, Y_holdout = split_validation_set_with_hold_out(train_data, train_target,
                                                                                            0.2)
print('Split train: ', len(X_train))
print('Split valid: ', len(X_test))
print('Split holdout: ', len(X_holdout))

model_from_cache = 0
if model_from_cache == 1:
    model = read_model()
    # 开始训练模型,loss代表损失函数(目标函数)
    adadelta = Adadelta(lr=0.001, rho=0.95, epsilon=1e-6)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=["accuracy"])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    '''
    model.fit(train_data, train_target, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_split=0.1)
    '''
    # 调用fit方法就是一个模型的训练的过程.show_accuracy=True, 训练时每一个epoch都输出accuracy.
    # validation_data是验证集
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))
else:
    model = VGG_16(pp + "model-zoo/vgg/vgg16_weights.h5")
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    # 开始训练模型,loss代表损失函数(目标函数)
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    # adadelta = Adadelta(lr=0.001, rho=0.95, epsilon=1e-6)
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=["accuracy"])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    '''
    model.fit(train_data, train_target, batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=1, validation_split=0.1)
    '''
    # 调用fit方法就是一个模型的训练的过程.show_accuracy=True, 训练时每一个epoch都输出accuracy.
    # validation_data是验证集
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
print('Score: ', score)
score = model.evaluate(X_holdout, Y_holdout, verbose=0)
print('Score holdout: ', score)
validate_holdout(model, X_holdout, Y_holdout)
save_model(model)

cache_path = os.path.join(pp + 'cache', 'test-3.dat')
if not os.path.isfile(cache_path):
    test_data, test_id = load_test()
    cache_data((test_data, test_id), cache_path)
else:
    print('Restore test from cache!')
    (test_data, test_id) = restore_data(cache_path)

test_data = np.array(test_data, dtype=np.uint8)
if color_type == 1:
    test_data = test_data.reshape(test_data.shape[0], color_type, img_rows, img_cols)
elif color_type == 3:
    test_data = test_data.transpose((0, 3, 1, 2))
test_data = test_data.astype('float32')

test_data /= 255

# for c in range(3):
#     test_data[:, c, :, :] = test_data[:, c, :, :] - mean_pixel[c]

print('Test shape:', test_data.shape)
print(test_data.shape[0], 'test samples')
predictions = model.predict(test_data, batch_size=128, verbose=1)

create_submission(predictions, test_id, score[0])
