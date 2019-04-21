from __future__ import division
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Activation, MaxPool2D, BatchNormalization
from img_load import img_load
import numpy as np

train_path = 'train'
test_path = 'test'

train_x, train_label = img_load(train_path)
test_x, test_label = img_load(test_path)

print train_x.shape, train_label.shape
print test_x.shape, test_label.shape

def alex():
    model = Sequential()

    model.add(Convolution2D(96, (5,5), input_shape=(64, 64, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3, 3)))
    model.add(BatchNormalization())

    model.add(Convolution2D(256, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3, 3), strides=2))
    model.add(BatchNormalization())

    model.add(Convolution2D(384, (3, 3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(384, (3, 3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(3,3), strides=2))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(879))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def cnn_model2():
    model = Sequential()
    #model.add(Convolution2D(16, (5, 5), activation='relu', input_shape=[64, 64, 3]))
    #model.add(Convolution2D(16, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=[64, 64, 3]))
    model.add(Convolution2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(64, (2, 2), activation='relu'))
    model.add(Convolution2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(879, activation='softmax'))
    for layer in model.layers:
        print layer.output_shape

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def cnn_model1():
    model = Sequential()
    model.add(Convolution2D(32, (1, 1), activation='relu', input_shape=[64, 64, 3]))
    model.add(Convolution2D(32, (1, 1), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(64, (1, 1), activation='relu'))
    model.add(Convolution2D(64, (1, 1), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(879, activation='softmax'))
    for layer in model.layers:
        print layer.output_shape

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model):
    model.fit(train_x, train_label,
              batch_size=64,
              epochs=50,
              validation_data=(test_x, test_label),
              shuffle=True)
    model.save_weights('2.h5')

def cal_precision(model, test_x, test_label):
    model.load_weights('2.h5')
    test_pred = model.predict(test_x)
    pred = label_2d_to_1d(test_pred)
    label = label_2d_to_1d(test_label)
    print pred
    print label
    f = open('precision.txt', 'w+')
    for i in range(1, 880):

        pred_index = np.where(pred == i)
        if i==1:
            print pred_index
        total_num = pred_index[0].shape[0]
        #print total_num
        precision_index = (label[pred_index] == pred[pred_index]).astype('int')
        prec_num = sum(precision_index)
        #print prec_num
        if prec_num==0 | total_num==0:
            precision = 0
        else:
            precision = prec_num / total_num
        f.write(str(i)+':'+str(precision)+'\n')
    f.close()


def label_2d_to_1d(labels):
    print labels
    labels_1d = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        labels_1d[i] = np.argmax(labels[i]) + 1
    return labels_1d

#model = cnn_model2()
#train_model(model)
#cal_precision(model,test_x, test_label)