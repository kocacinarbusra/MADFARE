import os,sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import model_from_json
import tensorflow as tf

config =tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.compat.v1.Session(config=config)

device_name = sys.argv[1]  
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):

    img_generator = ImageDataGenerator(rescale=1./255,
            validation_split=0.2)
    traindata = img_generator.flow_from_directory(directory="D:/Face-Mask Detection Graduation Project/maskDataset/3classMaskDataset/train",batch_size=32,target_size=(224,224),subset="training",class_mode="categorical")
    testdata = img_generator.flow_from_directory(directory="D:/Face-Mask Detection Graduation Project/maskDataset/3classMaskDataset/train",batch_size=32,target_size=(224,224),subset="validation",class_mode="categorical")

    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(224, 22)))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(512, activation='relu'))
    cnn.add(tf.keras.layers.Dense(3, activation='softmax'))

    cnn.summary()
    cnn.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=["accuracy"])


    cnn.summary()
    from tensorflow.keras.callbacks import ModelCheckpoint
    checkpoint = ModelCheckpoint("maskDetection.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    hist = cnn.fit_generator(generator=traindata, validation_data= testdata, use_multiprocessing=False,workers=16,epochs=10,callbacks=[checkpoint])

    # summarize history for accuracy  
    import matplotlib.pyplot as plt

    plt.subplot(211)  
    plt.plot(hist.history['accuracy'])  
    plt.plot(hist.history['val_accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    
    # summarize history for loss  
    
    plt.subplot(212)  
    plt.plot(hist.history['loss'])  
    plt.plot(hist.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.savefig('maskDetectionGrayscale.png')
    plt.show()
