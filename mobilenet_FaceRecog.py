import pandas as pd
import numpy as np
import os,sys
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D,Dropout
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam,SGD
import tensorflow as tf
from tensorflow.keras.regularizers import l2



config =tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.compat.v1.Session(config=config)


device_name = sys.argv[1] 

if device_name == "gpu":
    device_name = "/gpu:1"
else:
    device_name = "/cpu:0"

with tf.device(device_name):

    base_model=MobileNet(weights='imagenet',include_top=False) 

    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(1024,activation='relu')(x)
    x=Dense(1024,activation='relu')(x) 
    x=Dense(512,activation='relu')(x) 
    preds=Dense(12,activation='softmax',kernel_regularizer=l2(0.01))(x)





    model=Model(inputs=base_model.input,outputs=preds)




    for layer in model.layers[:23]:
        layer.trainable=False
    for layer in model.layers[23:]:
        layer.trainable=True



  


    train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest") 
    val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator=train_datagen.flow_from_directory(train_data_dir, 
                                                    target_size=(224,224),
                                                    batch_size=8,
                                                    class_mode='categorical',
                                                    shuffle=True)


    validation_generator=val_datagen.flow_from_directory(validation_data_dir, 
                                                    target_size=(224,224),
                                                    batch_size=8,
                                                    class_mode='categorical',
                                                    shuffle=True)



    model.compile(optimizer="sgd",loss='categorical_crossentropy',metrics=['accuracy'])


    history = model.fit_generator(generator=train_generator, validation_data= validation_generator, use_multiprocessing=False,epochs=10,shuffle=True)

    model.save("mobilenet.h5")


    import matplotlib.pyplot as plt

    plt.subplot(211)  
    plt.plot(history.history['accuracy'])  
    plt.plot(history.history['val_accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  

    # summarize history for loss  

    plt.subplot(212)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'validation'], loc='upper left')  
    plt.savefig('mobilenet22.png')
    plt.show()