import os
import numpy as np
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import get_dir_info

def pre(x):
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x[0]

def get_dataflow(dir_path, batch_size):
    
    train_path = os.path.join(dir_path, 'train/')
    valid_path = os.path.join(dir_path, 'valid/')
    print(train_path, valid_path)
    
    list_Labels = get_dir_info(train_path)['list_Labels']
    
    train_datagen = ImageDataGenerator(
    #     rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        preprocessing_function=pre
    )

    valid_datagen = ImageDataGenerator(
    #     rescale=1./255,
        preprocessing_function=pre
    )
    
    training_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(299, 299),
        color_mode='rgb',
        classes=list_Labels,
        batch_size=batch_size,
        class_mode='categorical')

    valid_generator = valid_datagen.flow_from_directory(
        valid_path,
        target_size=(299, 299),
        color_mode='rgb',
        classes=list_Labels,
        batch_size=batch_size,
        class_mode='categorical')
    
    return training_generator, valid_generator