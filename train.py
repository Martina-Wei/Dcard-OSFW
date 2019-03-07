import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from PIL import Image
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras
from tensorflow.keras.optimizers import Adam, RMSprop
from util import get_dataflow
from model import MyModel
from tensorflow.keras.callbacks import ModelCheckpoint


def train(args):
    
    if not args.skip_step_1:
        print('prepare datagen before fine tune')
        train_datagen, valid_datagen = get_dataflow(args.dir_path, args.batch_size_1)

        print('getting model')
        mymodel = MyModel()

        print('fix base model')
        for layer in mymodel.base_model.layers:
            layer.trainable = False

        print('compile model')
        opt = Adam(lr=args.lr_1)
        mckpt = ModelCheckpoint('model_fix_base.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
        mymodel.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        mymodel.model.fit_generator(train_datagen, 
                        epochs=args.epochs_1, 
                        class_weight='auto',
                        validation_data=valid_datagen, 
                        use_multiprocessing=True, 
                        callbacks=[mckpt],
                        workers=args.workers)
    
    
    print('evaluate best model before fine tune')
    mymodel.model.load_weights('model_fix_base.h5')
    print(mymodel.model.evaluate_generator(valid_datagen, verbose=1, use_multiprocessing=True, workers=args.workers))
    
    print('prepare datagen at fine tune')
    train_datagen, valid_datagen = get_dataflow(args.dir_path, args.batch_size_2)
    
    print('unlock base model')
    for layer in mymodel.base_model.layers:
        layer.trainable = True
    
    print('compile model')
    opt = Adam(lr=args.lr_2)
    mymodel.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    mckpt2 = ModelCheckpoint('model_final.h5', monitor='val_acc', save_best_only=True, save_weights_only=True)
    
    mymodel.model.fit_generator(train_datagen, 
                    epochs=args.epochs_2, 
                    class_weight='auto',
                    validation_data=valid_datagen, 
                    use_multiprocessing=True, 
                    callbacks=[mckpt2],
                    workers=args.workers)
    
    print('evaluate final best model')
    mymodel.model.load_weights('model_final.h5')
    print(mymodel.model.evaluate_generator(valid_datagen, verbose=1, use_multiprocessing=True, workers=args.workers))
    
if __name__ == '__main__':
    
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--dir_path', help='target training directory path', type=str)
    
    PARSER.add_argument('--batch_size_1', help='batch size before fine tune', default=32, type=int)
    PARSER.add_argument('--batch_size_2', help='batch size at fine tune', default=6, type=int)
    PARSER.add_argument('--lr_1', help='learning rate before fine tune', default=0.0005, type=float)
    PARSER.add_argument('--lr_2', help='learning rate at fine tune', default=0.00002, type=float)
    PARSER.add_argument('--epochs_1', help='epochs before fine tune', default=5, type=int)
    PARSER.add_argument('--epochs_2', help='epochs at fine tune', default=20, type=int)
    PARSER.add_argument('--workers', help='multiprocessing workers num', default=16, type=int)
    PARSER.add_argument('--skip_step_1', help='skip step 1', default=False, type=bool)
    
    ARGS = PARSER.parse_args()
    
    train(ARGS)