import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras
from tensorflow.keras.optimizers import Adam, RMSprop
from util import get_dataflow
from model import MyModel

def evaluate(args):
    print('prepare datagen')
    train_datagen, valid_datagen = get_dataflow(args.dir_path, args.batch_size)
    
    print('getting model')
    mymodel = MyModel()
    
    print('load weight')
    mymodel.model.load_weights(args.model_path)
    
    print('compile model')
    opt = Adam(lr=args.lr)
    mymodel.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(mymodel.model.evaluate_generator(valid_datagen, verbose=1, use_multiprocessing=True, workers=args.workers))
    
if __name__ == '__main__':
    
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--dir_path', help='target training directory path', type=str)
    PARSER.add_argument('--model_path', help='model path', type=str)
    
    PARSER.add_argument('--batch_size', help='batch size before fine tune', default=32, type=int)
    PARSER.add_argument('--lr', help='learning rate before fine tune', default=0.0005, type=float)
    PARSER.add_argument('--workers', help='multiprocessing workers num', default=16, type=int)
    
    ARGS = PARSER.parse_args()
    
    evaluate(ARGS)