import sys
import os
import shutil 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras
from tensorflow.keras.optimizers import Adam, RMSprop
from util import get_dataflow
from model import MyModel
from tqdm import tqdm
from skimage import io, transform
from tensorflow.keras.applications.nasnet import preprocess_input
from preprocess import get_dir_info

def predict(args):
    
    print('getting model')
    mymodel = MyModel()
    
    print('load weight')
    mymodel.model.load_weights(args.model_path)

    raw_img_list = os.listdir(args.src_path)
    
    list_Labels = get_dir_info(args.ref_path)['list_Labels']
    
    for x in tqdm(raw_img_list):

        try:
            img = io.imread(os.path.join(args.src_path, x))
            img = transform.resize(img, [299,299], preserve_range=True)

            if len(img.shape)==2:
                img = color.gray2rgb(img)
            elif len(img.shape)==4:
                img = img[0,...]

            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            img = img[:, :, :, :3]

            score = mymodel.model.predict(img)

            index = np.argmax(score)

            subdir = list_Labels[index]
            
            if not os.path.exists(os.path.join(args.des_path, subdir)):
                os.mkdir(os.path.join(args.des_path, subdir))
            shutil.copyfile(os.path.join(args.src_path, x), os.path.join(args.des_path, subdir, x)) 
        except:
            print(x)
            
if __name__ == '__main__':
    
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--ref_path', help='reference directory path', type=str)
    PARSER.add_argument('--src_path', help='no classed directory path', type=str)
    PARSER.add_argument('--des_path', help='destination directory path', type=str)
    PARSER.add_argument('--model_path', help='model path', type=str)

    ARGS = PARSER.parse_args()
    
    predict(ARGS)
