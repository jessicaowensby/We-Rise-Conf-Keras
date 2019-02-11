import cv2
import sys, getopt
import scipy
import numpy as np
import keras
from keras.utils import np_utils
import h5py
from keras.optimizers import SGD
from os import walk
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import csv

def vgg16_predict(test_path):
    f=[]
    classes=dict()
   
    # https://keras.io/applications/#vgg16
    model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    
    with open('imagenet-classes.csv', mode='r') as infile:
        reader = csv.reader(infile)
        classes = {rows[0]:rows[1] for rows in reader}
    
    for (dirpath, dirnames, filenames) in walk(test_path):
        f.extend(filenames)
        for file in filenames[1:]:
    
            img_path = test_path+file
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
    
            out = model.predict(x)
            prediction = np.argmax(out)
    
            print( img_path + " : " + str(classes[str(prediction)] ) )

def main(argv):
    testdir = sys.argv[1]

    vgg16_predict(testdir)

if __name__ == "__main__":
    main(sys.argv[1:])

