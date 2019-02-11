import cv2
import csv
import sys, getopt
import scipy
import numpy as np
import keras
import h5py
from os import listdir
from os.path import isfile, join
from keras.utils import np_utils
from keras.optimizers import SGD
from os import walk
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Activation, Dropout, Flatten, Dense, Input

def vgg16_model():

    input_shape = (224, 224, 3)
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])


    #input_shape = (224, 224, 3)
    #img_input = Input(shape=input_shape)
    #include_top = False
    #pooling = None

    ## Block 1
    #x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    #x = Convolution2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    ## Block 2
    #x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    #x = Convolution2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    ## Block 3
    #x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    #x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    #x = Convolution2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    ## Block 4
    #x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    #x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    #x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    ## Block 5
    #x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    #x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    #x = Convolution2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    #x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    #if include_top:
    #    # Classification block
    #    x = Flatten(name='flatten')(x)
    #    x = Dense(4096, activation='relu', name='fc1')(x)
    #    x = Dense(4096, activation='relu', name='fc2')(x)
    #    x = Dense(classes, activation='softmax', name='predictions')(x)
    #else:
    #    if pooling == 'avg':
    #       x = GlobalAveragePooling2D()(x)
    #    elif pooling == 'max':
    #       x = GlobalMaxPooling2D()(x)

    #model = Model(img_input, x, name='glam_n_glow_vgg16')

    return model

def vgg16_predict(train_path, test_path):
    f=[]
    classes=dict()
    batch_size = 16
  
    model = vgg16_model()

    model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])


    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

    model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

    model.save_weights('model_glam_n_glowy.h5') 
    #model.load_weights('model_glam_n_glowy.h5')

    ##https://keras.io/applications/#vgg16
    ##model = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None)
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #model.compile(optimizer=sgd, loss='categorical_crossentropy')
    #
    with open('imagenet-classes.csv', mode='r') as infile:
        reader = csv.reader(infile)
        classes = {rows[0]:rows[1] for rows in reader}

    glam_test_path = test_path + "/glam/"
    glowy_test_path = test_path + "/glowy/"
    glam_files = [f for f in listdir(glam_test_path) if isfile(join(glam_test_path, f))]
    glowy_files = [f for f in listdir(glowy_test_path) if isfile(join(glowy_test_path, f))]
   
    for file in glam_files:
        img_path = glam_test_path + file
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        out = model.predict(x)
        prediction = np.argmax(out)
        
        print( img_path + " : " + str(prediction) )
        #print( img_path + " : " + str(classes[str(prediction)] ) )

    for file in glowy_files:
        img_path = glowy_test_path + file
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        out = model.predict(x)
        print(out)
        prediction = np.argmax(out)
        
        print( img_path + " : " + str(prediction) )
        #print( img_path + " : " + str(classes[str(prediction)] ) )
        
        
def main(argv):
    traindir= sys.argv[1]
    testdir = sys.argv[2]

    vgg16_predict(traindir,testdir)

if __name__ == "__main__":
    main(sys.argv[1:])

