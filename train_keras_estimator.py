from __future__ import print_function
import keras, json, numpy, ast
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import backend as K
import tensorflow as tf
from PIL import Image

'''One-stop training of a new keras model for characterization of stamp-sizes holographic Lorenz-Mie images

Steps to follow:
-Create your dataset using pylorenzmie.utilities.mtd
-Place your test/train datasets under the same header directory
     example: train_path = datasets/train, test_path = datasets/test
-Update image/data parameters and training parameters in the lines below
-Run this file with python or nohup
'''

#Image dimensions

img_rows, img_cols = 201,201


#Parameter Info

zmin, zmax = 50, 600
amin, amax = 0.2, 5.0
nmin, nmax = 1.38, 2.5


#File names/numbers

file_header = '../datasets/pylm-final/'
numtrain = 10000
numtest = 1000


#Training Parameters

batch_size = 64
epochs = 200
seed=7


#Savefile

save_file = 'keras_models/predict_stamp.h5'


print('Opening data...')

img_file_middle = 'images_labels/image'
param_file_middle = 'params/image'


print('Training set')
img_train = []
z_train = []
a_train = []
n_train = []
for i in range(numtrain):
    img_file = file_header + 'train/' + img_file_middle + str(i).zfill(4) + '.png'
    param_file = file_header + 'train/' + param_file_middle + str(i).zfill(4) + '.json'
    im = Image.open(img_file)
    im_np = numpy.array(im)
    img_train.append(im_np)
    im.close()
    with open(param_file, 'r') as paramfile:
        params = json.load(paramfile)
    if len(params) != 1:
        print('more or less than one hologram in image', str(i).zfill(4))
    else:
        params = ast.literal_eval(params[0])
        z_train.append(params['z_p'])
        a_train.append(params['a_p'])
        n_train.append(params['n_p'])


img_train = numpy.array(img_train).astype('float32')
z_train = numpy.array(z_train).astype('float32')
a_train = numpy.array(a_train).astype('float32')
n_train = numpy.array(n_train).astype('float32')

print('Test set')
img_test = []
z_test = []
a_test = []
n_test = []
for i in range(numtest):
    img_file = file_header + 'test/' + img_file_middle + str(i).zfill(4) + '.png'
    param_file = file_header + 'test/' + param_file_middle + str(i).zfill(4) + '.json'
    im = Image.open(img_file)
    im_np = numpy.array(im)
    img_test.append(im_np)
    im.close()
    with open(param_file, 'r') as paramfile:
        params = json.load(paramfile)
    if len(params) != 1:
        print('more or less than one hologram in image', str(i).zfill(4))
    else:
        params = ast.literal_eval(params[0])
        z_test.append(params['z_p'])
        a_test.append(params['a_p'])
        n_test.append(params['n_p'])


img_test = numpy.array(img_test).astype('float32')
z_test = numpy.array(z_test).astype('float32')
a_test = numpy.array(a_test).astype('float32')
n_test = numpy.array(n_test).astype('float32')

img_train *= 1./255
img_test *= 1./255

#format data

print('Formatting...')
if K.image_data_format() == 'channels_first':
    img_train = img_train.reshape(img_train.shape[0], 1, img_rows, img_cols)
    img_test = img_test.reshape(img_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    img_train = img_train.reshape(img_train.shape[0], img_rows, img_cols, 1)
    img_test = img_test.reshape(img_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


#Data normalization
def rescale(min, max, list):
    scalar=1./(max-min)
    list = (list- min)*scalar
    return list

print('Rescaling target data...')
z_train = rescale(zmin, zmax, z_train)
z_test = rescale(zmin, zmax, z_test)

a_train = rescale(amin, amax, a_train)
a_test = rescale(amin, amax, a_test)

n_train = rescale(nmin, nmax, n_train)
n_test = rescale(nmin, nmax, n_test)


def multioutput_model():    
    model_input = keras.Input(shape=input_shape, name='image')
    x = model_input
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (4,4))(x)
    x = Flatten()(x)
    reg = 0.01
    x = Dense(20, activation='relu', kernel_regularizer = regularizers.l2(reg))(x)
    model_outputs = list()
    out_names = ['z', 'a', 'n']
    drop_rates = [0.01, 0.01, 0.2]
    regularizer_rates = [0.3, 0.3, 0.3]
    dense_nodes = [40, 20, 100]
    loss_weights = []
    for i in range(3):
        out_name = out_names[i]
        #drop_rate = drop_rates[i]
        #reg = regularizer_rates[i]
        dense_node = dense_nodes[i]
        drop_rate = 0.001
        local_output= x
        local_output = Dense(dense_node, activation='relu', kernel_regularizer = regularizers.l2(reg))(local_output)
        local_output = Dropout(drop_rate)(local_output)
        local_output = Dense(units=1, activation='linear', name = out_name)(local_output)
        model_outputs.append(local_output)
        loss_weights.append(1)

    Adamlr = keras.optimizers.Adam(lr=0.001)
    model = Model(model_input, model_outputs)
    model.compile(loss = 'mean_squared_error', optimizer='rmsprop', loss_weights = loss_weights)
    model.summary()
    return model
    

#Callbacks
callbacks = []

tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)
earlystopCB = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=2000, patience=10, verbose=0, mode='auto', baseline=None)

callbacks.append(tbCallBack)
#callbacks.append(earlystopCB)

estimator = multioutput_model()


estimator.fit({'image' : img_train},
              {'z' : z_train, 'a' : a_train,
               'n': n_train},
              batch_size = batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = ({'image' : img_test},
                                 {'z' : z_test, 'a' : a_test,
                                  'n': n_test}),
              callbacks=callbacks)

estimator.save(save_file)
