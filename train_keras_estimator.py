from __future__ import print_function
import keras, json, shutil, os, cv2
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy as np
from PIL import Image
from pylorenzmie.utilities.mtd import format_json, make_value, make_sample
try:
    from pylorenzmie.theory.CudaLMHologram import CudaLMHologram as LMHologram
except ImportError:
    from pylorenzmie.theory.LMHologram import LMHologram
from pylorenzmie.theory.Instrument import coordinates
from pylorenzmie.theory.Sphere import Sphere
from Estimator import rescale, format_image

'''One-stop training of a new keras model for characterization of stamp-sizes holographic Lorenz-Mie images

Steps to follow:
-edit train_config.json with appropriate params
(make sure you have available disk space)
-Run this file with python or nohup
(dataset generation + training will take at least few hours)
'''


configfile='train_config.json'
with open(configfile, 'r') as f:
    config = json.load(f)

#always only one particle per stamp
config['particle']['nspheres'] = [1,2]
    
'''Make Training Data'''
# set up pipeline for hologram calculation
shape = config['shape']
holo = LMHologram(coordinates=coordinates(shape))
holo.instrument.properties = config['instrument']
imgtype = config['imgtype']

#Parameter Info
particle = config['particle']
zmin, zmax = particle['z_p']
amin, amax = particle['a_p']
nmin, nmax = particle['n_p']


def makedata(settype='train/', nframes=10):
    # create directories and filenames
    directory = os.path.expanduser(config['directory'])+settype
    for dir in ('images', 'params'):
        if not os.path.exists(os.path.join(directory, dir)):
            os.makedirs(os.path.join(directory, dir))
    shutil.copy2(configfile, directory)
    filetxtname = os.path.join(directory, 'filenames.txt')
    imgname = os.path.join(directory, 'images', 'image{:04d}.' + imgtype)
    jsonname = os.path.join(directory, 'params', 'image{:04d}.json')
    filetxt = open(filetxtname, 'w')
    img_list = []
    zlist = []
    alist = []
    nlist = []
    for n in range(nframes):  # for each frame ...
        print(imgname.format(n))
        sample = make_sample(config) # ... get params for particles
        s = sample[0]
        zlist.append(s.z_p)
        alist.append(s.a_p)
        nlist.append(s.n_p)
        # ... calculate hologram
        frame = np.random.normal(0, config['noise'], shape)
        if len(sample) > 0:
            holo.particle = sample
            frame += holo.hologram().reshape(shape)
        else:
            frame += 1.
        frame = np.clip(100 * frame, 0, 255).astype(np.uint8)
        img_list.append(frame)
        
        # ... and save the results
        #do we need?
        cv2.imwrite(imgname.format(n), frame)
        with open(jsonname.format(n), 'w') as fp:
            fp.write(format_json(sample, config))
        filetxt.write(imgname.format(n) + '\n')
        
    img_list = np.array(img_list).astype('float32')
    img_list *= 1./255
    zlist = np.array(zlist).astype('float32')
    zlist = rescale(zmin, zmax, zlist)
    alist = np.array(alist).astype('float32')
    alist = rescale(amin, amax, alist)
    nlist = np.array(nlist).astype('float32')
    nlist = rescale(nmin, nmax, nlist)
    params_list = [zlist, alist, nlist]
    return img_list, params_list


#File names/numbers
file_header = config['directory']
numtrain = config['nframes_train']
numtest = config['nframes_test']


print('Training set')
img_train, params_train = makedata(settype='train/', nframes = numtrain)
z_train, a_train, n_train = params_train
print('Validation set')
img_test, params_test = makedata(settype='test/', nframes = numtest)
z_test, a_test, n_test = params_test


#Image dimensions
img_rows, img_cols = shape

img_train, input_shape = format_image(img_train, shape)
img_test, _ = format_image(img_test, shape)


#Training Parameters
batch_size = config['training']['batchsize']
epochs = config['training']['epochs']
save_file = config['training']['savefile']

save_keras = save_file+'.h5'
save_json = save_file+'.json'

with open(save_json, 'w') as f:
    json.dump(config, f)

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


print('Finished training')
estimator.save(save_file)
print('Saved keras model')

if config['delete_files_after_training']:
    head_dir = os.path.expanduser(config['directory'])
    shutil.rmtree(head_dir)