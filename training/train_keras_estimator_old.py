from __future__ import print_function
import keras, json, shutil, os, cv2, ast
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy as np
from pylorenzmie.utilities.mtd import make_value, make_sample, feature_extent
try:
    from pylorenzmie.theory.CudaLMHologram import CudaLMHologram as LMHologram
except ImportError:
    from pylorenzmie.theory.LMHologram import LMHologram
from pylorenzmie.theory.Instrument import coordinates
from CNNLorenzMie.Estimator import rescale, format_image, rescale_back

'''One-stop training of a new keras model for characterization of stamp-sizes holographic Lorenz-Mie images

Steps to follow:
-edit train_config.json with appropriate params
(make sure you have available disk space)
-Run this file with python or nohup
(dataset generation + training will take at least few hours)
'''

configfile='keras_train_config.json'
with open(configfile, 'r') as f:
    config = json.load(f)

#always only one particle per stamp
config['particle']['nspheres'] = [1,2]
    
'''Make Training Data'''
# set up pipeline for hologram calculation
shape = config['shape']
imgtype = config['imgtype']

#Parameter Info
particle = config['particle']
zmin, zmax = particle['z_p']
amin, amax = particle['a_p']
nmin, nmax = particle['n_p']


def format_json(sample, config, scale=1):
    '''Returns a string of JSON annotations'''
    annotation = []
    for s in sample:
        savestr = s.dumps(sort_keys=True)
        savedict = ast.literal_eval(savestr)
        savedict['scale'] = scale
        savestr = json.dumps(savedict)
        annotation.append(savestr)
    return json.dumps(annotation, indent=4)

def makedata(settype='train/', nframes=10, overwrite=True):
    # create directories and filenames
    directory = os.path.expanduser(config['directory'])+settype
    start = 0
    for dir in ('images', 'params'):
        path = os.path.join(directory, dir)
        if not os.path.exists(path):
            os.makedirs(path)
        already_files = len(os.listdir(path))
        if already_files > start and not overwrite:
            start = already_files
    #start += 1
    shutil.copy2(configfile, directory)
    filetxtname = os.path.join(directory, 'filenames.txt')
    imgname = os.path.join(directory, 'images', 'image{:04d}.' + imgtype)
    jsonname = os.path.join(directory, 'params', 'image{:04d}.json')
    filetxt = open(filetxtname, 'w')
    img_list = []
    scale_list = []
    zlist = []
    alist = []
    nlist = []
    for n in range(start, nframes+start):  # for each frame ...
        print(imgname.format(n))
        sample = make_sample(config) # ... get params for particles
        s = sample[0]
        zlist.append(s.z_p)
        alist.append(s.a_p)
        nlist.append(s.n_p)
        ext = feature_extent(s, config)
        #introduce 1% noise to ext
        ext = np.random.normal(ext, 0.01*ext)
        extsize = ext*2
        shapesize = shape[0]
        if extsize <= shapesize:
            scale = 1
        else:
            scale = int(np.floor(extsize/shapesize) + 1)
        newshape = [i * scale for i in shape]
        holo = LMHologram(coordinates=coordinates(newshape))
        holo.instrument.properties = config['instrument']
        # ... calculate hologram
        frame = np.random.normal(0, config['noise'], newshape)
        if len(sample) > 0:
            holo.particle = sample[0]
            holo.particle.x_p += (scale-1)*100
            holo.particle.y_p += (scale-1)*100
            frame += holo.hologram().reshape(newshape)
        else:
            frame += 1.
        frame = np.clip(100 * frame, 0, 255).astype(np.uint8)
        #decimate
        frame = frame[::scale, ::scale]
        img_list.append(frame)
        scale_list.append(scale)
        
        # ... and save the results
        #do we need?
        cv2.imwrite(imgname.format(n), frame)
        with open(jsonname.format(n), 'w') as fp:
            fp.write(format_json(sample, config, scale))
        filetxt.write(imgname.format(n) + '\n')
        
    img_list = np.array(img_list).astype('float32')
    img_list *= 1./255
    zlist = np.array(zlist).astype('float32')
    zlist = rescale(zmin, zmax, zlist)
    alist = np.array(alist).astype('float32')
    alist = rescale(amin, amax, alist)
    nlist = np.array(nlist).astype('float32')
    nlist = rescale(nmin, nmax, nlist)
    scale_list = np.array(scale_list).astype(int)
    params_list = [zlist, alist, nlist]
    return img_list, params_list, scale_list

def loaddata(settype='train/', nframes=10):
    directory = os.path.expanduser(config['directory'])+settype
    for dir in ('images', 'params'):
        if not os.path.exists(os.path.join(directory, dir)):
            print('No such directory, check your config file')
            break
    imgname = os.path.join(directory, 'images', 'image{:04d}.' + imgtype)
    jsonname = os.path.join(directory, 'params', 'image{:04d}.json')
    img_list = []
    zlist = []
    alist = []
    nlist = []
    scale_list = []
    for n in range(nframes):  # for each frame ...
        with open(jsonname.format(n), 'r') as fp:
            param_string = json.load(fp)[0]
            params = ast.literal_eval(param_string)
        zlist.append(params['z_p'])
        alist.append(params['a_p'])
        nlist.append(params['n_p'])
        scale_list.append(params['scale'])
        localim = cv2.imread(imgname.format(n))[:,:,0]
        img_list.append(localim)
    img_list = np.array(img_list).astype('float32')
    img_list *= 1./255
    zlist = np.array(zlist).astype('float32')
    zlist = rescale(zmin, zmax, zlist)
    alist = np.array(alist).astype('float32')
    alist = rescale(amin, amax, alist)
    nlist = np.array(nlist).astype('float32')
    nlist = rescale(nmin, nmax, nlist)
    scale_list = np.array(scale_list)
    params_list = [zlist, alist, nlist]
    return img_list, params_list, scale_list

    

#File names/numbers
file_header = config['directory']
numtrain = config['nframes_train']
numtest = config['nframes_test']


##Creating##
print('Training set')
img_train, params_train, scale_train = makedata(settype='train/', nframes = numtrain, overwrite = False)
z_train, a_train, n_train = params_train

bloop()
print('Validation set')
img_test, params_test, scale_test = makedata(settype='test/', nframes = numtest)
z_test, a_test, n_test = params_test
'''

##Loading##
print('Training set')
img_train, params_train, scale_train = loaddata(settype='train/', nframes = numtrain)
z_train, a_train, n_train = params_train
print('Validation set')
img_test, params_test, scale_test = loaddata(settype='test/', nframes = numtest)
z_test, a_test, n_test = params_test
'''

#Image dimensions
img_rows, img_cols = shape

img_train, input_shape = format_image(img_train, shape)
img_test, _ = format_image(img_test, shape)


#Training Parameters
batch_size = config['training']['batchsize']
epochs = config['training']['epochs']
save_file = str(config['training']['savefile'])

def multioutput_model():
    model_input_img = keras.Input(shape=input_shape, name='image')
    model_input_scale = keras.Input(shape=(1,), name='scale')
    model_inputs = [model_input_img, model_input_scale]
    x = model_input_img
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (2,2))(x)
    x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size = (4,4))(x)
    conv_out = Flatten()(x)
    reg = 0.01
    x = concatenate([model_input_scale, conv_out])
    x = Dense(20, activation='relu', kernel_regularizer = regularizers.l2(reg))(x)
    model_outputs = list()
    out_names = ['z', 'a', 'n']
    drop_rates = [0.005, 0.005, 0.005]
    regularizer_rates = [0.3, 0.3, 0.3]
    dense_nodes = [20, 40, 100]
    dense_layers = [1,1,1]
    loss_weights = []
    for i in range(3):
        out_name = out_names[i]
        #drop_rate = drop_rates[i]
        #reg = regularizer_rates[i]
        dense_node = dense_nodes[i]
        drop_rate = 0.001
        local_output= x
        for numlayer in range(dense_layers[i]):
            local_output = Dense(dense_node, activation='relu', kernel_regularizer = regularizers.l2(reg))(local_output)
            local_output = Dropout(drop_rate)(local_output)
        local_output = Dense(units=1, activation='linear', name = out_name)(local_output)
        model_outputs.append(local_output)
        loss_weights.append(1)

    Adamlr = keras.optimizers.Adam(lr=0.001)
    model = Model(model_inputs, model_outputs)
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


estimator.fit({'image' : img_train, 'scale': scale_train},
              {'z' : z_train, 'a' : a_train,
               'n': n_train},
              batch_size = batch_size,
              epochs=epochs,
              verbose=1,
              validation_data = ({'image' : img_test, 'scale': scale_test},
                                 {'z' : z_test, 'a' : a_test,
                                  'n': n_test}))
              #callbacks=callbacks)


print('Finished training')


save_keras = os.path.expanduser(save_file+'.h5')
save_json = os.path.expanduser(save_file+'.json')

estimator.save(save_keras)
print('Saved keras model')

#make eval dataset
numeval=config['nframes_eval']
img_eval, params_eval, scale_eval = makedata(settype='eval/', nframes = numeval)
z_eval, a_eval, n_eval = params_eval
img_eval, _ = format_image(img_eval, shape)


char_pred = estimator.predict([img_eval, scale_eval])
z_pred = char_pred[0]
a_pred = char_pred[1]
n_pred = char_pred[2]

def RMSE(gtru, pred):
    numimg = len(gtru)
    diff = np.zeros(numimg)
    for i in range(numimg):
        diff[i] = gtru[i] - pred[i]
    sqer = np.zeros(numimg)
    for i in range(numimg):
        sqer[i] = (diff[i])**2
    SST = np.sum(sqer)
    SST *= 1./numimg
    RMSE = np.sqrt(SST)
    return RMSE

z_RMSE = RMSE(z_eval, z_pred)
a_RMSE = RMSE(a_eval, a_pred)
n_RMSE = RMSE(n_eval, n_pred)


save_conf = config.copy()
del save_conf['directory']
del save_conf['imgtype']
del save_conf['delete_files_after_training']
save_conf['z_RMSE'] = z_RMSE
save_conf['a_RMSE'] = a_RMSE
save_conf['n_RMSE'] = n_RMSE

with open(save_json, 'w') as f:
    json.dump(save_conf, f)
print('Saved config')

if config['delete_files_after_training']:
    head_dir = os.path.expanduser(config['directory'])
    shutil.rmtree(head_dir)
