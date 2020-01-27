from __future__ import print_function
import keras, json
from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy as np
from CNNLorenzMie.Estimator import format_image


configfile='keras_train_config.json'
with open(configfile, 'r') as f:
    config = json.load(f)


#Image dimensions
_, input_shape = format_image(np.array([]), config['shape'])


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
def callbacks():
    callbacks = []

    tbCallBack = keras.callbacks.TensorBoard(log_dir='../Graph', histogram_freq=0, write_graph=True, write_images=True)
    earlystopCB = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=2000, patience=10, verbose=0, mode='auto', baseline=None)

    callbacks.append(tbCallBack)
    callbacks.append(earlystopCB)
    return callbacks



