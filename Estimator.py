import numpy as np
import json, keras
import re,os,sys
from matplotlib import pyplot as plt
from PIL import Image
from keras import backend as K
sys.path.append('/home/group/lauren_yolo/')
from pylorenzmie.theory.Instrument import Instrument
import tensorflow as tf

def format_image(img, crop_px):
    (crop_img_rows, crop_img_cols) = crop_px
    if K.image_data_format() == 'channels_first':
        img = img.reshape(img.shape[0], 1, crop_img_rows, crop_img_cols)
        input_shape = (1, crop_img_rows, crop_img_cols)
    else:
        img = img.reshape(img.shape[0], crop_img_rows, crop_img_cols, 1)
        input_shape = (crop_img_rows, crop_img_cols, 1)
    return(img)


def rescale(min, max, target, list):
    scalar = 1./(max-min)
    list = (list-min)*scalar
    return list

def rescale_back(min, max, list):
    scalar = (max-min)/1.
    list = list*scalar + min
    return list


class Estimator(object):

    '''
    Attributes
    __________
    model: Model
        Keras model with input: image, output: z_p, a_p, n_p
        image is np.ndarray of images with size=pixels, formatted with keras backend
        outputs are np.ndarray of floats
    pixels: tuple                    #coordinates instead?
        (img_rows, img_cols)
    instrument: Instrument
        Object resprenting the light-scattering instrument
        
    Methods
    _______
    load_model(model_path=None)
        loads model.h5 from str 'model_path', updates self.model
    predict(img_names_path=None, img_list=[], save_to_json=False, predictions_path='predictions.json')
        loads img_names.txt from str 'img_names_path', imports images
        img_names.txt contains absolute paths of images, separated by line break
        predicts on list of images using self.model
        saves output to predictions_path if save_to_json==True

    ////UNDER CONSTRUCTION////
    create_model(self)
        instantiates a new keras Model, updates self.model
    train(epochs=100, batch_size=64, train_img_path=None, test_img_path=None)
        trains model using images given
    '''
    
    def __init__(self,
                 model_path=None,
                 instrument=None):

        '''
        Parameters
        ----------
        pixels: tuple                    #coordinates instead?                                           
            (img_rows, img_cols)
        instrument: Instrument  
            Object resprenting the light-scattering instrument
        model_path: str
            path to model.h5 file
        '''
        
        ###################################
        # TensorFlow wizardry
        config = tf.ConfigProto()
        
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
 
        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
        # Create a session with the above options specified.
        K.tensorflow_backend.set_session(tf.Session(config=config))
        ###################################

        if instrument is None:
            self.instrument = Instrument()
        else:
            self.instrument = instrument
        if model_path is None:
            self.model=keras.models.Sequential()
            pix = (None, None)
        else:
            loaded = keras.models.load_model(model_path)
            loaded.summary()
            self.model= loaded
            (a,b,c,d) = loaded.input_shape
            pix = (b,c)
        self.pixels = pix


    @property
    def pixels(self):
        return self._pixels

    @pixels.setter
    def pixels(self, pixels):
        self._pixels = pixels
    
    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def load_model(self, model_path=''):
        loaded = keras.models.load_model(model_path)
        loaded.summary()
        self._model = loaded
        return self

    def predict(self, img_names_path=None, img_list=[], save_to_json=False, predictions_path='predictions.json'):
        crop_img = img_list
        if not img_names_path is None:
            with open(img_names_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                filename = line.rstrip()
                img_local = np.array(Image.open(filename))
                crop_img.append(img_local)
        crop_img = np.array(crop_img)
        crop_img = format_image(crop_img, self.pixels)
        crop_img = crop_img/255

        stamp_model = self.model
        char = stamp_model.predict(crop_img)
        z_pred = rescale_back(50, 600, char[0])
        a_pred = rescale_back(0.2, 5., char[1])
        n_pred = rescale_back(1.38, 2.5, char[2])
        
        zsave = [item for sublist in z_pred.tolist() for item in sublist]
        asave = [item for sublist in a_pred.tolist() for item in sublist]
        nsave = [item for sublist in n_pred.tolist() for item in sublist]

        #instrument_params = self.instrument.####
    
        data = {'z_p': zsave, 'a_p': asave, 'n_p': nsave}#, 'instrument':instrument_params}

        if save_to_json:
            with open(predictions_path, 'w') as outfile:
                json.dump(data, outfile)
        return data


if __name__ == '__main__':
    keras_model_path = '/home/group/lauren_yolo/Holographic-Characterization/models/predict_lab_stamp_pylm_800.h5'
    cropdir = '/home/group/endtoend/cropped_img/'
    img_filepath = cropdir+'filenames.txt'
    #predictions_json = '/home/group/endtoend/ML_predictions.json'       
    crop_img_rows, crop_img_cols = 200, 200
    pix = (crop_img_rows, crop_img_cols)
    
    estimator = Estimator(model_path=keras_model_path)
    data = estimator.predict(img_names_path = img_filepath)
    example_z = round(data['z_p'][0],1)
    example_a = round(data['a_p'][0],3)
    example_n = round(data['n_p'][0],3)
    print('Image 1:')
    print('Particle of size {}um with refractive index {} at height {}'.format(example_a, example_n, example_z))
    
