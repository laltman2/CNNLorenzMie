import sys
sys.path.append('/home/group/python/')
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import backend as K
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
    return img, input_shape


def rescale(min, max, list):
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
    instrument: Instrument
        Object resprenting the light-scattering instrument
    params_range: dict
        
    Methods
    _______
    load_model(model_path=None)
        loads model.h5 from str 'model_path', updates self.model
    predict(img_names_path=None, img_list=[], save_to_json=False, predictions_path='predictions.json')
        loads img_names.txt from str 'img_names_path', imports images
        img_names.txt contains absolute paths of images, separated by line break
        predicts on list of images using self.model
        saves output to predictions_path if save_to_json==True
    '''
    
    def __init__(self,
                 model_path=None,
                 instrument=None,
                 config_file=None):

        '''
        Parameters
        ----------
        instrument: instrument object
            Object resprenting the light-scattering instrument
        config: json config file from training
        model_path: str
            path to model.h5 file
        '''
        
        ###################################
        # TensorFlow wizardry
        config = tf.compat.v1.ConfigProto()
        
        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True
 
        # Only allow a total of half the GPU memory to be allocated
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
        # Create a session with the above options specified.
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        ###################################
        self.params_range = {}
        if instrument is None:
            self.instrument = Instrument()
        else:
            self.instrument = instrument
        if config_file is None:
            self.params_range['z_p'] = [50, 600]
            self.params_range['a_p'] = [0.2, 5.0]
            self.params_range['n_p'] = [1.38, 2.5]
        else:
            ins = config_file['instrument']
            self.instrument.wavelength = ins['wavelength']
            self.instrument.magnification = ins['magnification']
            self.instrument.n_m = ins['n_m']
            particle = config_file['particle']
            self.params_range['z_p'] = particle['z_p']
            self.params_range['a_p'] = particle['a_p']
            self.params_range['n_p'] = particle['n_p']
        if model_path is None:
            self.model=keras.models.Sequential()
            pix = (None, None)
        else:
            loaded = keras.models.load_model(model_path)
            loaded.summary()
            self.model= loaded
            (a,b,c,d) = loaded.input_shape[0]
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
        
    @property
    def params_range(self):
        return self._params_range

    @params_range.setter
    def params_range(self, params_range):
        self._params_range = params_range

    def load_model(self, model_path=''):
        loaded = keras.models.load_model(model_path)
        loaded.summary()
        self._model = loaded
        return self

    def predict(self, img_names_path=None, img_list=[], scale_list=[]):
        crop_img = img_list
        if not img_names_path is None:
            with open(img_names_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                filename = line.rstrip()
                img_local = np.array(Image.open(filename))
                crop_img.append(img_local)
        crop_img = np.array(crop_img)
        scale_list = np.array(scale_list)
        if crop_img.size == 0: #empty list case
            data = {'z_p': [], 'a_p': [], 'n_p': []}
            return data
        if crop_img.shape[0] != scale_list.shape[0]: #bad input case
            raise Exception('Error: unequal number of images ({}) and scales ({})'.format(crop_img.shape[0], scale_list.shape[0]))
        if crop_img.shape[-1]==3: #if color image, convert to grayscale
            crop_img = crop_img[:,:,:,0]
        crop_img, _ = format_image(crop_img, self.pixels)
        crop_img = crop_img/255.

        
        stamp_model = self.model
        char = stamp_model.predict([crop_img, scale_list])

        zmin, zmax = self.params_range['z_p']
        amin, amax = self.params_range['a_p']
        nmin, nmax = self.params_range['n_p']
        z_pred = rescale_back(zmin, zmax, char[0])
        a_pred = rescale_back(amin, amax, char[1])
        n_pred = rescale_back(nmin, nmax, char[2])
        
        zsave = [item for sublist in z_pred.tolist() for item in sublist]
        asave = [item for sublist in a_pred.tolist() for item in sublist]
        nsave = [item for sublist in n_pred.tolist() for item in sublist]

        data = {'z_p': zsave, 'a_p': asave, 'n_p': nsave}

        return data


if __name__ == '__main__':
    import cv2, json
    keras_model_path = 'keras_models/predict_stamp_best.h5'
    with open('keras_models/predict_stamp_best.json') as f:
        config_json = json.load(f)
    img_path = 'examples/test_image_crop_201.png'
    img = cv2.imread(img_path)
    estimator = Estimator(model_path=keras_model_path, config_file=config_json)
    data = estimator.predict(img_list = [img], scale_list=[1])
    example_z = round(data['z_p'][0],1)
    example_a = round(data['a_p'][0],3)
    example_n = round(data['n_p'][0],3)
    print('Example Image:')
    print('Particle of size {}um with refractive index {} at height {}'.format(example_a, example_n, example_z))
    
