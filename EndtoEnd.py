import numpy as np
import json, keras
import re,os,sys
import warnings
from matplotlib import pyplot as plt
from PIL import Image
from keras import backend as K
from pylorenzmie.theory.Instrument import Instrument, coordinates
from Estimator import Estimator
from Localizer import Localizer
from crop_feature import crop_feature
from pylorenzmie.theory.Feature import Feature
from lmfit import report_fit

class EndtoEnd(object):

    '''
    Attributes
    __________
    localizer: Localizer
        Object resprenting the trained YOLO model
    estimator: Estimator
        Object representing the trained Keras model
    instrument: Instrument
        Object resprenting the light-scattering instrument
        
    Methods
    _______
    predict(img_names_path=None, img_list=[], save_predictions=False, predictions_path='predictions.json', save_crops=False, crop_dir='./cropped_img')
        loads img_names.txt from str 'img_names_path', imports images
        img_names.txt contains absolute paths of images, separated by line break
        or, just input images as a list
        predicts on list of images using self.model
        saves output to predictions_path if save_predictions = True
        saves cropped images to crop_dir if save_crops = True
    '''
    
    def __init__(self,
                 localizer=None,
                 estimator=None):

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
        if estimator is None:
            self.estimator = Estimator()
        else:
            self.estimator = estimator
        if localizer is None:
            self.localizer = Localizer()
        else:
            self.localizer = localizer
        if estimator.instrument != localizer.instrument:
            warnings.warn("Warning: estimator and localizer have different instruments")
        self.instrument = estimator.instrument

    @property
    def coordinates(self):
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates):
        self._coordinates = coordinates
    
    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, estimator):
        self._estimator = estimator
        
    @property
    def localizer(self):
        return self._localizer

    @localizer.setter
    def localizer(self, localizer):
        self._localizer = localizer

    def predict(self, img_list=[],
                save_predictions=False, predictions_path='predictions.json'):
        '''
        output:
        predictions: list of features
        n images => n lists of features
        '''
        crop_px = self.estimator.pixels
        yolo_predictions = self.localizer.predict(img_list = img_list, save_to_json=False)
        (imcols, imrows, channels) = img_list[0].shape
        old_shape = (imrows, imcols)
        out_features = crop_feature(img_list = img_list, xy_preds = yolo_predictions, old_shape = old_shape, new_shape = crop_px)
        structure = list(map(len, out_features))
        flat_features = [item for sublist in out_features for item in sublist]
        imlist = [feat.data*100 for feat in flat_features]
        char_predictions = self.estimator.predict(img_list = imlist, save_to_json=False)
        zpop = char_predictions['z_p']
        apop = char_predictions['a_p']
        npop = char_predictions['n_p']
        for framenum in range(len(structure)):
            listlen = structure[framenum]
            frame = out_features[framenum]
            index = 0
            while listlen>index:
                feature = frame[index]
                feature.model.particle.z_p = zpop.pop(0)
                feature.model.particle.a_p = apop.pop(0)
                feature.model.particle.n_p = npop.pop(0)
                feature.model.coordinates = feature.coordinates
                feature.instrument = self.instrument
                index+=1
        return out_features
            


if __name__ == '__main__':
    instrument = Instrument()
    instrument.wavelength = 0.447
    instrument.magnification = 0.048
    instrument.n_m = 1.340

    keras_model_path = 'keras_models/predict_stamp_auto.h5'
    estimator = Estimator(model_path=keras_model_path, instrument=instrument)

    
    config_path =  'cfg_darknet/holo.cfg'
    weight_path ='cfg_darknet/holo_55000.weights'
    meta_path = 'cfg_darknet/holo.data'
    localizer = Localizer(config_path = config_path, weight_path = weight_path, meta_path = meta_path, instrument=instrument)

    img_file = 'examples/test_image_large.png'
    import cv2
    img = cv2.imread(img_file)
    img_list = [img]
    
    e2e = EndtoEnd(estimator=estimator, localizer=localizer)
    features = e2e.predict(img_list = img_list)
    example = features[0][0]


    print('Example feature')
    print(example.model.particle)
    pix = estimator.pixels
    
    h = example.model.hologram()
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(example.data.reshape(pix), cmap='gray')
    ax2.imshow(h.reshape(pix), cmap='gray')
    fig.suptitle('Data, Predicted Hologram')
    plt.show()


    result = example.optimize()
    report_fit(result)
    h = example.model.hologram()
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(example.data.reshape(pix), cmap='gray')
    ax2.imshow(h.reshape(pix), cmap='gray')
    ax3.imshow(example.residuals().reshape(pix), cmap='gray')
    fig.suptitle('Data, optimized hologram, residual')
    plt.show()
