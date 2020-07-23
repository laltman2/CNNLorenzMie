import numpy as np
import tensorflow.keras as keras
import warnings
from tensorflow.keras import backend as K
from pylorenzmie.theory.Instrument import Instrument, coordinates
from CNNLorenzMie.Estimator import Estimator
from CNNLorenzMie.Localizer import Localizer
from CNNLorenzMie.crop_feature import crop_feature
from CNNLorenzMie.filters.nodoubles import nodoubles
from CNNLorenzMie.filters.no_edges import no_edges
from pylorenzmie.analysis import Feature


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

    def predict(self, img_list=[], doubles_tol=0, edge_tol=0):
        '''
        output:
        predictions: list of features
        n images => n lists of features
        '''
        crop_px = self.estimator.pixels
        yolo_predictions = self.localizer.predict(img_list=img_list)
        yolo_predictions = nodoubles(yolo_predictions, tol=doubles_tol)
        (imcols, imrows, channels) = img_list[0].shape
        old_shape = (imrows, imcols)
        yolo_predictions = no_edges(
            yolo_predictions, tol=edge_tol, image_shape=old_shape)
        out_features, est_images, scales = crop_feature(
            img_list=img_list, xy_preds=yolo_predictions, new_shape=crop_px)
        structure = list(map(len, out_features))
        char_predictions = self.estimator.predict(
            img_list=est_images, scale_list=scales)
        zpop = char_predictions['z_p']
        apop = char_predictions['a_p']
        npop = char_predictions['n_p']
        for framenum in range(len(structure)):
            listlen = structure[framenum]
            frame = out_features[framenum]
            index = 0
            while listlen > index:
                feature = frame[index]
                feature.model.particle.z_p = zpop.pop(0)
                feature.model.particle.a_p = apop.pop(0)
                feature.model.particle.n_p = npop.pop(0)
                feature.model.instrument = self.instrument
                index += 1
        return out_features


if __name__ == '__main__':
    from lmfit import report_fit
    import cv2
    import json
    from matplotlib import pyplot as plt

    keras_head_path = 'keras_models/predict_stamp_best'
    keras_model_path = keras_head_path+'.h5'
    keras_config_path = keras_head_path+'.json'
    with open(keras_config_path, 'r') as f:
        kconfig = json.load(f)
    estimator = Estimator(model_path=keras_model_path, config_file=kconfig)

    localizer = Localizer(configuration='holo', weights='_100k')

    img_file = 'examples/test_image_large.png'
    img = cv2.imread(img_file)
    img_list = [img]

    e2e = EndtoEnd(estimator=estimator, localizer=localizer)
    features = e2e.predict(img_list=img_list)
    example = features[0][0]

    print('Example feature')
    print(example.model.particle)
    px = int(np.sqrt(example.data.size))
    pix = (px, px)
    cpix = estimator.pixels

    h = example.model.hologram()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(example.data.reshape(pix), cmap='gray')
    ax2.imshow(h.reshape(pix), cmap='gray')
    fig.suptitle('Data, Predicted Hologram')
    plt.show()

    result = example.optimize()
    print(result)
    print(example.model.particle)
    h = example.model.hologram()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(example.data.reshape(pix), cmap='gray')
    ax2.imshow(h.reshape(pix), cmap='gray')
    ax3.imshow(example.residuals().reshape(pix), cmap='gray')
    fig.suptitle('Data, optimized hologram, residual')
    plt.show()
