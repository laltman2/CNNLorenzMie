import numpy as np
import re, os, json
import sys, shutil
from darknet import *
from pylorenzmie.theory.Instrument import Instrument

class Localizer(object):
    '''
    Attributes
    __________
    net: network
    meta: metadata
    instrument: Instrument
        Object resprenting the light-scattering instrument                                                                                                                         
    Methods
    _______
    predict(img_list, save_to_json, predictions_path)'''
    
    def __init__(self,
                 config_path='',
                 meta_path='',
                 weight_path='',
                 instrument=None):
        if instrument is None:
            self.instrument = Instrument()
        else:
            self.instrument = instrument
        self.net, self.meta = instantiate(config_path, weight_path, meta_path)

    @property
    def instrument(self):
        return self._instrument

    @instrument.setter
    def instrument(self, instrument):
        self._instrument = instrument

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, net):
        self._net = net

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, meta):
        self._meta = meta


    def predict(self, img_list = [], thresh = 0.5,
                save_to_json=False, predictions_path='yolo_predictions.json'):
        '''
        input:
        img_list: list of images (ie cv2.imread('image.png'))
        if save_to_json==True, saves predictions to predictions_path
        (predictions_path does nothing otherwise)
        thresh: int
        threshold for detection
        
        output:
        predictions: list of list of dicts
        n images => n lists of dicts
        per holo prediction:
             {'conf': 50%, 'bbox': (x_centroid, y_centroid, width, height)}
        '''
        
        predictions = []
        for image in img_list:
            yolopred = detect(self.net, self.meta, image, thresh)
            imagepreds = []
            for pred in yolopred:
                (label,conf,bbox) = pred
                imagepreds.append({'conf': conf, 'bbox': bbox})
            predictions.append(imagepreds)
        
        if save_to_json:
            with open(predictions_path, 'w') as outfile:
                json.dump(predictions, outfile)
        return predictions

if __name__=='__main__':
    config_path = 'cfg_darknet/holo.cfg'
    weight_path = 'cfg_darknet/holo_55000.weights'
    meta_path = 'cfg_darknet/holo.data'
    localizer = Localizer(config_path = config_path, weight_path = weight_path, meta_path = meta_path)
    import cv2
    img_file = 'examples/test_image_large.png'
    test_img = cv2.imread(img_file)
    pred_path = 'examples/test_yolo_pred.json'
    detection = localizer.predict(img_list = [test_img], save_to_json=True, predictions_path=pred_path)
    example = detection[0]
    for holo in example:
        (x,y,w,h) = holo['bbox']
        conf = holo['conf']*100
        x = round(x,3)
        y = round(y,3)
        conf = round(conf)
        print('Hologram detected at ({},{}) with {}% confidence.'.format(x,y,conf))
