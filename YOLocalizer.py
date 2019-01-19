import numpy as np
import re, os, json
import sys, shutil

##Put your darknet path here##
darknet_filehead = '/home/group/lauren_yolo/darknet/'

sys.path.append(darknet_filehead)
pydn_path = darknet_filehead+'pydarknet.py'
if not os.path.exists(pydn_path): #conflict between darknet.py and darknet.c
    dn_path = darknet_filehead+'darknet.py'
    shutil.copyfile(dn_path, pydn_path)
from pydarknet import performDetect

sys.path.append('/home/group/lauren_yolo/')
from pylorenzmie.theory.Instrument import Instrument

class Localizer(object):
    '''
    Attributes
    __________
    config_path: str
    meta_path: str
    weight_path: str
    pixels: tuple
        (img_rows, img_cols)
    instrument: Instrument
        Object resprenting the light-scattering instrument                                                                                                                         
    Methods
    _______
    predict(img_names_path, save_to_json, predictions_path)'''
    
    def __init__(self,
                 config_path='',
                 meta_path='',
                 weight_path='',
                 pixels=(1024,1280),
                 instrument=None):
        self.pixels = pixels
        self.config_path = config_path
        self.meta_path = meta_path
        self.weight_path = weight_path
        if instrument is None:
            self.instrument = Instrument()
        else:
            self.instrument = instrument
        performDetect(configPath = config_path, weightPath = weight_path, metaPath= meta_path, showImage= False, initOnly=True)

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
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, config_path):
        self._config_path = config_path

    @property
    def meta_path(self):
        return self._meta_path

    @meta_path.setter
    def meta_path(self, meta_path):
        self._meta_path = meta_path

    @property
    def weight_path(self):
        return self._weight_path

    @weight_path.setter
    def weight_path(self, weight_path):
        self._weight_path = weight_path

    def predict(self, img_names_path='', save_to_json=False, predictions_path='yolo_predictions.json'):
        '''
        input:
        img_names_path: path to filenames.txt 
        if save_to_json==True, saves predictions to predictions_path
        (predictions_path does nothing otherwise)
        
        output:
        predictions: list of list of dicts
        n images => n lists of dicts
        per holo prediction:
             {'conf': 50%, 'bbox': (x_centroid, y_centroid, width, height)}
        '''
        
        predictions = []
        config_path = self._config_path
        weight_path = self._weight_path
        meta_path = self._meta_path
        with open(img_names_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            filename = line.rstrip()
            yolopred = performDetect(imagePath=filename, configPath = config_path, weightPath = weight_path, metaPath= meta_path, showImage= False)
            imagepreds = []
            for pred in yolopred:
                (label,conf,bbox) = pred
                imagepreds.append({'conf': conf, 'bbox': bbox})
            predictions.append(imagepreds)

        #predictions.append(instrument)
        
        if save_to_json:
            with open(predictions_path, 'w') as outfile:
                json.dump(predictions, outfile)
        return predictions

if __name__=='__main__':
    config_path = darknet_filehead + 'cfg/holo.cfg'
    weight_path = darknet_filehead + 'backup/holo_55000.weights'
    meta_path = darknet_filehead + 'cfg/holo.data'
    img_files = '/home/group/example_data/movie_img/filenames.txt'
    localizer = Localizer(config_path = config_path, weight_path = weight_path, meta_path = meta_path)
    detections = localizer.predict(img_names_path = img_files, save_to_json = False)
    example = detections[0]
    print('Image 1:')
    for holo in example:
        (x,y,w,h) = holo['bbox']
        conf = holo['conf']*100
        x = round(x,3)
        y = round(y,3)
        conf = round(conf)
        print('Hologram detected at ({},{}) with {}% confidence.'.format(x,y,conf))
