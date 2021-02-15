#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import CNNLorenzMie.darknet as darknet
import torch
import os
import json


class Localizer(object):

    '''
    Attributes
    __________
    configuration: str
        Name of trained configuration

    names: list
        List of class names


    threshold: float
        Confidence threshold for feature detection
        default: 0.5

    Methods
    _______
    predict(img_list)
    '''

    def __init__(self, configuration='holo', threshold=0.3, version=''):
        basedir = os.path.dirname(os.path.abspath(__file__)) + '/cfg_yolov5/'
        self.configuration = configuration
        weightspath = basedir + configuration + str(version) + '/weights/best.pt'
        cfgpath = basedir + configuration + '.json'

        with open(cfgpath, 'r') as f:
            cfg = json.load(f)

        self.names = cfg['particle']['names']

        if not os.path.exists(weightspath):
            raise ImportError
        
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weightspath) 
        self.model.conf = threshold

    def predict(self, img_list=[]):
        '''Detect and localize features in an image

        Inputs
        ------
        img_list: list
           images to be analyzed

        thresh: float
           threshold confidence for detection

        Outputs
        -------
        predictions: list
            list of dicts
        n images => n lists of dicts
        per holo prediction:
             {'conf': 50%, 'bbox': (x_centroid, y_centroid, width, height)}
        '''
        results = self.model(img_list).xyxy
        predictions = []
        for image in results:
            image = image.cpu().numpy()
            imagepreds = []
            for pred in image:
                x1, y1, x2, y2 = pred[:4]
                w = x2-x1
                h = y2-y1
                x_p = (x1+x2)/2.
                y_p = (y1+y2)/2.
                bbox = [x_p, y_p, w, h]
                conf = pred[4]
                ilabel = int(pred[5])
                label = self.names[ilabel]
                imagepreds.append({'label': label, 'conf': conf, 'bbox': bbox})
            predictions.append(imagepreds)
        return predictions
            
            

if __name__ == '__main__':
    import cv2
    import matplotlib

    matplotlib.use('TKAgg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    localizer = Localizer('yolov5_test', version=2)
    print('done')
    img_file = 'examples/test_image_large.png'
    test_img = cv2.imread(img_file)
    detection = localizer.predict(img_list=[test_img])
    detection = localizer.predict(img_list=[test_img])
    example = detection[0]
    fig, ax = plt.subplots()
    ax.imshow(test_img, cmap='gray')
    for feature in example:
        (x, y, w, h) = feature['bbox']
        conf = feature['conf']
        msg = 'Feature at ({0:.1f}, {1:.1f}) with {2:.2f} confidence'
        print(msg.format(x, y, conf))
        print(w*2, h*2)
        test_rect = Rectangle(xy=(x - w/2, y - h/2), width=w, height=h, fill=False, linewidth=3, edgecolor='r')
        ax.add_patch(test_rect)
    plt.show()
