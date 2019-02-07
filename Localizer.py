#!/usr/bin/env python
# -*- coding: utf-8 -*-

import darknet
import os


class Localizer(object):

    '''
    Attributes
    __________
    configuration: str
        Name of trained configuration

    threshold: float
        Confidence threshold for feature detection
        default: 0.5

    Methods
    _______
    predict(img_list)
    '''

    def __init__(self, configuration='holo', threshold=0.5):
        dir = 'cfg_darknet'
        self.configuration = configuration
        conf = os.path.join(dir, self.configuration + '.cfg')
        weights = os.path.join(dir, self.configuration + '.weights')
        metadata = os.path.join(dir, self.configuration + '.data')
        self.net, self.meta = darknet.instantiate(conf,
                                                  weights,
                                                  metadata)
        self.threshold = threshold

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

        predictions = []
        for image in img_list:
            yolopred = darknet.detect(
                self.net, self.meta, image, self.threshold)
            imagepreds = []
            for pred in yolopred:
                (label, conf, bbox) = pred
                imagepreds.append({'conf': conf, 'bbox': bbox})
            predictions.append(imagepreds)
        return predictions


if __name__ == '__main__':
    import cv2

    localizer = Localizer('holo')
    print('done')
    img_file = 'examples/test_image_large.png'
    test_img = cv2.imread(img_file)
    detection = localizer.predict(img_list=[test_img])
    example = detection[0]
    for feature in example:
        (x, y, w, h) = feature['bbox']
        conf = feature['conf']
        msg = 'Feature at ({0:.1f}, {1:.1f}) with {2:.2f} confidence'
        print(msg.format(x, y, conf))
