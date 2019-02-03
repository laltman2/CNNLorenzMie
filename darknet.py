#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ctypes import (CDLL, RTLD_GLOBAL, POINTER, pointer, Structure,
                    c_void_p, c_char_p, c_int, c_float)
import numpy as np
import os
from wurlitzer import pipes
import logging

logging.basicConfig()
logger = logging.getLogger('darknet')
logger.setLevel(logging.INFO)

# load libdarknet.so

package_dir = os.path.dirname(os.path.realpath(__file__))
libpath = os.path.join(package_dir, 'darknet', 'libdarknet.so')

lib = CDLL(libpath, RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class FEATURE(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# wrap darknet functions
load_network = lib.load_network
load_network.argtypes = [c_char_p, c_char_p, c_int]
load_network.restype = c_void_p

load_metadata = lib.get_metadata
load_metadata.argtypes = [c_char_p]
load_metadata.restype = METADATA

analyze_image = lib.network_predict_image
analyze_image.argtypes = [c_void_p, IMAGE]
analyze_image.restype = POINTER(c_float)

get_features = lib.get_network_boxes
get_features.argtypes = [c_void_p,
                         c_int,
                         c_int,
                         c_float,
                         c_float,
                         POINTER(c_int),
                         c_int,
                         POINTER(c_int)]
get_features.restype = POINTER(FEATURE)

merge_features = lib.do_nms_obj
merge_features.argtypes = [POINTER(FEATURE), c_int, c_int, c_float]

free_features = lib.free_detections
free_features.argtypes = [POINTER(FEATURE), c_int]

'''
load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(FEATURE)

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(FEATURE), c_int, c_int, c_float]

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]
letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE
'''


def instantiate(config, weights, metadata):
    '''Load darknet

    Inputs
    ------
    config: str
        path to config filename
    weights: str
        path to weights file
    metadata: str
        path to metadata

    Outputs
    -------
    network: pointer to loaded network
    metadata: pointer to loaded metadata
    '''

    logger.info('Loading network and metadata')
    with pipes() as (out, err):
        net = load_network(config.encode('ascii'),
                           weights.encode('ascii'), 0)
        meta = load_metadata(metadata.encode('ascii'))
    logger.info(err.read())
    return net, meta


def detect(network, metadata, image, thresh=0.5, hier_thresh=0.5, nms=0.45):
    '''Detect features in an image

    Inputs
    ------
    network: darknet instance
    metadata: metadata for network
    image: numpy.ndarray
        image to be analyzed
    thresh:
    hier_thresh:
    nms: threshold for non-maximal suppression

    Output
    ------
    List of features detected in image.
    feature: tuple
        classification: str
        confidence: float
        bounding box: [x, y, w, h]
    '''

    # convert numpy.ndarray to C image
    image = image.transpose(2, 0, 1)/255.
    c_image = IMAGE()
    c_image.c = image.shape[0]
    c_image.h = image.shape[1]
    c_image.w = image.shape[2]
    image = image.astype(np.float32).flatten()
    c_image.data = np.ctypeslib.as_ctypes(image)

    # analyze image
    analyze_image(network, c_image)
    nfeatures = c_int(0)
    pnfeatures = pointer(nfeatures)
    features = get_features(network, c_image.w, c_image.h,
                            thresh, hier_thresh,
                            None, 0, pnfeatures)
    nfeatures = pnfeatures[0]
    merge_features(features, nfeatures, metadata.classes, nms)

    # transfer features from C to python
    res = []
    for j in range(nfeatures):
        for i in range(metadata.classes):
            if features[j].prob[i] > 0:
                b = features[j].bbox
                res.append((metadata.names[i],
                            features[j].prob[i],
                            (b.x, b.y, b.w, b.h)))
    free_features(features, nfeatures)
    return res


if __name__ == "__main__":
    import cv2
    import time

    config = 'cfg_darknet/holo.cfg'
    weights = 'cfg_darknet/holo_55000.weights'
    metadata = 'cfg_darknet/holo.data'
    net, meta = instantiate(config, weights, metadata)

    image = cv2.imread('examples/test_image_large.png')

    for n in range(10):
        start = time.time()
        r = detect(net, meta, image)
        print(time.time() - start)
    print(r)
