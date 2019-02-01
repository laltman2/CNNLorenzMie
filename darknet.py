#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ctypes import (CDLL, RTLD_GLOBAL, POINTER, pointer, Structure,
                    c_void_p, c_char_p, c_int, c_float)
import os


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

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

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


def instantiate(config_path, weight_path, meta_path):
    '''Load darknet

    Arguments
    ---------
    config_path: path to confi filename
    weight_path: path to weights file
    meta_path: path to metadata
    '''
    net = load_network(config_path.encode('ascii'),
                       weight_path.encode('ascii'), 0)
    meta = load_metadata(meta_path.encode('ascii'))
    return net, meta


def array_to_image(arr):
    '''Convert numpy ndarray to darknet image'''
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr / 255.0).flatten()
    data = (c_float * len(arr))()
    data[:] = arr
    im = IMAGE(w, h, c, data)
    return im


def classify(net, meta, im):
    out = analyze_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


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

    c_image = array_to_image(image)
    analyze_image(network, c_image)
    nfeatures = c_int(0)
    pnfeatures = pointer(nfeatures)
    features = get_features(network, c_image.w, c_image.h,
                            thresh, hier_thresh,
                            None, 0, pnfeatures)
    nfeatures = pnfeatures[0]
    merge_features(features, nfeatures, metadata.classes, nms)

    res = []
    for j in range(nfeatures):
        for i in range(metadata.classes):
            if features[j].prob[i] > 0:
                b = features[j].bbox
                res.append((metadata.names[i],
                            features[j].prob[i],
                            (b.x, b.y, b.w, b.h)))
    # res = sorted(res, key=lambda x: -x[1])
    free_features(features, nfeatures)
    return res


if __name__ == "__main__":
    import cv2
    image = cv2.imread('examples/test_image_large.png')
    config = 'cfg_darknet/holo.cfg'
    weights = 'cfg_darknet/holo_55000.weights'
    metadata = 'cfg_darknet/holo.data'
    net, meta = instantiate(config, weights, metadata)
    r = detect(net, meta, image)
    print(r)
