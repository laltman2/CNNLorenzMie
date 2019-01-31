import ctypes as ct
import random


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


def array_to_image(arr):
    arr = arr.transpose(2, 0, 1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = (arr / 255.0).flatten()
    data = c_array(ct.c_float, arr)
    im = IMAGE(w, h, c, data)
    return im


class BOX(ct.Structure):
    _fields_ = [("x", ct.c_float),
                ("y", ct.c_float),
                ("w", ct.c_float),
                ("h", ct.c_float)]


class DETECTION(ct.Structure):
    _fields_ = [("bbox", BOX),
                ("classes", ct.c_int),
                ("prob", ct.POINTER(ct.c_float)),
                ("mask", ct.POINTER(ct.c_float)),
                ("objectness", ct.c_float),
                ("sort_class", ct.c_int)]


class IMAGE(ct.Structure):
    _fields_ = [("w", ct.c_int),
                ("h", ct.c_int),
                ("c", ct.c_int),
                ("data", ct.POINTER(ct.c_float))]


class METADATA(ct.Structure):
    _fields_ = [("classes", ct.c_int),
                ("names", ct.POINTER(ct.c_char_p))]


libpath = "/home/dg86/python/CNNLorenzMie/darknet/libdarknet.so"
lib = ct.CDLL(libpath, ct.RTLD_GLOBAL)
lib.network_width.argtypes = [ct.c_void_p]
lib.network_width.restype = ct.c_int
lib.network_height.argtypes = [ct.c_void_p]
lib.network_height.restype = ct.c_int

predict = lib.network_predict
predict.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float)]
predict.restype = ct.POINTER(ct.c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [ct.c_int]

make_image = lib.make_image
make_image.argtypes = [ct.c_int, ct.c_int, ct.c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [ct.c_void_p,
                              ct.c_int,
                              ct.c_int,
                              ct.c_float,
                              ct.c_float,
                              ct.POINTER(ct.c_int),
                              ct.c_int,
                              ct.POINTER(ct.c_int)]
get_network_boxes.restype = ct.POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [ct.c_void_p]
make_network_boxes.restype = ct.POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [ct.POINTER(DETECTION), ct.c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [ct.POINTER(ct.c_void_p), ct.c_int]

network_predict = lib.network_predict
network_predict.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [ct.c_void_p]

load_net = lib.load_network
load_net.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int]
load_net.restype = ct.c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [ct.POINTER(DETECTION), ct.c_int, ct.c_int, ct.c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [ct.POINTER(DETECTION), ct.c_int, ct.c_int, ct.c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, ct.c_int, ct.c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [ct.c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [ct.c_char_p, ct.c_int, ct.c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [ct.c_void_p, IMAGE]
predict_image.restype = ct.POINTER(ct.c_float)


def instantiate(config_path, weight_path, meta_path):
    net = load_net(config_path.encode('ascii'), weight_path.encode('ascii'), 0)
    meta = load_meta(meta_path.encode('ascii'))
    return net, meta


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    if isinstance(image, bytes):
        # image is a filename
        # i.e. image = b'/darknet/data/dog.jpg'
        im = load_image(image, 0, 0)
    else:
        # image is a numpy array
        # i.e. image = cv2.imread('/darknet/data/dog.jpg')
        im = array_to_image(image)
        rgbgr_image(im)
    num = ct.c_int(0)
    pnum = ct.pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(
        net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms):
        do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append(
                    (meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes):
        free_image(im)
    free_detections(dets, num)
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
