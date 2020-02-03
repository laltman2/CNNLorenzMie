import numpy as np
from pylorenzmie.analysis import Feature
from pylorenzmie.theory import LMHologram
from pylorenzmie.theory import coordinates


def crop_center(img_local, center, cropshape):
    (xc, yc) = center
    (crop_img_rows, crop_img_cols) = cropshape
    (img_cols, img_rows) = img_local.shape[:2]
    if crop_img_rows % 2 == 0:
        right_frame = left_frame = int(crop_img_rows/2)
    else:
        left_frame = int(np.floor(crop_img_rows/2.))
        right_frame = int(np.ceil(crop_img_rows/2.))
    xbot = xc - left_frame
    xtop = xc + right_frame
    if crop_img_cols % 2 == 0:
        top_frame = bot_frame = int(crop_img_cols/2.)
    else:
        top_frame = int(np.ceil(crop_img_cols/2.))
        bot_frame = int(np.floor(crop_img_cols/2.))
    ybot = yc - bot_frame
    ytop = yc + top_frame
    if xbot < 0:
        xbot = 0
        xtop = crop_img_rows
    if ybot < 0:
        ybot = 0
        ytop = crop_img_cols
    if xtop > img_rows:
        xtop = img_rows
        xbot = img_rows - crop_img_rows
    if ytop > img_cols:
        ytop = img_cols
        ybot = img_cols - crop_img_cols
    cropped = img_local[ybot:ytop, xbot:xtop]
    corner = (xbot, ybot)
    return cropped, corner


'''
Cropping function meant for intermediate use in EndtoEnd object

If you're looking to use cropping as a standalone function, chances are crop.py will be more useful to you.
'''


def crop_feature(img_list=[], xy_preds=[],
                 new_shape=(201, 201)):
    '''
    img_list: list of images (np.ndarray) with shape: old_shape
    xy_preds is the output of a yolo prediction: list of list of dicts
    xy_preds[i] corresponds to img_list[i]


    output:
    list of list of feature objects
    '''

    numfiles = len(img_list)
    numpreds = len(xy_preds)
    if numfiles != numpreds:
        raise Exception('Number of images: {} does not match number of predictions: {}'.format(
            numfiles, numpreds))

    frame_list = []
    est_input_img = []
    est_input_scale = []
    for num in range(numfiles):
        feature_list = []
        img_local = img_list[num]
        preds_local = xy_preds[num]
        for pred in preds_local:
            f = Feature(model=LMHologram())
            conf = pred["conf"]*100
            (x, y, w, h) = pred["bbox"]
            xc = int(np.round(x))
            yc = int(np.round(y))
            ext = np.amax([int(w), int(h)])
            if ext <= new_shape[0]:
                crop_shape = new_shape
                scale = 1
            else:
                scale = int(np.floor(ext/new_shape[0]) + 1)
                crop_shape = np.multiply(new_shape, scale)
            cropped, corner1 = crop_center(img_local, (xc, yc), crop_shape)
            cropped = cropped[:, :, 0]
            est_img = cropped[::scale, ::scale]
            est_input_img.append(est_img)
            est_input_scale.append(scale)
            newcenter = [int(x) for x in np.divide(crop_shape, 2)]
            ext_shape = (ext, ext)
            data, corner2 = crop_center(cropped, newcenter, ext_shape)
            corner = np.add(corner1, corner2)
            data = np.array(data)/100.
            f.data = data
            coords = coordinates(shape=ext_shape, corner=corner)
            f.model.coordinates = coords
            f.model.particle.x_p = x
            f.model.particle.y_p = y
            feature_list.append(f)
        feature_list = np.array(feature_list)
        frame_list.append(feature_list)
    frame_list = np.array(frame_list)
    frlistsize = 0
    for frame in frame_list:
        frlistsize += len(frame)
    est_input_img = np.array(est_input_img)
    est_input_scale = np.array(est_input_scale)
    if frlistsize != len(est_input_img):
        print('error in output sizes')
        print('Frame list size:', frlistsize)
        print('Estimator input size:', len(est_input_img))
    return frame_list, est_input_img, est_input_scale


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import json
    import cv2

    img_file = 'examples/test_image_large.png'
    img = cv2.imread(img_file)
    preds_file = 'examples/test_yolo_pred.json'

    with open(preds_file, 'r') as f:
        data = json.load(f)
    xy_preds = data

    nshape = (101, 101)
    imlist = crop_feature(img_list=[img], xy_preds=xy_preds, new_shape=nshape)
    print(imlist[2])
    example = imlist[0][0]
    print('Example Image')
    for fnum in range(len(example)):
        print('Feature ', fnum+1)
        feat = example[fnum]
        print('(x,y) = ', feat.model.particle.r_p[:2])
        data = feat.data
        size = data.size
        shape = (int(np.sqrt(size)), int(np.sqrt(size)))
        plt.imshow(data.reshape(shape), cmap='gray')
        plt.show()
