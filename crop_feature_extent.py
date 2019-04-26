import numpy as np
from pylorenzmie.theory.Feature import Feature
from pylorenzmie.theory.Instrument import coordinates

'''
Cropping function meant for intermediate use in EndtoEnd object

If you're looking to use cropping as a standalone function, chances are crop.py will be more useful to you.
'''
def crop_feature(img_list=[], xy_preds=[],
         old_shape = (1280,1024), new_shape=(201,201)):

    '''
    img_list: list of images (np.ndarray) with shape: old_shape
    xy_preds is the output of a yolo prediction: list of list of dicts
    xy_preds[i] corresponds to img_list[i]


    output:
    list of list of feature objects
    '''
    (img_rows, img_cols) = old_shape


    numfiles = len(img_list)
    numpreds = len(xy_preds)
    if numfiles!=numpreds:
        raise Exception('Number of images: {} does not match number of predictions: {}'.format(numfiles, numpreds))
            
    frame_list = []
    for num in range(numfiles):
        feature_list = []
        img_local = img_list[num]
        preds_local = xy_preds[num]
        for pred in preds_local:
            f = Feature()
            conf = pred["conf"]*100
            (x,y,w,h) = pred["bbox"]
            xc = int(np.round(x))
            yc = int(np.round(y))
            ext = np.amax([int(w),int(h)])
            if ext <= new_shape[0]:
                (crop_img_rows, crop_img_cols) = new_shape
            else:
                scale = int(np.floor(ext/new_shape[0]) + 1)
                (crop_img_rows, crop_img_cols) = np.multiply(new_shape, scale)
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
            if xbot<0:
                xbot = 0
                xtop = crop_img_rows
            if ybot<0:
                ybot = 0
                ytop = crop_img_cols
            if xtop>img_rows:
                xtop = img_rows
                xbot = img_rows - crop_img_rows
            if ytop>img_cols:
                ytop = img_cols
                ybot = img_cols - crop_img_cols
            cropped = img_local[ybot:ytop, xbot:xtop]
            data = cropped[:,:,0]
            data = np.array(data)/100.
            data = np.array([item for sublist in data for item in sublist])
            f.data = data
            coords = coordinates(shape = (crop_img_rows, crop_img_cols), corner=(xbot, ybot))
            f.coordinates = coords
            f.model.particle.x_p = x
            f.model.particle.y_p = y
            feature_list.append(f)
        feature_list = np.array(feature_list)
        frame_list.append(feature_list)
    frame_list = np.array(frame_list)
    return frame_list

if __name__=='__main__':
    from matplotlib import pyplot as plt
    import json, cv2
    
    img_file = 'examples/test_image_large.png'
    img = cv2.imread(img_file)
    preds_file = 'examples/test_yolo_pred.json'

    with open(preds_file, 'r') as f:
        data = json.load(f)
    xy_preds = data

    nshape = (101,101)
    imlist = crop_feature(img_list=[img], xy_preds = xy_preds, new_shape=nshape)
    example = imlist[0]
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
