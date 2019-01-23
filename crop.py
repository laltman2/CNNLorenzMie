import numpy as np
import json, os
from matplotlib import pyplot as plt
from PIL import Image



def crop(img_list=[], img_names_path=None,
         xy_preds=[], xy_preds_json=None,
         old_pixels=(1024,1280), new_pixels=(200,200),    #not sure i need old_pixels, coords might be more useful for masking etc
         showImage=False, verbose=False,
         save_to_folder=False, crop_dir='./cropped_img/'):

    '''
    option to load images/predictions from file or to feed in as lists
    if both, the variable input is listed before file input
    xy_preds[i] corresponds to img_list[i]


    output:
    list of images
    '''

    (img_rows, img_cols) = old_pixels
    (crop_img_rows, crop_img_cols) = new_pixels
    if not img_names_path is None:
        with open(img_names_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            filename = line.rstrip()
            img_local = np.array(Image.open(filename))
            img_list.append(img_local)

    if not xy_preds_json is None:
        with open(xy_preds_json, 'r') as f:
            data = json.load(f)
        xy_preds += data

    numfiles = len(img_list)
    numpreds = len(xy_preds)
    if numfiles!=numpreds:
        raise Exception('Number of images: {} does not match number of predictions: {}'.format(numfiles, numpreds))
            
    crop_img = []
    x_pred=[]
    y_pred=[]
    for num in range(numfiles):
        if verbose:
            print('File:', num+1)
        img_local = img_list[num]
        preds_local = xy_preds[num]
        if showImage:
            fig,ax = plt.subplots(1)
            ax.imshow(img_local, cmap='gray')
            fig.suptitle('Example detection %i' % (num+1))
        for pred in preds_local:
            conf = pred["conf"]*100
            (x,y,w,h) = pred["bbox"]
            center = (round(x,2),round(y,2))
            if verbose:
                print("Detection at {} with {}% confidence.".format(center, round(conf)))
            if showImage:
                ax.plot([x], [y], 'ro')
                ax.text(x+10, y-10, '{}%'.format(round(conf)), bbox=dict(facecolor='white', alpha=0.5))
            xc = int(x)-1
            yc = int(y)+1
            if crop_img_rows % 2 == 0:
                right_frame = left_frame = int(crop_img_rows/2)
            else:
                left_frame = int(np.floor(crop_img_rows/2))
                right_frame = int(np.ceil(crop_img_rows/2))
            xbot = xc - left_frame
            xtop = xc + right_frame
            if crop_img_cols % 2 == 0:
                top_frame = bot_frame = int(crop_img_cols/2)
            else:
                top_frame = int(np.ceil(crop_img_cols/2))
                bot_frame = int(np.floor(crop_img_cols/2))
            ybot = yc - bot_frame
            ytop = yc + top_frame
            cropped = img_local[ybot:ytop, xbot:xtop]
            crop_img.append(cropped[:,:,0])
            x_pred.append(x)
            y_pred.append(y)
        if showImage:
            plt.show()

    img_files = []
    if save_to_folder:
        crop_dir = os.path.abspath(crop_dir)
        if not os.path.exists(crop_dir):
            os.makedirs(crop_dir)
        numcrops = len(crop_img)
        filenames_save = ''
        for num in range(numcrops):
            local_img_save = Image.fromarray(crop_img[num])
            local_img_path = crop_dir+'/image'+str(num+1).zfill(4)+'.png'
            img_files.append(local_img_path)
            local_img_save.save(local_img_path, 'png')
            filenames_save += local_img_path+'\n'
        with open(crop_dir+'/filenames.txt', 'w') as f:
            f.write(filenames_save)

    return crop_img

if __name__=='__main__':
    img_files = '/home/group/example_data/movie_img/filenames.txt'
    preds_file = './yolo_predictions.json'
    crop(img_names_path=img_files, xy_preds_json=preds_file, save_to_folder=False, showImage=True)
