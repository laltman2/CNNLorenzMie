import numpy as np
import os, cv2
#from cv2 import *
from matplotlib import pyplot as plt
from vmedian import vmedian

'''
pipeline for converting videos of experimental data to normalized images that are ready to feed into the models.

for each dataset, you should have a measurement video and a background video

function normalize_image returns list of normalized frames (in addition to saving them)

normalized images will be saved as 3-channel .png with the naming scheme:

norm_images/image0000.png
norm_images/image0001.png

in order of frames
'''



def normalize_video(bg_path, vid_path, save_folder = './norm_images/', order = 2):
    #get first frame of background
    vidObj = cv2.VideoCapture(bg_path)
    success, img0 = vidObj.read()
    img0 = img0[:,:,0]
    if not success:
        print('background video not found')
        return

    print('Opening and computing background')
    #instantiate vmedian object
    v = vmedian(order=order, dimensions=img0.shape)
    v.add(img0)
    while success:
        success, image = vidObj.read()
        if success:
            image = image[:,:,0]
            v.add(image)
    #get background once video is done
    bg = v.get()
    
    '''
    #save background image
    bgimpath = save_folder + 'background.png'
    cv2.imwrite(bgimpath, bg)
    plt.imshow(bg, cmap='gray')
    plt.show()
    '''

    print('Opening measurement video')

    #make save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    
    #get videocap object for measurement video
    vidObj = cv2.VideoCapture(vid_path)
    nframes = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    print(nframes, 'frames')

    print('Computing dark count')
    #get dark count
    samplecount=100 #how many frames to sample (at random)
    subtract=5 #offset dark count
    min_cand = []
    positions = np.random.choice(nframes, samplecount, replace=False) #get random frames to sample
    for i in range(samplecount):
        vidObj.set(cv2.CAP_PROP_POS_FRAMES, positions[i])
        success, image = vidObj.read()
        if success:
            min_cand.append(image.min())
        else:
            print('Something went wrong')
    dark = min(min_cand) - subtract
 
    print('Normalizing')
    #load and normalize measurement video
    img_return = []
    success = 1
    count=0
    vidObj.set(cv2.CAP_PROP_POS_FRAMES, count)
    frame = vidObj.get(cv2.CAP_PROP_POS_FRAMES)
    while success:
        success, image = vidObj.read()
        if success:
            numer =image[:,:,0] - dark
            denom = np.clip((bg-dark),1,255)
            testimg = np.divide(numer, denom)*100.
            testimg = np.clip(testimg, 0, 255)
            img_return.append(testimg)
            filename = os.path.dirname(save_folder) + '/image' + str(count).zfill(4) + '.png'
            cv2.imwrite(filename, testimg)
            print(filename, end='\r')
            count+= 1
    return img_return


if __name__ == '__main__':
    dir = '/home/lauren/Desktop/birefringence/datasets/run1/'
    bkgpath = dir+'vaterite_2_bkg.avi'
    vidpath = dir+'vaterite_2.avi'
    normalize_video(bkgpath, vidpath)
