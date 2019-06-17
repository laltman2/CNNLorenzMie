import numpy, os, cv2
from keras import backend as K
from matplotlib import pyplot as plt

'''
pipeline for converting videos of experimental data to normalized images that are ready to feed into the models.

for each dataset, you should have a measurement video and a background video

function normalize_image returns list of normalized frames (in addition to saving them)

normalized images will be saved as 3-channel .png with the naming scheme:

normalized_data/image0000.png
normalized_data/image0001.png

in order of frames
'''

# Function to extract frames 
#https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0

    # checks whether frames were extracted 
    success = 1

    img_list = []
    while success: 
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read()
        img_list.append(image)
        count += 1

    #remove the last (success=False) element of list
    img_list = img_list[:-1]
    return img_list



def normalize_video(bg_path, vid_path, save_folder = './norm_images/'):
    print('opening background video')
    img_bkg = FrameCapture(bg_path)
    print('{} frames'.format(len(img_bkg)))


    print('computing background')
    #Normalize
    bg_img = numpy.zeros(img_bkg[0].shape)
    for i in range(len(bg_img)):
        for j in range(len(bg_img[0])):
            pix_i = []
            for file in img_bkg:
                pix_i.append(file[i][j])
            pixel = numpy.median(pix_i)
            bg_img[i][j] = pixel

    #free up memory
    img_bkg = None

    '''
    bgim = Image.fromarray(numpy.uint8(bg_img))
    bgim.save('background.png')
    bgim.show()

    plt.imshow(bg_img)
    plt.show()
    '''

    print('opening measurement video')
    img_measure = FrameCapture(vid_path)
    print('{} frames'.format(len(img_measure)))
    dark = min([frame.min() for frame in img_measure])
    
    
    print('normalizing and saving')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    img_save = []
    #background_val and max_val are for debugging
    #background_val =[]
    #max_val = []
    for i in range(len(img_measure)):
        testimg = img_measure[i][:]
        numer =testimg - dark
        denom = numpy.clip((bg_img-dark),1,255)
        testimg = numpy.divide(numer, denom)*100.
        testimg = numpy.clip(testimg, 0, 255)
        #background_val.append(numpy.median(testimg))
        #max_val.append(numpy.max(testimg))
        img_measure[i] = None
        img_save.append(testimg)
        filename = os.path.dirname(save_folder) + '/image' + str(i).zfill(4) + '.png'
        cv2.imwrite(filename, testimg)

    #img_save =  numpy.array(img_save).astype('float32')
    return img_save


if __name__ == '__main__':
    dir = '/home/lauren/Desktop/birefringence/datasets/run1/'
    bkgpath = dir+'vaterite_2_bkg.avi'
    vidpath = dir+'vaterite_2.avi'
    normalize_video(bkgpath, vidpath)
