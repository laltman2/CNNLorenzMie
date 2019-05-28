import numpy, os, cv2
from keras import backend as K
from PIL import Image
from matplotlib import pyplot as plt

'''
pipeline for converting videos of experimental data to normalized images that are ready to feed into the models.

for each dataset, you should have a measurement video and a background video

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

    while success: 
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        count += 1


print('opening background image')
img_bkg = []
numbkg = len(os.listdir('bkg_images'))-1
bkg_file_head = './bkg_images/image'
for i in range(numbkg):
    stri = str(i+1).zfill(4)
    im = Image.open(bkg_file_head + stri+ '.png')
    im_np = numpy.array(im)
    #Convert to Grayscale
    im_np = im_np[:,:,0]
    img_bkg.append(im_np)
    im.close()


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

bgim = Image.fromarray(numpy.uint8(bg_img))
bgim.save('background.png')
bgim.show()


#Open images
print('opening images')
img_eval =[]
min_cand = []
numdirs = 32
for j in range(numdirs):
    dirname = 'raws/raw_images_{}'.format(str(j).zfill(2))
    numfiles = len(os.listdir(dirname))-1
    file_head = './{}/image'.format(dirname)
    for i in range(numfiles):
        stri = str(i+1)
        stri = stri.zfill(4)
        im = Image.open(file_head + stri+ '.png')
        im_np = numpy.array(im)
        #Convert to Grayscale
        im_np = im_np[:,:,0]
        img_eval.append(im_np)
        min_cand.append(im_np.min())
        im.close()
    
img_eval = numpy.array(img_eval).astype('float32')
img_bkg = numpy.array(img_bkg).astype('float32')
min_cand = numpy.array(min_cand).astype('float32')
dark = min_cand.min()
print(img_eval.shape)

print('saving')
img_save = []
background_val =[]
max_val = []
for i in range(len(img_eval)):
    testimg = img_eval[i][:]
    numer =testimg - dark
    denom = numpy.clip((bg_img-dark),1,255)
    testimg = numpy.divide(numer, denom)*100.
    testimg = numpy.clip(testimg, 0, 255)
    background_val.append(numpy.median(testimg))
    max_val.append(numpy.max(testimg))
    img_save.append(testimg)

img_save =  numpy.array(img_save).astype('float32')
print(img_save.shape)

background_valafter = []
max_valafter = []
for i in range(len(img_eval)):
    pilim = Image.fromarray(numpy.uint8(img_save[i]))
    npim = numpy.array(pilim)
    background_valafter.append(numpy.median(npim))
    max_valafter.append(numpy.max(npim))
    stri = str(i+1)
    stri = stri.zfill(4)
    filestr = './norm_images/image' + stri + '.png'
    pilim.save(filestr)


x = numpy.linspace(1, numfiles, num=len(img_eval))

plt.plot(x,max_val, 'b')
plt.plot(x, max_valafter, 'r')
plt.show()

plt.plot(x,background_val, 'b')
plt.plot(x, background_valafter, 'r')
plt.show()

pilim.show()



