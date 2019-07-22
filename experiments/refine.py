from CNNLorenzMie.crop_feature import crop_feature
import cv2, json
from matplotlib import pyplot as plt
import numpy as np
from lmfit import report_fit
from time import time


with open('your_MLpreds.json', 'r') as f:
    MLpreds =  json.load(f)

savedict = []
path = os.getcwd()+'/norm_images'
numimgs = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])

savedict = []

#just do one at a time for now
for i in range(numimgs):
    filepath = './norm_images/image' + str(i).zfill(4) + '.png'
    localim = cv2.imread(filepath)
    localpreds = [x for x in MLpreds if x['x_p'] > 750 and x['framenum']==i]
    #reformat for cropping
    for pred in localpreds:
        localxy = {"conf":1}
        x_p = pred['x_p']
        y_p = pred['y_p']
        ext = pred['ext']
        localxy["bbox"] = [x_p, y_p, ext, ext]
        features,_,_ = crop_feature(img_list = [localim], xy_preds = [[localxy]])
        if len(features[0]) != 1:
            print('Something went wrong')
            print(len(features[0]))
        feature = features[0][0]
        p = feature.model.particle
        p.z_p = pred['z_p']
        p.a_p = pred['a_p']
        p.n_p = pred['n_p']
        ins = feature.model.instrument
        ins.wavelength = 0.447
        ins.magnification = 0.048
        ins.n_m = 1.34
        feature.model.coordinates = feature.coordinates
        start = time()
        result = feature.optimize(method='lm')
        print("Time to fit: {:03f}".format(time() - start))
        redchi = (result.fun).dot(result.fun) / (result.fun.size - result.x.size)
        localdict = {'framenum':i}
        particle = feature.model.particle
        localdict['x_p'] = particle.x_p
        localdict['y_p'] = particle.y_p
        localdict['z_p'] = particle.z_p
        localdict['a_p'] = particle.a_p
        localdict['n_p'] = particle.n_p
        localdict['redchi'] = redchi
        savedict.append(localdict)
        z_last = particle.z_p
        pix = (ext, ext)
        #cropdata = feature.data.reshape(pix)*100
        #fname= './crops/image{}.png'.format(str(i).zfill(4))
        #print(fname)
        #cv2.imwrite(fname, cropdata)
    print('Completed frame {}'.format(i))


with open('your_refined.json', 'w') as f:
    json.dump(savedict, f)
print('Saved NM fits')
    

firstimagepreds = features
firstimagepath = 'norm_images/image{}.png'.format(str(numimgs-1).zfill(4))
firstimg = cv2.imread(firstimagepath)
fig, ax = plt.subplots()
ax.imshow(firstimg, cmap='gray')
for feature in firstimagepreds:
    x = feature.model.particle.x_p
    y = feature.model.particle.y_p
    ax.scatter([x],[y], color='red')
plt.show()

example = features[0]

print('Example feature')
print(example.model.particle)
px = int(np.sqrt(example.data.size))
pix = (px,px)
cpix = estimator.pixels
    
h = example.model.hologram()
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(np.clip(example.data.reshape(pix)*60, 0, 255), cmap='gray')
ax2.imshow(np.clip(h.reshape(pix)*60, 0, 255), cmap='gray')
fig.suptitle('Data, Predicted Hologram')
plt.show()
