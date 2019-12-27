import cv2, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from CNNLorenzMie.crop_feature import crop_feature
from scipy import stats

from matplotlib.patches import Rectangle


with open('PSflow_mix1_refined.json', 'r') as f:
    d = json.load(f)

with open('PSflow_mix1_MLpreds.json', 'r') as f:
    dML = json.load(f)

dML = [x for x in dML if not np.allclose([x['x_p'], x['y_p']], [678, 119], atol=5)]
d = [x for x in d if not np.allclose([x['x_p'], x['y_p']], [678, 119], atol=5)]
 

true = [x for x in d if x['redchi']<10]
print(len(true))
a_t = [x['a_p'] for x in true]
n_t = [x['n_p'] for x in true]

n_p = [x['n_p'] for x in dML]
a_p = [x['a_p'] for x in dML]
z_p = [x['z_p'] for x in dML]

xy = np.vstack([a_p, n_p])
z = stats.gaussian_kde(xy)(xy)

plt.scatter(a_p, n_p, c=z, alpha=0.3)
plt.scatter(a_t, n_t, color='red', alpha=0.4)
plt.xlabel('a_p')
plt.ylabel('n_p')
plt.title('ML predictions')
plt.show()

xy = np.vstack([z_p, n_p])
z = stats.gaussian_kde(xy)(xy)

plt.scatter(z_p, n_p, c=z, alpha=0.3)
plt.show()



numframes = 100
    
count=0
while True:
    for pred in true:
        i = pred['framenum']
        filepath = './norm_images/image' + str(i).zfill(4) + '.png'
        localim = cv2.imread(filepath)
        print('ML: {}'.format(pred))
        localxy = {"conf":1}
        x_p = pred['x_p']
        y_p = pred['y_p']
        #ext = pred['ext']
        ext= 401
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
        print('ML: {}'.format(pred))
        #scales.append(int(np.floor(ext/201.))+1)
        #da_p.append(p.a_p - 1.15)
        ins = feature.model.instrument
        ins.wavelength = 0.447
        ins.magnification = 0.048
        ins.n_m = 1.34
        feature.model.coordinates = feature.coordinates
        pix = (ext, ext)
        cropdata = feature.data.reshape(pix)*100
        h = feature.model.hologram()
        res = feature.residuals()
        #cv2.imwrite('./crops/0617image{}.png'.format(str(i).zfill(4)), cropdata)
        count += 1
        
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.imshow(np.clip(feature.data.reshape(pix)*60, 0, 255), cmap='gray')
        ax2.imshow(np.clip(h.reshape(pix)*60, 0, 255), cmap='gray')
        ax3.imshow(res.reshape(pix), cmap='gray')
        fig.suptitle('Data, Predicted Hologram')
        plt.show()
        

