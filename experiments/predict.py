from CNNLorenzMie.Estimator import Estimator
from CNNLorenzMie.experiments.normalize_image import normalize_video
from CNNLorenzMie.Localizer import Localizer
from CNNLorenzMie.EndtoEnd import EndtoEnd
import cv2, json
from matplotlib import pyplot as plt
import numpy as np
from lmfit import report_fit
import os, os.path


vid_path = './videos/your_measurement_vid.avi'
bkg_path = './videos/your_background_vid.avi'

img_eval = normalize_video(bkg_path, vid_path, save_folder='./norm_images/')



keras_head_path = '/home/group/endtoend/OOe2e/keras_models/predict_stamp_fullrange_adamnew_extnoise_lowscale'
keras_model_path = keras_head_path+'.h5'
keras_config_path = keras_head_path+'.json'
with open(keras_config_path, 'r') as f:
    kconfig = json.load(f)
estimator = Estimator(model_path=keras_model_path, config_file=kconfig)

    
localizer = Localizer(configuration = 'yolonew', weights='_100000')

e2e = EndtoEnd(estimator=estimator, localizer=localizer)

savedict = []
path = os.getcwd()+'/norm_images'
numimgs = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])


#just do one at a time for now
for i in range(numimgs):
    filepath = './norm_images/image' + str(i).zfill(4) + '.png'
    localim = cv2.imread(filepath)
    features = e2e.predict(img_list = [localim])[0]
    counter = 0
    for feature in features:
        counter += 1
        localdict = {'framenum':i}
        particle = feature.model.particle
        localdict['x_p'] = particle.x_p
        localdict['y_p'] = particle.y_p
        localdict['z_p'] = particle.z_p
        localdict['a_p'] = particle.a_p
        localdict['n_p'] = particle.n_p
        extsize = feature.data.size
        extpx = int(np.sqrt(extsize))
        localdict['ext'] = extpx
        savedict.append(localdict)
    print('Completed frame {}'.format(i), end='\r')

with open('your_MLpreds.json', 'w') as f:
    json.dump(savedict, f)
print('saved ML')


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
