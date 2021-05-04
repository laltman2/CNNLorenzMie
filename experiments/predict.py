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

#img_eval = normalize_video(bkg_path, vid_path, save_folder='./norm_images/')



keras_head_path = '../keras_models/predict_stamp_best'
keras_model_path = keras_head_path+'.h5'
keras_config_path = keras_head_path+'.json'
with open(keras_config_path, 'r') as f:
    kconfig = json.load(f)
estimator = Estimator(model_path=keras_model_path, config_file=kconfig)

    
localizer = Localizer(configuration = 'holo', weights='_100k')

e2e = EndtoEnd(estimator=estimator, localizer=localizer)

savedict = []
path = os.getcwd()+'/norm_images'
numimgs = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])


#just do one at a time for now
for i in range(numimgs):
    filepath = path + '/image' + str(i).zfill(4) + '.png'
    localim = cv2.imread(filepath)
    features = e2e.predict(img_list = [localim])[0]
    for feature in features:
        localdict = feature.particle.properties
        shape = int(np.sqrt(feature.coordinates.shape[1]))
        localdict['shape'] = shape
        localdict['framenum'] = i
        localdict['framepath'] = os.path.abspath(filepath)
        print(localdict)
        savedict.append(localdict)
    #print('Completed frame {}'.format(i), end='\r')

with open('your_MLpreds.json', 'w') as f:
    json.dump(savedict, f)
print('saved ML')
