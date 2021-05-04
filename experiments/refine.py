from CNNLorenzMie.crop_feature import crop_feature
import cv2, json
from matplotlib import pyplot as plt
import numpy as np
from lmfit import report_fit
from time import time
import os

with open('your_MLpreds.json', 'r') as f:
    MLpreds =  json.load(f)

savedict = []
path = os.getcwd()+'/norm_images'
numimgs = len([f for f in os.listdir(path)if os.path.isfile(os.path.join(path, f))])

savedict = []

#just do one at a time for now
for i in range(numimgs):
    filepath = path + '/image' + str(i).zfill(4) + '.png'
    localim = cv2.imread(filepath)
    localpreds = [x for x in MLpreds if x['framenum']==i]
    #reformat for cropping
    for pred in localpreds:
        print(pred)
        localxy = {"conf":1}
        x_p = pred['x_p']
        y_p = pred['y_p']
        ext = pred['shape']
        localxy["bbox"] = [x_p, y_p, ext, ext]
        features,_,_ = crop_feature(img_list = [localim], xy_preds = [[localxy]])
        #instatiates a feature, puts in data, coords, x_p, y_p
        if len(features[0]) != 1:
            print('Something went wrong')
            print(len(features[0]))
        feature = features[0][0]
        feature.deserialize(pred) #puts in rest of feature info
        start = time()
        result = feature.optimize(method='amoeba-lm')
        print("Time to fit: {:03f}".format(time() - start))
        print(result)
        localdict = feature.serialize(exclude=['data'])
        localdict['framenum'] = i
        localdict['framepath'] = os.path.abspath(filepath)
        localdict['redchi'] = result.redchi
        savedict.append(localdict)
    print('Completed frame {}'.format(i))

with open('your_refined.json', 'w') as f:
    json.dump(savedict, f)
print('Saved fits')
