import numpy as np
from itertools import combinations

'''
filter for use with localizer

removes predictions within a certain distance (tol) of each other

input: list of list of dicts (output of Localizer.predict)
output: list of list of dicts, with doubles removed
'''


def nodoubles(preds_list=[], tol=5):
    num_img = len(preds_list)
    preds_copy = preds_list.copy()
    for num in range(num_img):
        img_pred = preds_copy[num]
        for img1, img2 in combinations(img_pred, 2):
            x1, y1 = img1['bbox'][:2]
            x2, y2 = img2['bbox'][:2]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist<tol:
                img_pred.remove(img2)
    return preds_copy


if __name__=='__main__':
    import json
    preds_file = 'examples/test_yolo_pred.json'

    with open(preds_file, 'r') as f:
        xy_preds = json.load(f)
    
    print('Before:{}'.format(xy_preds))
    #the sample predictions were not close
    #using a ridiculous tolerance for demonstration purposes
    print('After:{}'.format(nodoubles(xy_preds, tol=1000)))
