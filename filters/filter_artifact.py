import numpy as np

'''
filter for use with localizer

removes predictions within a certain distance (tol) of a specified location
for use on video artifacts, etc

input: list of list of dicts (output of Localizer.predict)
output: list of list of dicts, with doubles removed
'''


def filter_artifact(preds_list=[], location=[0,0], tol=0):
    num_img = len(preds_list)
    preds_copy = preds_list.copy()
    for num in range(num_img):
        img_pred = preds_copy[num]
        for img in img_pred:
            x1, y1 = location
            x2, y2 = img['bbox'][:2]
            dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
            if dist<tol:
                img_pred.remove(img)
    return preds_copy


if __name__=='__main__':
    import json
    preds_file = 'examples/test_yolo_pred.json'

    with open(preds_file, 'r') as f:
        xy_preds = json.load(f)
    
    print('Before:{}'.format(xy_preds))
    print('After:{}'.format(filter_artifact(xy_preds, location=[330,480], tol=10)))
