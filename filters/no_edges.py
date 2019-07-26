import numpy as np

'''
filter for use with localizer

removes predictions within a certain distance (tol) of each other

input: list of list of dicts (output of Localizer.predict)
output: list of list of dicts, with doubles removed
'''


def no_edges(preds_list=[], tol=200, image_shape=(1280,1024)):
    #checks for parameter compatibility
    minwidth = np.min(image_shape)
    if tol < 0 or tol > minwidth/2:
        print('Invalid tolerance for this frame size')
        return None
    
    xmin, ymin = (tol, tol)
    xmax, ymax = np.subtract(image_shape, (tol, tol))
    preds_copy = []
    for img in preds_list:
        preds_local = [pred for pred in img if pred['bbox'][0] >xmin and pred['bbox'][0] < xmax
                       and pred['bbox'][1] > ymin and pred['bbox'][1] < ymax]
        preds_copy.append(preds_local)
    return preds_copy


if __name__=='__main__':
    import json
    preds_file = '../examples/test_yolo_pred.json'

    with open(preds_file, 'r') as f:
        xy_preds = json.load(f)
    
    print('Before:{}'.format(xy_preds))

    #the sample predictions were not close
    #using a ridiculous tolerance for demonstration purposes
    print('After:{}'.format(no_edges(xy_preds, tol=325)))
