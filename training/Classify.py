import json
import numpy as np

'''
----------------------------------------
script for describing how YOLO will classify

probably will need to be written custom for each model
----------------------------------------
'''

#should it be a class?

def classify(sphere, config):
    names = config['particle']['names']
    if len(names) == 1:
        #only one class
        return 0
    
    if '+n_p' in names:
        #classify based on sign of (n_p - n_m)
        if sphere.n_p < config['instrument']['n_m']:
            return 0
        else:
            return 1
