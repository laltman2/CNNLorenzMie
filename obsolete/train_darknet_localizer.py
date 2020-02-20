from __future__ import print_function
import json, shutil, os, cv2
import numpy as np
from mtd4train import mtd
try:
    from pylorenzmie.theory.CudaLMHologram import CudaLMHologram as LMHologram
except ImportError:
    from pylorenzmie.theory.LMHologram import LMHologram
from pylorenzmie.theory.Instrument import coordinates

'''One-stop training of a new darknet model for classification and localization of features in frame

Steps to follow:
-edit darknet_train_config.json with appropriate params
(make sure you have available disk space)
-Run this file with python or nohup
(dataset generation + training will take at least few hours)

Weights files will save to cfg_darknet/backup every 100 epochs, then, after 1000, every 1000 epochs
'''


configfile='darknet_train_config.json'

with open(configfile, 'r') as f:
    config = json.load(f)


#File names/numbers
file_header = os.path.abspath(config['directory'])
numtrain = config['nframes_train']
numtest = config['nframes_test']

numclasses = 1 #single class version (for now)

#Make test/train data
mtd_config = config.copy()
train_dir = file_header + '/train'
test_dir = file_header + '/test'

mtd_config['directory'] = train_dir
mtd_config['nframes'] = numtrain
print('Training set')
mtd(config = mtd_config)

mtd_config['directory'] = test_dir
mtd_config['nframes'] = numtest
print('Validation set')
mtd(config = mtd_config)

#prepare config files
save_header = os.path.abspath(config['training']['savefile'])
backup_dir = os.getcwd()+'/cfg_darknet/backup'
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

namesfile = save_header + '.names'
names = config['particle']['name']
with open(namesfile, 'w') as fw:
    fw.write(names)
print('Created names file')
    
datafile = save_header+'.data'
data = ('classes= {} \n'.format(numclasses) +
        'train= {}/filenames.txt \n'.format(train_dir) +
        'test= {}/filenames.txt \n'.format(test_dir) +
        'names= {} \n'.format(namesfile) +
        'backup= {}'.format(backup_dir))
with open(datafile, 'w') as fw:
    fw.write(data)
print('Created data file')
    
cfgfile = save_header + '.cfg'
istiny = config['training']['tiny']
if istiny:
    og_cfg = os.getcwd() + '/cfg_darknet/fromscratch/yolov3-tiny.cfg'
else:
    og_cfg = os.getcwd() + '/cfg_darknet/fromscratch/yolov3.cfg'
    
with open(og_cfg, 'r') as fr:
    cfg_lines = fr.readlines()

batch = config['training']['batch']
subdivisions = config['training']['subdivisions']
cfg_lines[2] = 'batch={}\n'.format(batch)
cfg_lines[3] = 'subdivisions={}\n'.format(subdivisions)


if istiny:
    numfilters = (numclasses+5)*3
    cfg_lines[134] = cfg_lines[176] = 'classes={}\n'.format(numclasses)
    cfg_lines[126] = cfg_lines[170] = 'filters={}\n'.format(numfilters)
else:
    numfilters = (numclasses+5)*3
    cfg_lines[609] = cfg_lines[695] = cfg_lines[782] = 'classes={}\n'.format(numclasses)
    cfg_lines[602] = cfg_lines[688] = cfg_lines[775] = 'filters={}\n'.format(numfilters)

with open(cfgfile, 'w') as fw:
    fw.writelines(cfg_lines)
print('Created config file')

save_json = save_header+'.json'
with open(save_json, 'w') as f:
    json.dump(config, f)
print('Saved training config')

#Train
weightsfile = config['training']['weightsfile']
cmd = 'darknet/darknet detector train {} {} {}'.format(datafile, cfgfile, weightsfile)
os.system(cmd)

if config['delete_files_after_training']:
    shutil.rmtree(file_header)
