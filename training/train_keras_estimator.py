from __future__ import print_function
from CNNLorenzMie.training.Batch_Generator import makedata, loaddata, Batch_Generator
from CNNLorenzMie.training.estimator_arch import multioutput_model, callbacks
import json, shutil, os
import numpy as np

'''One-stop training of a new keras model for characterization of stamp-sizes holographic Lorenz-Mie images
'''

configfile='keras_train_config.json'
with open(configfile, 'r') as f:
    config = json.load(f)

makedata(config)

train_generator = Batch_Generator(config=config, settype='train')
test_generator = Batch_Generator(config=config, settype='test')

#Training Parameters
batch_size = config['training']['batchsize']
epochs = config['training']['epochs']
save_file = str(config['training']['savefile'])
numtrain=config['train']['nframes']
numtest=config['test']['nframes']


estimator = multioutput_model()

estimator.fit_generator(generator= train_generator,
                        steps_per_epoch= (numtrain // batch_size),
                        epochs= epochs,
                        verbose=1,
                        validation_data= test_generator,
                        validation_steps= (numtest // batch_size))


print('Finished training')


save_keras = os.path.expanduser(save_file+'.h5')
save_json = os.path.expanduser(save_file+'.json')

estimator.save(save_keras)
print('Saved keras model')

([img_eval, scale_eval], params_eval) = loaddata(config, settype='eval')
z_eval, a_eval, n_eval = params_eval

char_pred = estimator.predict([img_eval, scale_eval])
z_pred = char_pred[0]
a_pred = char_pred[1]
n_pred = char_pred[2]

def RMSE(gtru, pred):
    numimg = len(gtru)
    diff = np.zeros(numimg)
    for i in range(numimg):
        diff[i] = gtru[i] - pred[i]
    sqer = np.zeros(numimg)
    for i in range(numimg):
        sqer[i] = (diff[i])**2
    SST = np.sum(sqer)
    SST *= 1./numimg
    RMSE = np.sqrt(SST)
    return RMSE

z_RMSE = RMSE(z_eval, z_pred)
a_RMSE = RMSE(a_eval, a_pred)
n_RMSE = RMSE(n_eval, n_pred)

save_conf = config.copy()
del save_conf['directory']
del save_conf['imgtype']
del save_conf['delete_files_after_training']
save_conf['z_RMSE'] = z_RMSE
save_conf['a_RMSE'] = a_RMSE
save_conf['n_RMSE'] = n_RMSE

with open(save_json, 'w') as f:
    json.dump(save_conf, f)
print('Saved config')

if config['delete_files_after_training']:
    head_dir = os.path.expanduser(config['directory'])
    shutil.rmtree(head_dir)
