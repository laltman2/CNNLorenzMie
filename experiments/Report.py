import cv2, json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from CNNLorenzMie.crop_feature import crop_feature
from scipy import stats
from matplotlib.patches import Rectangle


class Report(object):

        '''
        Attributes
        __________
        ML_preds: list of dicts

        refined_preds: list of dicts

        omit: list of bools
            coniditional statements, if condition(feature) == True, omit from report
            example statement: (lambda x: x['n_p'] > 2.)


        Methods
        _______
        do_omit():
            runs the omit conditions on both sets of predictions

        display_detections(frame):
            displays frame with YOLO boxes around detections

        report_feature(conditions, predtype):
            predtype = 'ML', 'refined' 
            TO DO: add 'both'
            conditions: list of bools, like omit

            plots feature data, model hologram, and residuals, prints prediction
            for each feature of type which satisfies all conditions

        characterization_plot(predtype):
            predtype = 'ML', 'refined', 'both'

            displays map of a_p, n_p gaussian KDE

        TO DO:
        z_trajectory(xy_bounds, predtype):
            predtype = 'ML', 'refined', 'both'

            links a trajectory of single particle across multiple frames, plots time vs z_p
        '''
        
        def __init__(self,
                     ML_preds = [],
                     refined_preds = [],
                     omit= [] ):

                self.ML_preds = ML_preds
                self.refined_preds = refined_preds
                self.omit = omit
                
        @property
        def ML_preds(self):
                return self._ML_preds

        @ML_preds.setter
        def ML_preds(self, preds):
                self._ML_preds = preds
        
        @property
        def refined_preds(self):
                return self._refined_preds

        @refined_preds.setter
        def refined_preds(self, preds):
                self._refined_preds = preds
        
        @property
        def omit(self):
                return self._omit

        @omit.setter
        def omit(self, omit):
                self._omit = omit

        
        def do_omit(self):
                if not self.omit:
                        return
                
                for cond in self.omit:
                        self.ML_preds = [x for x in self.ML_preds if not cond(x)]
                
                for cond in self.omit:
                        self.refined_preds = [x for x in self.refined_preds if not cond(x)]
                                

        def display_detections(self, frame):
                if isinstance(frame, str):
                        frame_img = cv2.imread(frame)
                        detections = [x for x in self.ML_preds if x['framepath'] == frame]
                elif isinstance(frame, int):
                        detections = [x for x in self.ML_preds if x['framenum'] == frame]
                        if not detections:
                                print('No detections found for frame {}'.format(frame))
                                return
                        framepath = detections[0]['framepath']
                        print(framepath)
                        frame_img = cv2.imread(framepath)
                else:
                        print('Invalid frame type')
                        return
                        
                fig, ax = plt.subplots()
                fig.suptitle('Frame {}'.format(frame))
                ax.imshow(frame_img, cmap='gray')
                for feature in detections:
                        x = feature['x_p']
                        y = feature['y_p']
                        w,h = feature['shape']
                        test_rect = Rectangle(xy=(x - w/2, y - h/2), width=w, height=h, fill=False, linewidth=3, edgecolor='r')
                        ax.add_patch(test_rect)
                plt.show()

        def report_feature(self, conditions, predtype):
                self.do_omit()
                if predtype == 'ML':
                        predictions = [x for x in self.ML_preds for cond in conditions if np.all(cond(x))]
                elif predtype == 'refined':
                        predictions = [x for x in self.refined_preds  for cond in conditions if np.all(cond(x))]
                else:
                        print('Invalid predictions type')
                for pred in predictions:
                        localim = cv2.imread(pred['framepath'])
                        shape = pred['shape']
                        localxy = {"conf":1}
                        x_p = pred['x_p']
                        y_p = pred['y_p']
                        ext = shape[0]
                        localxy["bbox"] = [x_p, y_p, ext, ext]
                        features,_,_ = crop_feature(img_list = [localim], xy_preds = [[localxy]])
                        feature = features[0][0]
                        feature.deserialize(pred)
                        h = feature.model.hologram()
                        res = feature.residuals()
                        print(pred)
                        fig, axes = plt.subplots(1,3)
                        for ax in axes:
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.xaxis.set_label_position('top') 
                        (ax1, ax2, ax3) = axes
                        cropped_data = np.clip(feature.data.reshape(shape)*100, 0, 255)
                        ax1.imshow(cropped_data, cmap='gray')
                        ax2.imshow(np.clip(h.reshape(shape)*100, 0, 255), cmap='gray')
                        ax3.imshow(res.reshape(shape), cmap='gray')
                        ax1.set_xlabel('Data')
                        ax2.set_xlabel('Predicted Hologram')
                        ax3.set_xlabel('Residual')
                        z_str = '%.3f'%pred['z_p']
                        a_str = '%.3f'%pred['a_p']
                        n_str = '%.3f'%pred['n_p']
                        fig.suptitle('Frame {}'.format(pred['framenum']))
                        fig.text(0.5, 0.2, 'z_p = {}px \n a_p = {}um \n n_p = {}'.format(z_str, a_str, n_str), horizontalalignment='center')
                        plt.show()
                        return cropped_data
                        
        def characterization_plot(self, predtype):
                self.do_omit()
                if predtype == 'ML':
                        a_p = [x['a_p'] for x in self.ML_preds]
                        n_p = [x['n_p'] for x in self.ML_preds]
                        color = 'hot'
                        label = predtype
                elif predtype == 'refined':
                        a_p = [x['a_p'] for x in self.refined_preds]
                        n_p = [x['n_p'] for x in self.refined_preds]
                        color = 'cool'
                        label = predtype
                elif predtype == 'both':
                        pass #to do
                if not a_p:
                        print('No {} predictions found'.format(predtype))
                        return
                xy = np.vstack([a_p, n_p])
                z = stats.gaussian_kde(xy)(xy)
                fig, ax = plt.subplots()
                kde = ax.scatter(a_p, n_p, c=z, alpha=0.3, cmap = color)
                ax.set_xlabel('a_p')
                ax.set_ylabel('n_p')
                cb = fig.colorbar(kde)
                cb.set_label(label)
                cb.set_alpha(1)
                cb.draw_all()
                plt.show()
                
                
        

if __name__ == '__main__':
        
        with open('/home/group/datasets/Si_drop/1_5um/data/Si_MLpreds_serial_1218.json', 'r') as f:
                d = json.load(f)
                
        r = Report(ML_preds = d)
        #r.omit.append(lambda x: x['a_p']<1)
        #r.characterization_plot('ML')
        #r.display_detections(15)
        conditions = [(lambda x: x['framenum'] == 15)]
        r.report_feature(conditions, 'ML')
