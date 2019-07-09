##########
# viz_util
#####################
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
from PIL import Image, ImageDraw, ImageFont

# TODO: draw bbox using PIL package
def draw_bbox_with_text(image, bbox, text, color, thickness=4, text_origin='up'):
    temp = image.copy()
    cv2.rectangle(temp, bbox[0:2], bbox[2:4], color=color, thickness=thickness)
    
    # draw text in filled rectangle
    margin = 10
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    if text_origin == 'up':
        cv2.rectangle(temp, (bbox[0],bbox[1]-text_size[0][1]-margin), (bbox[0]+text_size[0][0], bbox[1]), color=tuple(color), thickness=-1)
        cv2.putText(temp, text, (bbox[0],bbox[1]-margin//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), thickness=2)
    elif text_origin == 'down':
        cv2.rectangle(temp, (bbox[0],bbox[3]+text_size[0][1]+margin), (bbox[0]+text_size[0][0], bbox[3]), color=tuple(color), thickness=-1)
        cv2.putText(temp, text, (bbox[0],bbox[3]+text_size[0][1]+margin//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), thickness=2)
    
    return temp


def get_colors(cmap, n_classes):
    colors = np.array(cmap.colors) * 255
    color_list = []
    
    if len(colors) < n_classes:
        warnings.warn('Number of classes is bigger than # of Colors')
        for i in range(n_classes):
            temp_color = colors[np.random.randint(0, n_classes, 1) % len(colors)][0]
            color_list.append(temp_color)
    else:
        color_list = colors[:n_classes]
        
    return np.array(color_list)
    
    ############
    # Image utils
    ##############
    # jpg => png 변환 (배경 날려서)
def jpg2png(mask, file_path) : 
    
    mask = mask.astype("uint8")
    
    tmp = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    
    r, g, b = cv2.split(mask)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    
    cv2.imwrite(file_path.replace("jpg", "png"), dst)
    
def count_contour_area(file_path, rgb):
    contour_areas = []
    
    mask = cv2.imread(file_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    
    _, cnts, h = get_contours_of_rgb(mask, rgb)
    
    for i in range(len(cnts)):
        if h[0][i][-1] == -1:
            area = cv2.contourArea(cnts[i])
            contour_areas.append(int(area))
            
    return np.array(contour_areas, dtype=np.int32)


def fill_contour(file_path, rgb):
    
    mask = cv2.imread(file_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    _, cnts, h = get_contours_of_rgb(mask, rgb)
    
    for i in range(len(cnts)):
        if h[0][i][-1] == -1:
            area = cv2.contourArea(cnts[i])
            if int(area) < 2000:
                mask = cv2.drawContours(mask, cnts, i, [255], 3)
    
    plt.imshow(mask)
    plt.show()
    return mask


def get_contours_of_rgb(image, rgb):
    
    value = np.array(rgb)
    mask_bi = cv2.inRange(image, value-1, np.minimum(value+1, np.array([255, 255, 255])))
    kernel = np.ones((2, 2), np.uint8)
    mask_bi = cv2.morphologyEx(mask_bi, cv2.MORPH_OPEN, kernel)
    
    if cv2.__version__ == "4.1.0":
        cnts, h = cv2.findContours(mask_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        _, cnts, h = cv2.findContours(mask_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    return cnts, h


def get_center_point(cnt):
    moments = cv2.moments(cnt)
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])
    
    return (cx, cy)


def combine_two_images(fg_image, bg_image, target_path=None):
    
    fg_image_gray = cv2.cvtColor(fg_image.copy(), cv2.COLOR_BGR2GRAY)
    fg_mask = cv2.inRange(fg_image_gray, 10, 254, cv2.THRESH_BINARY)
    bg_mask = cv2.bitwise_not(fg_mask)
    
    target_fg = cv2.bitwise_and(fg_image, fg_image, mask=fg_mask)
    target_bg = cv2.bitwise_and(bg_image, bg_image, mask=bg_mask)
    
    target_image = cv2.add(target_fg, target_bg)
    
    return target_image
    
#############
# Keras
############
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras import backend as K
import numpy as np


class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, 
                 histogram_freq=0, 
                 write_graph=True, 
                 write_images=False):  # add other arguments to __init__ if you need
        super(LRTensorBoard, self).__init__(log_dir=log_dir, 
                                            histogram_freq=histogram_freq, 
                                            write_graph=write_graph, 
                                            write_images=write_images)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency.
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.
    # Example for CIFAR-10 w/ batch size 100:
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # References
      - [Cyclical Learning Rates for Training Neural Networks](
      https://arxiv.org/abs/1506.01186)
    """

    def __init__(
            self,
            base_lr=0.001,
            max_lr=0.006,
            step_size=2000.,
            mode='triangular',
            gamma=1.,
            scale_fn=None,
            scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        if mode not in ['triangular', 'triangular2',
                        'exp_range']:
            raise KeyError("mode must be one of 'triangular', "
                           "'triangular2', or 'exp_range'")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2.**(x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** x
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * \
                np.maximum(0, (1 - x)) * self.scale_fn(self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault(
            'lr', []).append(
            K.get_value(
                self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
