import argparse
import logging
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

# set arguments
parser = argparse.ArgumentParser()
parser.add_argument("--annotation_dir", '-ad', type=str, default='sample_images')
parser.add_argument("--save_dir", type=str, default='sample_images/result')
parser.add_argument("--mode", default='grid')
args = parser.parse_args()

# set logger
logger = logging.getLogger("MaskManipulater")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s|%(filename)s] %(asctime)s > %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class MaskManipulater:
    
    def __init__(self, annotation_dir, save_dir, rgb_dict):
        self.annotation_dir = annotation_dir
        self.save_dir = save_dir
        self.rgb_dict = rgb_dict
        
    def convert_annotations(self, mode, grid_size=(9,9), threshold=0.5, use_multiprocessing=False):
        """
        Arguments:
            mode: one of ["grid", "ellipse"], str
            
        """
        # TODO: implement mode select
        
        fname_list = os.listdir(self.annotation_dir)
        pbar = tqdm(total=len(fname_list))
        
        # TODO: Implement when use multiprocessing
        if use_multiprocessing:
            logger.info("Start converting using multiprocess")
            return
        else:
            logger.info("Start converting without multiprocessing")
            for f in fname_list:
                # check whether image file name is correct
                if len(os.path.splitext(f)[-1]) == 0:
                    logger.warning("{} is not a file".format(f))
                    pbar.update()
                    continue
                result = self._convert_using_grid(f, grid_size, threshold, self.rgb_dict)
                
                plt.imsave(os.path.join(self.save_dir, f), result)
                pbar.update()
        pbar.close()
        
    def _convert_using_grid(self, base_name, grid_size, threshold, rgb_dict):
        
        annotation_image = cv2.imread(os.path.join(self.annotation_dir, base_name))
        annot_h, annot_w, annot_c = annotation_image.shape
        grid_h, grid_w = grid_size
        grid_area = grid_h * grid_w
        num_row = int(np.ceil(annot_h/grid_h))
        num_col = int(np.ceil(annot_w/grid_w))
        
        empty_image = np.ones([annot_h, annot_w, annot_c], dtype=np.uint8) * 255
        
        for k, v in rgb_dict.items():
            v = np.array(v)
            mask = cv2.inRange(annotation_image, v-1, v+1, cv2.THRESH_BINARY)
            for r in range(num_row):
                for c in range(num_col):
                    partial_sum = np.sum(mask[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w])
                    if partial_sum / grid_area > threshold:
                        empty_image[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w, :] = v
        
        return empty_image
                        
    def _fit_ellipse(self, base_name, rgb_dict):
        
        # load annotation image
        annotation_image = cv2.imread(os.path.join(self.annotation_dir, base_name))
        annot_h, annot_w, annot_c = annotation_image.shape
        
        # create empty image
        empty_image = np.ones([annot_h, annot_w, annot_c], dtype=np.uint8) * 255
        
        for k, rgb in rgb_dict.items():
            cnts, h = get_contours_of_rgb(annotation_image, rgb)
            
            for i in range(len(cnts)):
                # fit an ellipse when contour is parent
                if h[0][i][-1] == -1 and len(cnts[i]) > 4:
                    ellipse = cv2.fitEllipse(cnts[i])
                    empty_image = cv2.ellipse(empty_image, ellipse, rgb, -1)
        
        return empty_image
    

def get_contours_of_rgb(image, rgb):
    """
    Get contours that have given rgb value from image 
    
    Arguments:
        image: RGB image, numpy.ndarray
        rgb: list, [R, G, B] 
        
    Returns:
        cnts: points of contours, list
        h: hierarchy of contours, list
    """
    
    value = np.array(rgb)
    mask_bi = cv2.inRange(image, value-1, np.minimum(value+1, np.array([255, 255, 255])))
    kernel = np.ones((2, 2), np.uint8)
    mask_bi = cv2.morphologyEx(mask_bi, cv2.MORPH_OPEN, kernel)
    
    if cv2.__version__ == "4.1.0":
        cnts, h = cv2.findContours(mask_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    else:
        _, cnts, h = cv2.findContours(mask_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    return cnts, h
    
    
def main():
    rgb_dict = yaml.load(open('configs/damage_rgb.yaml'), Loader=yaml.FullLoader)
    mask_manipulater = MaskManipulater(args.annotation_dir, args.save_dir, rgb_dict)
    mask_manipulater.convert_annotations(mode=args.mode)

    
if __name__ == "__main__":
    main()
