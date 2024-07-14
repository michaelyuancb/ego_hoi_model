import numpy as np 
import pdb
import cv2
from PIL import Image

from ego_hoi_segmentation import EgoHOISegmentation
from ego_hoi_detection import EgoHOIDetection
from ego_hos_wrapper import EgoHOSWrapper

if __name__ == "__main__":

    image_fp = "image.jpg"
    
    ego_hoi_detection = EgoHOIDetection(repo_path='../repo')
    det_result = ego_hoi_detection.detect(image_fp, vis=True)

    # I choose to use the max-area of semantic SAM, which help me to:
    # (1) segment two hands & arms when only 1 hand / arm is detected by detector.
    # (2) get the conservative estimation of dynamic area.

    ego_hoi_segmentation = EgoHOISegmentation(repo_path='../repo')
    seg_result, det_result = ego_hoi_segmentation.segment(image_fp, hand_threshold=0.2, vis=True)
    Image.fromarray((seg_result['dynamic_area']*255).astype(np.uint8)).save('mask_dynamic.png')
    pdb.set_trace()

    # det_result: {'left': {'offset': array([ 0.03231527,  0.08044394, -0.05940348], dtype=float32), 'bbox_obj': array([239.67389, 161.16383, 288.72702, 205.0075 ], dtype=float32), 'bbox_hand': array([205.20311, 187.94205, 245.9829 , 242.21669], dtype=float32), 'confidence_hand': 0.99403363, 'state': 'P', 'state_explaination': 'Portable Object'}, 'right': {'offset': array([ 0.06348144, -0.02577479,  0.09662122], dtype=float32), 'bbox_obj': array([229.75223, 185.38179, 273.95844, 217.96408], dtype=float32), 'bbox_hand': array([263.487   ,  54.853817, 443.8651  , 240.19383 ], dtype=float32), 'confidence_hand': 7.680194e-06, 'state': 'P', 'state_explaination': 'Portable Object'}}
    # seg_result.keys() = dict_keys(['left_hand', 'left_obj', 'dynamic_area'])
    # seg_result['left_hand']: numpy, (h, w), np.bool_


    ego_hos_wrapper = EgoHOSWrapper(cache_path="/home/ycb/dflowmap_project/dflowmap/dfm/hoi/cache",  # an absolute file-path
                                    repo_path='../repo')
    ego_hos_wrapper.segment(image_fp, vis=True)