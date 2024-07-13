import os
import torch
import pdb
import numpy as np 
from tqdm import tqdm

from PIL import Image
import cv2
import time
import argparse
import onnxruntime
import scipy.ndimage
from base64 import b64encode
from scipy.ndimage import zoom
from torchvision.transforms import ToTensor

from ego_hoi_detection import EgoHOIDetection

from semantic_sam import prepare_image, plot_multi_results, build_semantic_sam, plot_results, \
    SemanticSAMPredictor, SemanticSamAutomaticMaskGenerator

class EgoHOISegmentation(object):

    def __init__(self, 
                 # efficient_sam_weight_cfg='EfficientSAM/weights/efficient_sam_vits.pt',
                 repo_path='.',
                 semantic_sam_ckpt_cfg="Semantic-SAM/swint_only_sam_many2many.pth",
                 ego_hoi_det_cfg='ego_hand_detector/cfgs/res101.yml',
                 ego_hoi_det_pth='ego_hand_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth',
                 downsample_ratio=1,     # for faster inference
                 device='cuda',
                 ):
        self.repo_path = repo_path
        
        self.device = device
        self.semantic_sam_ckpt_cfg = semantic_sam_ckpt_cfg
        self.ego_hoi_det_cfg = ego_hoi_det_cfg
        self.ego_hoi_det_pth = ego_hoi_det_pth
        self.downsample_ratio = downsample_ratio

        self.hoi_det_model = EgoHOIDetection(repo_path=repo_path, cfg_file=self.ego_hoi_det_cfg, pretrained_path=self.ego_hoi_det_pth)

        model_type = semantic_sam_ckpt_cfg.split('/')[-1][4].upper()   
        # model_type: 'L' / 'T', depends on your checkpint
        self.semantic_sam = SemanticSAMPredictor(build_semantic_sam(model_type=model_type, ckpt=repo_path+'/'+semantic_sam_ckpt_cfg)) 


    def get_segment_point_from_bbox(self, input_points, image_shape: tuple):
        # input_points: numpy (4, )
        h, w = image_shape
        input_points = np.array([input_points[:2], input_points[2:]])
        input_points[..., 0] = input_points[..., 0] / (w - 1)
        input_points[..., 1] = input_points[..., 1] / (h - 1)
        input_points = input_points.mean(axis=0)[None]    
        # input point [[w, h]] relative location, i.e, [[0.5, 0.5]] is the center of the image
        return input_points

    
    def get_segment_mask_from_bbox(self, original_image, input_image, input_points, image_shape: tuple):
        h, w = image_shape
        input_points = self.get_segment_point_from_bbox(input_points, image_shape)
        masks, ious = self.semantic_sam.predict(original_image, input_image, point=input_points)  
        masks_area = (masks > 0).sum(axis=-1).sum(axis=-1)
        max_idx = masks_area.argmax().cpu().detach().item()
        mask = masks[max_idx].cpu().detach().numpy()                  # we choose the max mask
        mask = zoom(mask, (h / mask.shape[0], w / mask.shape[1]), order=1)
        mask = mask > 0
        return mask


    def segment(self, image_fp, hand_threshold=0.5, vis=False):

        det_result, (h, w) = self.hoi_det_model(image_fp, vis=vis)
        original_image, input_image = prepare_image(image_pth=image_fp)  

        seg_result = dict()
        dynamic_mask = []

        if det_result['left']['confidence_hand'] > hand_threshold:
            mask = self.get_segment_mask_from_bbox(original_image, input_image, det_result['left']['bbox_hand'], (h, w))
            seg_result['left_hand'] = mask
            dynamic_mask.append(mask)
            if det_result['left']['state'] not in ['N']:
                mask = self.get_segment_mask_from_bbox(original_image, input_image, det_result['left']['bbox_obj'], (h, w))
                seg_result['left_obj'] = mask
                if det_result['left']['state'] not in ['F']:
                    dynamic_mask.append(mask)

        if det_result['right']['confidence_hand'] > hand_threshold:
            mask = self.get_segment_mask_from_bbox(original_image, input_image, det_result['right']['bbox_hand'], (h, w))
            seg_result['right_hand'] = mask
            dynamic_mask.append(mask)
            if det_result['right']['state'] not in ['N']:
                mask = self.get_segment_mask_from_bbox(original_image, input_image, det_result['right']['bbox_obj'], (h, w))
                seg_result['right_obj'] = mask
                if det_result['right']['state'] not in ['F']:
                    dynamic_mask.append(mask)
        
        if len(dynamic_mask) > 0:
            dynamic_mask = np.stack(dynamic_mask)
            dynamic_mask = dynamic_mask.astype(np.float32).sum(axis=0) > 0
            seg_result['dynamic_area'] = dynamic_mask
        else:
            seg_result['dynamic_area'] = np.zeros((h, w)).astype(np.bool_)

        return seg_result, det_result