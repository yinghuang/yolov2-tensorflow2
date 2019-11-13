# -*- coding: utf-8 -*-
import numpy as np

class Config():
  def __init__(self, yolo_tiny=False):
      # 参考https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg
      # 注意，这个宽高是在最后特征图的尺度下的，不是在原图尺度上，在原图尺度上的话还要乘以步长32
#      self.ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828] # anchors coords unit : grid cell
      self.ANCHORS = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
      if yolo_tiny:
          self.ANCHORS = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
      # 0-GRID_W / GRID_H
      self.ANCHORS = np.array(self.ANCHORS)
      self.ANCHORS = self.ANCHORS.reshape(-1,2)
      
      self.IMAGE_W = 416
      self.IMAGE_H = 416
      self.GRID_W = 13
      self.GRID_H = 13
      self.NUM_ANCHORS = 5
      self.NUM_CLASSES = 20
      self.LAMBDA_NOOBJECT  = 1
      self.LAMBDA_OBJECT    = 5
      self.LAMBDA_CLASS     = 1
      self.LAMBDA_COORD     = 1
    
VOC_NAME_LABEL_CLASS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}
VOC_NAME_LABEL = {key:v[0] for key,v in VOC_NAME_LABEL_CLASS.items()}
VOC_LABEL_NAME = {v[0]:key for key,v in VOC_NAME_LABEL_CLASS.items()}