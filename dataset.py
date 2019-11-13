import numpy as np
import tensorflow as tf
from lxml import etree
from config import VOC_NAME_LABEL
import os 
import dataset_util
import logging

IMAGE_FEATURE_MAP = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32), # 如果数据中存放的list长度大于1, 表示数据是不定长的, 使用VarLenFeature解析
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    'image/object/view': tf.io.VarLenFeature(tf.string),
}



def parse_example(serialized_example,height,width):
  x = tf.io.parse_single_example(serialized_example, IMAGE_FEATURE_MAP)
  x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
  x_train = tf.image.resize(x_train, (height,width))
#  class_text = x['image/object/class/text'] # 原始类型是SparseTensor, https://blog.csdn.net/JsonD/article/details/73105490
#  class_text = tf.sparse.to_dense(x['image/object/class/text'], default_value='') 
  labels = tf.cast(tf.sparse.to_dense(x['image/object/class/label']), tf.float32)
  y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']), # shape: [m]
                      tf.sparse.to_dense(x['image/object/bbox/ymin']), # shape: [m]
                      tf.sparse.to_dense(x['image/object/bbox/xmax']), # shape: [m]
                      tf.sparse.to_dense(x['image/object/bbox/ymax']), # shape: [m]
                      labels  # shape: [m]
                      ], axis=1) # shape:[m, 5], m是图片中目标的个数, 每张图片的m可能不一样

  # 每个图片最多包含100个目标
  paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]] # 上下左右分别填充0, 100 - tf.shape(y_train)[0], 0, 0
  # The padded size of each dimension D of the output is:
  # paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]
  y_train = tf.pad(y_train, paddings)
  return x_train, y_train


    
def ground_truth_generator(dataset, ANCHORS, IMAGE_W, IMAGE_H, GRID_W, GRID_H, NUM_CLASSES):
  anchors = np.array(ANCHORS)
  anchors = anchors.reshape(-1,2) # [0-GRID_W, 0-GRID_H]
  num_anchors = anchors.shape[0]
    
  for batch,(x,y) in enumerate(dataset):
    x = tf.image.resize(x, (IMAGE_H, IMAGE_W)) # input: [batch, height, width, channels]
    x = x / 255
    y = y.numpy()
    
    batch_size = y.shape[0]
    detector_mask = np.zeros([batch_size, GRID_W, GRID_H, num_anchors, 1])
    y_true_anchor_boxes = np.zeros([batch_size, GRID_W, GRID_H, num_anchors, 5])
    y_true_class_hot = np.zeros([batch_size, GRID_W, GRID_H, num_anchors, NUM_CLASSES])
    y_true_boxes_all = np.zeros(y.shape) # [batch_size, 100, 5]
    for i in range(batch_size):
      boxes = y[i]
      for j, box in enumerate(boxes):
        # 0-1
        w = box[2] - box[0]
        h = box[3] - box[1]
        cx = (box[0] + box[2])/2 # center point coords
        cy = (box[1] + box[3])/2
        
        # 0-GRID_W / 0-GRID_H
        w *= GRID_W
        h *= GRID_H
        cx *= GRID_W
        cy *= GRID_H
        
        y_true_boxes_all[i,j]=np.array([cx,cy,w,h, box[4]])
        if w*h<=0:
          continue
        
        # cell index
        cell_col = np.floor(cx).astype(np.int)
        cell_row = np.floor(cy).astype(np.int)
        
        # find best anchor with highest iou, coords ->0-1
        anchors_w, anchors_h = anchors[:,0], anchors[:, 1]
        intersect = np.minimum(w, anchors_w) * np.minimum(h, anchors_h)
        union = anchors_w * anchors_h + w * h - intersect
        iou = intersect / union
        
        anchor_best = np.argmax(iou)
        
        cls_idx = int(box[4])
        y_true_anchor_boxes[i, cell_col, cell_row, anchor_best] = [cx, cy, w, h, cls_idx]
        y_true_class_hot[i, cell_col, cell_row, anchor_best, cls_idx-1] = 1 # class from 1-20
        detector_mask[i, cell_col, cell_row, anchor_best] = 1
        
    detector_mask = tf.convert_to_tensor(detector_mask, dtype='int64')
    y_true_anchor_boxes = tf.convert_to_tensor(y_true_anchor_boxes, dtype='float32')
    y_true_boxes_all = tf.convert_to_tensor(y_true_boxes_all, dtype='float32')
    y_true_class_hot = tf.convert_to_tensor(y_true_class_hot, dtype='float32')
    batch = (x, detector_mask, y_true_anchor_boxes, y_true_class_hot, y_true_boxes_all)
    yield batch
  
def transform(x, y, cfg):
  
  '''
  Ground truth batch generator from a yolo dataset, ready to compare with YOLO prediction in loss function.

  Parameters
  ----------
  - x : raw images tensors (batch_size, h, w, 3)
  - y : raw labels(xmin, ymin, xmax, ymax, label) (batch_size, max annot, 5)
  - ANCHORS : 预定义的n个anchors的宽高 (n, 2) 
  - IMAGE_H : 网络输入图片的高
  - IMAGE_W : 网络输入图片的宽
      
  Returns
  -------
  - x : images to predict. tensor (shape : batch_size, IMAGE_H, IMAGE_W, 3)
  - detector_mask : tensor, shape (batch, size, GRID_W, GRID_H, anchors_count, 1)
      1 if bounding box detected by grid cell, else 0
  - y_true_anchor_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
      Contains adjusted coords of bounding box in YOLO format
  - y_true_class_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
      One hot representation of bounding box label
  - y_true_boxes_all : annotations : tensor (shape : batch_size, max annot, 5)
      true_boxes format : x, y, w, h, c, coords unit : grid cell
  '''
  
  # 0-GRID_W / GRID_H
  anchors = cfg.ANCHORS
  
  num_anchors = anchors.shape[0]
  
#  x = tf.image.resize(x, (cfg.IMAGE_H, cfg.IMAGE_W)) # input: [batch, height, width, channels]
#  x = x / 255
  y = y.numpy()

  batch_size = y.shape[0]
  detector_mask = np.zeros([batch_size, cfg.GRID_W, cfg.GRID_H, num_anchors, 1])
  y_true_anchor_boxes = np.zeros([batch_size, cfg.GRID_W, cfg.GRID_H, num_anchors, 5])
  y_true_class_hot = np.zeros([batch_size, cfg.GRID_W, cfg.GRID_H, num_anchors, cfg.NUM_CLASSES])
  y_true_boxes_all = np.zeros(y.shape) # [batch_size, 100, 5]
  
  for i in range(batch_size):
    boxes = y[i]
    for j, box in enumerate(boxes):
      # 0-1
      w = box[2] - box[0]
      h = box[3] - box[1]
      cx = (box[0] + box[2])/2 # center point coords
      cy = (box[1] + box[3])/2
      
      # 0-GRID_W / 0-GRID_H
      w *= cfg.GRID_W
      h *= cfg.GRID_H
      cx *= cfg.GRID_W
      cy *= cfg.GRID_H
      
      y_true_boxes_all[i,j]=np.array([cx,cy,w,h, box[4]])
      if w*h<=0:
        continue
      
      # cell index
      cell_col = np.floor(cx).astype(np.int)
      cell_row = np.floor(cy).astype(np.int)
      
      # find best anchor with highest iou, coords ->0-1
      anchors_w, anchors_h = anchors[:,0], anchors[:, 1]
      intersect = np.minimum(w, anchors_w) * np.minimum(h, anchors_h)
      union = anchors_w * anchors_h + w * h - intersect
      iou = intersect / union
      
      anchor_best = np.argmax(iou)
      
      cls_idx = int(box[4])
      y_true_anchor_boxes[i, cell_col, cell_row, anchor_best] = [cx, cy, w, h, cls_idx]
      y_true_class_hot[i, cell_col, cell_row, anchor_best, cls_idx-1] = 1
      detector_mask[i, cell_col, cell_row, anchor_best] = 1
  
  detector_mask = tf.convert_to_tensor(detector_mask, dtype='int64')
  y_true_anchor_boxes = tf.convert_to_tensor(y_true_anchor_boxes, dtype='float32')
  y_true_boxes_all = tf.convert_to_tensor(y_true_boxes_all, dtype='float32')
  y_true_class_hot = tf.convert_to_tensor(y_true_class_hot, dtype='float32')
  batch = (x, detector_mask, y_true_anchor_boxes, y_true_class_hot, y_true_boxes_all)
  return batch

def OB_tfrecord_dataset(dataset_path, batch_size, cfg, shuffle=False):
  files = tf.data.Dataset.list_files(dataset_path)
  dataset = files.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(lambda x:parse_example(x, height=cfg.IMAGE_H, width=cfg.IMAGE_W))
  if shuffle:
    dataset = dataset.shuffle(buffer_size=500)
  dataset = dataset.batch(batch_size=batch_size)
  dataset = dataset.prefetch(
      buffer_size=tf.data.experimental.AUTOTUNE) #提前获取数据存在缓存里来减少gpu因为缺少数据而等待的情况
  return dataset












def get_imgpath_and_annots(data_dir, years, 
                           image_subdirectory = 'JPEGImages',
                           annotations_dir = 'Annotations',
                           ignore_difficult_instances = False):

    annos_list = []
    for year in years.keys():
      logging.info('Reading from PASCAL %s dataset.', year)
      sets = years[year]
      for _set in sets:
#        print('\nfor year: {}, set: {}'.format(year, _set))
        examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                     _set + '.txt')
        _annotations_dir = os.path.join(data_dir, year, annotations_dir)
        _examples_list = dataset_util.read_examples_list(examples_path)
        _annos_list = [os.path.join(_annotations_dir, example + '.xml') for example in _examples_list]
        annos_list += _annos_list
    
    img_names = []
    max_obj = 200
    annots = [] # 存放每个图片的boxes
    print('walking annot for each img...')
    for path in annos_list:
        with tf.io.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        
        width = int(data['size']['width'])
        height = int(data['size']['height'])
        
        boxes = []
        if 'object' not in data:
            continue
        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue
            
            box = np.array([
                    float(obj['bndbox']['xmin']) / width,
                    float(obj['bndbox']['ymin']) / height,
                    float(obj['bndbox']['xmax']) / width,
                    float(obj['bndbox']['ymax']) / height,
                    VOC_NAME_LABEL[obj['name']]
                    ])
            boxes.append(box) # 一个图片的box可能有多个
        boxes = np.stack(boxes)
        annots.append(boxes)
        
        img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
        img_path = os.path.join(data_dir, img_path)
        img_names.append(img_path)
    print('done')
    true_boxes = np.zeros([len(img_names), max_obj, 5])
    for idx, boxes in enumerate(annots):
        true_boxes[idx, :boxes.shape[0]] = boxes
    return img_names, true_boxes


def parse_image(filename, true_boxes, img_h, img_w):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (img_h,img_w))
    return image, true_boxes


def OB_tensor_slices_dataset(data_dir, years, batch_size, cfg, shuffle=False):
    img_names, bboxes = get_imgpath_and_annots(data_dir, years, 
                                                   image_subdirectory = 'JPEGImages',
                                                   annotations_dir = 'Annotations',
                                                   ignore_difficult_instances = False)
    print('bboxes shape:',bboxes.shape)
    dataset = tf.data.Dataset.from_tensor_slices((img_names, bboxes))
    dataset = dataset.map(lambda x,y:parse_image(x, y, img_h=cfg.IMAGE_H, img_w=cfg.IMAGE_W))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=500)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE) #提前获取数据存在缓存里来减少gpu因为缺少数据而等待的情况
    return dataset
