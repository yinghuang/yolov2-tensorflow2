
import os
import glob
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import VOC_LABEL_NAME

# save weights
def save_best_weights(model, name, val_loss_avg):
  # delete existing weights file
#  files = glob.glob(os.path.join('weights/', name + '*'))
#  for file in files:
#      os.remove(file)
  # create new weights file
  name = name + '_' + str(val_loss_avg) + '.h5'
  path_name = os.path.join('weights/', name)
  model.save_weights(path_name)
  return path_name
  
def log_loss(loss, val_loss, step):
  tf.summary.scalar('loss', loss, step)
  tf.summary.scalar('val_loss', val_loss, step)
  

def plot_img(img, boxes, labels, scores, cfg):
  """
  boxes : 物体box (cx, cy, w, h) [n, 4]
  """
  fig,ax = plt.subplots(1, figsize=(8,8))
  
  img = img.numpy()
  boxes = boxes.numpy().reshape(-1, 4)
  labels = labels.numpy()
  scores = scores.numpy()
  
  ax.imshow(img)
  IMG_H, IMG_W= img.shape[0:2]
  colors = ['r', 'orange', 'g', 'b', 'pink', 'purple']
  count = 0
  for box, label, score in zip(boxes, labels, scores):
    cx, cy, w, h = box
    if w*h<=0:
      continue
    count += 1
    cx = cx/cfg.GRID_W * IMG_W
    cy = cy/cfg.GRID_H * IMG_H
    w = w/cfg.GRID_W * IMG_W
    h = h/cfg.GRID_H * IMG_H
    name = VOC_LABEL_NAME[label+1]
#    print(name)
    ax.scatter(cx, cy, s=10, c='yellow')
    text = ' No:%d'%(count)+'_'+name+' %.3f'%(score)
    ax.text(cx-w/2, cy+h/2, text, fontdict={'size':15,'color':colors[count-1]})
    rect = patches.Rectangle((cx-w/2,cy-h/2), w, h, edgecolor=colors[count-1], linewidth=3.0, facecolor='none')
    ax.add_patch(rect)


def display_img(input_img, model, score_threshold, iou_threshold, cfg):
  """
  input_img : [ height, width, 3]
  model : detector
  score_threshold : for filter boxes with low confidence
  iou_threshold : for filter boxes by nms
  """
  
  y_pred = model.predict_on_batch(tf.expand_dims(input_img, 0))
  
  # 0-GRID_W -1 / GRID_H -1
  cell_coord_x = tf.cast(tf.reshape(tf.tile(tf.range(cfg.GRID_W), [cfg.GRID_H]), (1, cfg.GRID_H, cfg.GRID_W, 1, 1)), tf.float32)
  cell_coord_y = tf.transpose(cell_coord_x, (0,2,1,3,4))
  cell_coords = tf.tile(tf.concat([cell_coord_x, cell_coord_y], -1), [y_pred.shape[0], 1, 1, 5, 1])
  
  
  # 0-GRID_W / GRID_H
  anchors = cfg.ANCHORS
  
  # 0-GRID_W / GRID_H
  pred_xy = K.sigmoid(y_pred[:,:,:,:,0:2])
  pred_xy = pred_xy + cell_coords
  pred_wh = K.exp(y_pred[:,:,:,:,2:4]) * anchors
  # pred_confidence
  box_conf = K.sigmoid(y_pred[:,:,:,:,4:5])
#  print('max box conf: %.3f'%(K.max(box_conf).numpy()))
  # pred_class
  box_class_prob = K.softmax(y_pred[:,:,:,:,5:])
#  print('max box class prob: %.3f'%(K.max(box_class_prob).numpy()))
  # Convert box coords from x,y,w,h to x1,y1,x2,y2
  box_xy1 = pred_xy - 0.5 * pred_wh
  box_xy2 = pred_xy + 0.5 * pred_wh
  boxes = K.concatenate((box_xy1, box_xy2), axis=-1)
      
  # Filter boxes
  box_scores = box_conf * box_class_prob
  
  box_classes = K.argmax(box_scores, axis=-1) # best score index
  box_class_scores = K.max(box_scores, axis=-1) # best score
  print('max box class-specific score: %.3f'%(K.max(box_class_scores).numpy()))
  prediction_mask = box_class_scores >= score_threshold
  boxes = tf.boolean_mask(boxes, prediction_mask)
  scores = tf.boolean_mask(box_class_scores, prediction_mask)
  classes = tf.boolean_mask(box_classes, prediction_mask)
  
  # Non Max Supression
  selected_idx = tf.image.non_max_suppression(boxes, scores, 50, iou_threshold=iou_threshold)
  boxes = K.gather(boxes, selected_idx) # shape: [n, 4]
  scores = K.gather(scores, selected_idx) # shape: [n,]
  classes = K.gather(classes, selected_idx) # shape: [n,]
  _boxes = K.stack((
      0.5*(boxes[:,0] + boxes[:,2]),
      0.5*(boxes[:,1] + boxes[:,3]),
      boxes[:,2] - boxes[:,0],
      boxes[:,3] - boxes[:,1]), axis=-1) # x1, y1, x2, y2 ==> cx, cy, w, h
  
  plot_img(input_img, _boxes, classes, scores, cfg)
  return _boxes