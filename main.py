# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import tensorflow.keras.backend as K
from dataset import OB_tfrecord_dataset, OB_tensor_slices_dataset, transform
from utils import display_img, save_best_weights, log_loss
from model import yolov2, yolov2_tiny, WeightReader, cal_iou
from config import Config
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

  
def yolov2_loss(detector_mask, y_true_anchor_boxes, y_true_class_hot, y_true_boxes_all, y_pred, cfg):
    """
    Calculate YOLO V2 loss from prediction (y_pred) and ground truth tensors
    (detector_mask, y_true_anchor_boxes, y_true_class_hot, y_true_boxes_all,)
        
    Parameters
    ----------
    - detector_mask : tensor, shape (batch, size, GRID_W, GRID_H, anchors_count, 1)
          1 if bounding box detected by grid cell, else 0
    - y_true_anchor_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
          Contains adjusted coords of bounding box in YOLO format
    - y_true_class_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
          One hot representation of bounding box label
    - y_true_boxes_all : annotations : tensor (shape : batch_size, max annot, 5)
          y_true_boxes_all format : x, y, w, h, c (coords unit : grid cell)
    - y_pred : prediction from model. tensor (shape : batch_size, GRID_W, GRID_H, anchors count, (5 + labels count)
    
    Returns
    -------
    - loss : scalar
    """
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
    
    #================ 1. 坐标损失
    # 计算loss_wh损失时，需要根据gt的wh计算系数，系数作用是w和h值越小，损失系数越大，可以更好地学习尺度较小的box
    # coordinate loss
    lambda_wh = K.expand_dims(2-(y_true_anchor_boxes[:,:,:,:,2]/cfg.GRID_W) * (y_true_anchor_boxes[:,:,:,:,3]/cfg.GRID_H))
    detector_mask = K.cast(detector_mask, tf.float32) # batch_size, GRID_W, GRID_H, n_anchors, 1
    n_objs = K.sum( K.cast( detector_mask>0, tf.float32 ) )
    #=========1.1 基于坐标值计算坐标损失
#    loss_xy = cfg.LAMBDA_COORD * K.sum( detector_mask * K.square( y_true_anchor_boxes[:,:,:,:,0:2] - pred_xy)) / (n_objs + 1e-6)
#    loss_wh = cfg.LAMBDA_COORD * K.sum( lambda_wh * detector_mask * K.square( y_true_anchor_boxes[:,:,:,:,2:4] - pred_wh)) / (n_objs + 1e-6)
#    #  loss_wh = cfg.LAMBDA_COORD * K.sum(detector_mask * K.square(K.sqrt(y_true_anchor_boxes[...,2:4]) - 
#    #                                                            K.sqrt(pred_wh))) / (n_objs + 1e-6)
    #=========1.2 基于预测值计算坐标损失
    y_txy = y_true_anchor_boxes[...,0:2] - cell_coords
    y_twh = K.log(y_true_anchor_boxes[...,2:4]*1.0/anchors + 1e-16)
    pred_txy = K.sigmoid(y_pred[:,:,:,:,0:2])
    pred_twh = y_pred[:,:,:,:,2:4]
    loss_xy = cfg.LAMBDA_COORD * K.sum( detector_mask * K.square( y_txy - pred_txy)) / (n_objs + 1e-6)
    loss_wh = cfg.LAMBDA_COORD * K.sum( lambda_wh * detector_mask * K.square( y_twh - pred_twh)) / (n_objs + 1e-6)
    
    
    loss_coord = loss_xy + loss_wh
  
    #================ 2. 类别损失
    pred_class = K.softmax(y_pred[:,:,:,:,5:])
    #  y_true_class = tf.argmax(y_true_class_hot, -1)
    #  loss_cls = K.sparse_categorical_crossentropy(target=y_true_class, output=pred_class, from_logits=True)
    #  loss_cls = K.expand_dims(loss_cls, -1) * detector_mask
    loss_cls = detector_mask * K.square( y_true_class_hot - pred_class )
    loss_cls = cfg.LAMBDA_CLASS * K.sum(loss_cls) / (n_objs + 1e-6)
    
    #================ 3. bbox置信度损失
    #================ 3.1. 包含目标的预测的bbox置信度损失
    # for each detector : iou between prediction and ground truth
    x1 = y_true_anchor_boxes[...,0]
    y1 = y_true_anchor_boxes[...,1]
    w1 = y_true_anchor_boxes[...,2]
    h1 = y_true_anchor_boxes[...,3]
    x2 = pred_xy[...,0]
    y2 = pred_xy[...,1]
    w2 = pred_wh[...,0]
    h2 = pred_wh[...,1]
    ious = cal_iou(x1, y1, w1, h1, x2, y2, w2, h2)
    ious = K.expand_dims(ious, -1)
    
    # 在线计算预测的box和gtbox的iou作为置信度的target
    # object confidence loss
    pred_conf = K.sigmoid(y_pred[...,4:5])
    loss_conf_obj = cfg.LAMBDA_OBJECT * K.sum(detector_mask * K.square(ious - pred_conf)) / (n_objs + 1e-6)
    
    
    #================ 3.2. 不包含目标的预测的bbox置信度损失
    # xmin, ymin, xmax, ymax of pred bbox
    pred_xy = K.expand_dims(pred_xy, 4) # shape : batch_size, GRID_W, GRID_H, n_anchors, 1, 2 
    pred_wh = K.expand_dims(pred_wh, 4)
    pred_wh_half = pred_wh / 2.
    pred_mins = pred_xy - pred_wh_half
    pred_maxes = pred_xy + pred_wh_half
    
    # xmin, ymin, xmax, ymax of true bbox
    true_boxe_shape = K.int_shape(y_true_boxes_all)
    true_boxes_grid = K.reshape(y_true_boxes_all, [true_boxe_shape[0], 1, 1, 1, true_boxe_shape[1], true_boxe_shape[2]])
    true_xy = true_boxes_grid[...,0:2] # shape: batch_size, 1, 1, 1, max_annot, 2
    true_wh = true_boxes_grid[...,2:4] # shape: batch_size, 1, 1, 1, max_annot, 2
    true_wh_half = true_wh * 0.5
    true_mins = true_xy - true_wh_half
    true_maxes = true_xy + true_wh_half
    
    # 从预测的box中，计算每一个box与所有GTbox的IOU，找出最大的IOU，如果小于阈值(0.6,并且不负责GT，根据1 - detector_mask)，该预测的box就加入noobj，计算置信度损失
    intersect_mins = K.maximum(pred_mins, true_mins) # shape : batch_size, GRID_W, GRID_H, n_anchors, max_annot, 2 
    intersect_maxes = K.minimum(pred_maxes, true_maxes) # shape : batch_sizem, GRID_W, GRID_H, n_anchors, max_annot, 2
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.) # shape : batch_size, GRID_W, GRID_H, n_anchors, max_annot, 1
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1] # shape : batch_size, GRID_W, GRID_H, n_anchors, max_annot, 1
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1] # shape : batch_size, GRID_W, GRID_H, n_anchors, 1, 1
    true_areas = true_wh[..., 0] * true_wh[..., 1] # shape : batch_size, GRID_W, GRID_H, n_anchors, max_annot, 1
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas # shape : batch_size, GRID_W, GRID_H, n_anchors, max_annot, 1
    best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
    best_ious = K.expand_dims(best_ious) # shape : batch_size, GRID_W, GRID_H, n_anchors, 1
    
    # no object confidence loss
    no_object_detection = K.cast(best_ious < 0.6, K.dtype(best_ious)) 
    noobj_mask = no_object_detection * (1 - detector_mask)
    n_noobj  = K.sum(tf.cast(noobj_mask  > 0.0, tf.float32))
    
    loss_conf_noobj =  cfg.LAMBDA_NOOBJECT * K.sum(noobj_mask * K.square(-pred_conf)) / (n_noobj + 1e-6)
    
    #================ 4. 三种损失汇总
    # total confidence loss
    loss_conf = loss_conf_noobj + loss_conf_obj
    
    # total loss
    loss = loss_conf + loss_cls + loss_coord
    sub_loss = [loss_conf, loss_cls, loss_coord] 
    return loss, sub_loss


  


#=============== prepare config and data
data_dir = r'D:\dataset\pascal_voc\VOCdevkit'
val_set = {'VOC2012':['val']}
train_set = {'VOC2007':['train', 'val', 'test'], 'VOC2012':['train']}

dset_train_path = 'data/train.tfrecord'
dset_val_path = 'data/val.tfrecord'
batch_size = 6
num_epochs = 30
num_iters = 30
train_name = 'training_1'
# --------------yolov2
#cfg = Config()
#model_weights_path = 'weights/yolov2-voc.weights'
# --------------yolov2-tiny
cfg = Config(yolo_tiny=True)
model_weights_path = 'weights/yolov2-tiny-voc.weights'

n_progress = 20 # 进度条分为多少份。如果不够就向上取整

# log (tensorboard)
summary_writer = tf.summary.create_file_writer(os.path.join('logs/', train_name), flush_millis=20000)
summary_writer.set_as_default()

# 用tfrecord的形式读取数据集
#dset_train = OB_tfrecord_dataset(dset_train_path, batch_size, cfg, shuffle=False)
#dset_val = OB_tfrecord_dataset(dset_val_path, batch_size, cfg, shuffle=True)

# 用tf.data.Dataset.from_tensor_slices的形式, 每次即时的从硬盘读取图片(首先会生成所有图片的Ground Truth)
dset_train = OB_tensor_slices_dataset(data_dir, train_set, batch_size, cfg, shuffle=True)
dset_val = OB_tensor_slices_dataset(data_dir, val_set, batch_size, cfg, shuffle=False)
len_batches_train = tf.data.experimental.cardinality(dset_train).numpy()
len_batches_val = tf.data.experimental.cardinality(dset_val).numpy()

#=============== prepare model
weight_reader = WeightReader(model_weights_path)
# --------------yolov2
#model = yolov2(cfg) # 一共23个卷积层包括最后一层
#weight_reader.load_weights(model, num_convs=23, iflast=False)
# --------------yolov2-tiny
model = yolov2_tiny(cfg) # 一共23个卷积层包括最后一层
weight_reader.load_weights(model, num_convs=9, iflast=False)


#=============== prepare training
train_loss_history = []
val_loss_history = []
best_val_loss = 1e6
initial_learning_rate = 2e-5
decay_epochs = 30 * num_iters
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(	
    initial_learning_rate,	
    decay_steps=decay_epochs,	
    decay_rate=0.5,	
    staircase=True)
#lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
#    [2,4], [2e-5,1e-5,2e-5])
optim = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#optim = tf.keras.optimizers.SGD(learning_rate=1e-4)
# --------------yolov2
#train_layers = ['conv_22', 'norm_22', 'conv_23']
# --------------yolov2-tiny
train_layers = ['conv_8', 'norm_8', 'conv_9']
train_vars = []
for name in train_layers:
     train_vars += model.get_layer(name).trainable_variables 
#raise ValueError

#=============== begin training
for epoch in range(num_epochs):
    epoch_loss = []
    epoch_sub_loss = []
    epoch_val_loss = []
    epoch_val_sub_loss = []
    print('\nEpoch {} :'.format(epoch))
    
    # train
#    pbar = tqdm(total=len_batches_train)
    for bs_idx, (x,y) in enumerate(dset_train):
        x, detector_mask, y_true_anchor_boxes, y_true_class_hot, y_true_boxes_all = transform(x,y, cfg)
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss, sub_loss = yolov2_loss(detector_mask, y_true_anchor_boxes, y_true_class_hot, 
                                         y_true_boxes_all, y_pred, cfg)
            _loss = loss * 0.01
        grads = tape.gradient(_loss, train_vars)
        optim.apply_gradients(grads_and_vars = zip(grads, train_vars))
        epoch_loss.append(loss)
        epoch_sub_loss.append(sub_loss)
        if (bs_idx+1)%( math.ceil(num_iters/n_progress) ) == 0:
            print('-', end='')
#        pbar.update(1)
        if (bs_idx+1)==num_iters: break
#    pbar.close()
        
    # val
#    pbar = tqdm(total=len_batches_val)
    for bs_idx, (x,y) in enumerate(dset_val):
        x, detector_mask, y_true_anchor_boxes, y_true_class_hot, y_true_boxes_all = transform(x,y, cfg)
        with tf.GradientTape() as tape:
            y_pred = model(x, training=False)
            loss, sub_loss = yolov2_loss(detector_mask, y_true_anchor_boxes, y_true_class_hot, 
                                         y_true_boxes_all, y_pred, cfg)
        epoch_val_loss.append(loss)
        epoch_val_sub_loss.append(sub_loss)
        print('-', end='')
#        pbar.update(1)
        if (bs_idx+1)==1: break
#    pbar.close()
    
    # record
    loss_avg = np.mean(np.array(epoch_loss))
    sub_loss_avg = np.mean(np.array(epoch_sub_loss), axis=0)
    val_loss_avg = np.mean(np.array(epoch_val_loss))
    val_sub_loss_avg = np.mean(np.array(epoch_val_sub_loss), axis=0)
    
    log_loss(loss_avg, val_loss_avg, step=epoch)
    train_loss_history.append(loss_avg)
    val_loss_history.append(val_loss_avg)
    
    if loss_avg < best_val_loss:
        print('\nfind better model for train')
        best_model_path = save_best_weights(model, train_name+'_epoch%d'%(epoch), loss_avg)
        best_val_loss = loss_avg
    print(' \ntrain_loss={:.3f} (conf={:.3f}, class={:.3f}, coords={:.3f}), val_loss={:.3f} (conf={:.3f}, class={:.3f}, coords={:.3f})'.format(
            loss_avg, sub_loss_avg[0], sub_loss_avg[1], sub_loss_avg[2],
            val_loss_avg, val_sub_loss_avg[0], val_sub_loss_avg[1], val_sub_loss_avg[2]))
      
save_best_weights(model, train_name+'_epoch%dfinal'%(epoch), 666)

#=============== begin testing
model.load_weights(best_model_path)
for bs_idx, (x,y) in enumerate(dset_val):
    x, detector_mask, y_true_anchor_boxes, y_true_class_hot, y_true_boxes_all = transform(x,y, cfg)
    
    score_threshold = 0.5 # 根据box_conf * box_class_prob筛选boxes
    iou_threshold = 0.45 # 对boxes做NMS时的阈值
    display_img(x[0], model, score_threshold, iou_threshold, cfg)
    if (bs_idx+1)==10: break

fig,ax = plt.subplots(1, figsize=(8,8))
ax.plot(train_loss_history[10:])
plt.show()