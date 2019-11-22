"""
Faster RCNN中RPN的anchor boxes生成
参考
https://blog.csdn.net/u014380165/article/details/80379812#commentBox
https://blog.csdn.net/sinat_33486980/article/details/81099093#commentBox
"""
import numpy as np

#用于返回width,height,(x,y)中心坐标(对于一个anchor窗口)  
def _whctrs(anchor):  
    """ 
    Return width, height, x center, and y center for an anchor (window). 
    """  
    #anchor:存储了窗口左上角，右下角的坐标  
    w = anchor[2] - anchor[0] + 1  
    h = anchor[3] - anchor[1] + 1  
    x_ctr = anchor[0] + 0.5 * (w - 1)  #anchor中心点坐标  
    y_ctr = anchor[1] + 0.5 * (h - 1)  
    return w, h, x_ctr, y_ctr

#给定一组宽高向量, 以及中心点坐标, 输出各个anchor, 即预测窗口. **输出anchor的面积相等，只是宽高比不同**  
def _mkanchors(ws, hs, x_ctr, y_ctr):  
    #ws:[23 16 11]，hs:[12 16 22],ws和hs一一对应。  
    """ 
    Given a vector of widths (ws) and heights (hs) around a center 
    (x_ctr, y_ctr), output a set of anchors (windows). 
    """  
    ws = ws[:, np.newaxis]  #newaxis:将数组转置  
    hs = hs[:, np.newaxis]  
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),    #hstack、vstack:合并数组  
                         y_ctr - 0.5 * (hs - 1),    #anchor：[[-3.5 2 18.5 13]  
                         x_ctr + 0.5 * (ws - 1),     #        [0  0  15  15]  
                         y_ctr + 0.5 * (hs - 1)))     #       [2.5 -3 12.5 18]]  
    return anchors 

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)  #返回宽高和中心坐标，w:16,h:16,x_ctr:7.5,y_ctr:7.5
    size = w * h   #size:16*16=256
    size_ratios = size / ratios  #256/ratios[0.5,1,2]=[512,256,128]
    #round()方法返回x的四舍五入的数字，sqrt()方法返回数字x的平方根
    ws = np.round(np.sqrt(size_ratios)) #ws:[23 16 11]
    hs = np.round(ws * ratios)    #hs:[12 16 22],ws和hs一一对应。as:23&12
    #给定一组宽高向量, 和中心点坐标, 输出各个预测窗口. 也就是将（宽，高，中心点横坐标，中心点纵坐标）的形式，转成
    #四个坐标值(左上角x, 左上角y, 右下角x, 右下角y)的形式
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)  
    return anchors

#枚举一个anchor的各种尺度，以anchor[0 0 15 15]为例,scales[8 16 32]  
def _scale_enum(anchor, scales):  
    """   列举关于一个anchor的三种尺度 128*128,256*256,512*512 
    Enumerate a set of anchors for each scale wrt an anchor. 
    """  
    w, h, x_ctr, y_ctr = _whctrs(anchor) #返回宽高和中心坐标，w:16,h:16,x_ctr:7.5,y_ctr:7.5
    ws = w * scales   #[128 256 512]  
    hs = h * scales   #[128 256 512]  
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr) #[[-56 -56 71 71] [-120 -120 135 135] [-248 -248 263 263]]  
    return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
 
    base_anchor = np.array([1, 1, base_size, base_size]) - 1 # [0,0, 15, 15]表示[左上角x, 左上角y, 右下角x, 右下角y]
    print ("base anchors\n",base_anchor)
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    print ("anchors after ratio\n",ratio_anchors)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    print ("achors after ration and scale\n",anchors)
    return anchors


feat_stride = 16 # 缩放系数, feature map上1点表示原始图多少个像素
scales=[8, 16, 32]
ratios=[0.5, 1, 2]
im_info = [600, 900] # 原图高 宽?
feat_height = 600 / feat_stride
feat_width = 900 / feat_stride
allowed_border = 0 # 把坐标小于这个的anchor过滤掉, 就是过滤掉坐标超出原图的框
gt_boxes = '?'

#======================================================== 1. 生成特征图的anchors(特征图1个点对应K个anchor boxes, 它们中心点坐标一样, 都是对应原图区域的中心点)
# 生成feature map上左上角(0,0)像素点的K个anchors
base_anchors = generate_anchors(base_size=feat_stride, ratios=np.array(ratios), scales=np.array(scales))
num_anchors = base_anchors.shape[0]


shift_x = np.arange(0, feat_width) * feat_stride
shift_y = np.arange(0, feat_height) * feat_stride
shift_x, shift_y = np.meshgrid(shift_x, shift_y)
shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose() # feature_map上每个像素点相对于第一个像素点

# A默认是9
A = num_anchors
# K其实就是该层feature map的宽*高，比如高是38，宽是50，那么K就是1900。
# 注意all_anchors最后一个维度是4，表示4个坐标相关的信息。
K = shifts.shape[0]
all_anchors = base_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
all_anchors = all_anchors.reshape((K * A, 4))



# only keep anchors inside the image
# inds_inside表示anchor的4个点坐标都在图像内部的anchor的index。
inds_inside = np.where((all_anchors[:, 0] >= -allowed_border) &
                       (all_anchors[:, 1] >= -allowed_border) &
                       (all_anchors[:, 2] < im_info[1] + allowed_border) &
                       (all_anchors[:, 3] < im_info[0] + allowed_border))[0]


# keep only inside anchors
# 将不完全在图像内部（初始化的anchor的4个坐标点超出图像边界）的anchor都过滤掉，
# 一般过滤后只会有原来1/3左右的anchor。如果不将这部分anchor过滤，则会使训练过程难以收敛。
anchors = all_anchors[inds_inside, :]
#======================================================== 2. 给anchor boxes打标签
# label: 1 is positive, 0 is negative, -1 is dont care
# 前面得到的只是anchor的4个坐标信息，接下来就要为每个anchor分配标签了，
# 初始化的时候标签都用-1来填充，-1表示无效，这类标签的数据不会对梯度更新起到帮助。
labels = np.empty((len(inds_inside),), dtype=np.float32)
labels.fill(-1)


if gt_boxes.size > 0:
    # overlap between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(anchors.astype(np.float), gt_boxes.astype(np.float)) # 这里是计算每个anchor和每个object的IOU, [n_anchor, m_object]
    argmax_overlaps = overlaps.argmax(axis=1) # 计算每个anchor和哪个object的IOU最大, [n_anchor, 1]
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] # 每个anchor对应的最大的具体的IOU值
    gt_argmax_overlaps = overlaps.argmax(axis=0) # 计算每个object和哪个anchor的IOU最大, [m_object, 1]
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])] # 每个gt对应的最大的具体的IOU值
    # 博客原文: 因为如果有多个anchor和某个object的IOU值都是最大且一样, 那么gt_argmax_overlaps只会得到index最小的那个
    # 个人理解: 因为如果多个object和某个anchor的IOU值都是最大且一样的, 那么gt_argmax_overlaps只会得到index最小的那个
    # 所以需要gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]将IOU最大的那些anchor都捞出来
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]  # [m_object,1]???


# 这个条件语句默认是执行的，目的是将IOU小于某个阈值的anchor的标签都标为0，也就是背景类。
# 阈值config.TRAIN.RPN_NEGATIVE_OVERLAP默认是0.3。
# 如果某个anchor和所有object的IOU的最大值比这个阈值小，那么就是背景。
    if not config.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
# 有两种类型的anhor其标签是1，标签1表示foreground，也就是包含object。
# 第一种是和任意一个object有最大IOU的anchor，也就是前面得到的gt_argmax_overlaps。
    labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IoU
# 第二种是和所有object的IOU的最大值超过某个阈值的anchor，
# 其中阈值config.TRAIN.RPN_POSITIVE_OVERLAP默认是0.7。
    labels[max_overlaps >= config.TRAIN.RPN_POSITIVE_OVERLAP] = 1

# 这一部分是和前面if not config.TRAIN.RPN_CLOBBER_POSITIVES条件语句互斥的，
# 区别在于背景类anchor的标签定义先后顺序不同，这主要涉及到标签1和标签0之间的覆盖。
    if config.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < config.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
else:
# 如果ground truth中没有object，则所有标签都是背景。
    labels[:] = 0