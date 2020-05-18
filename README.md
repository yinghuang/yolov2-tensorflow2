YOLOv2在Tensorflow2.0上的复现  
aaa  
bbb  
- 原理解析见[博客](https://blog.csdn.net/ying86615791/article/details/102957513)
- 代码参考[jmpap/YOLOV2-Tensorflow-2.0](https://github.com/jmpap/YOLOV2-Tensorflow-2.0)
- 1.准备数据
	- 1.1 下载PASCAL VOC， [2017 train and val](http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar)， [2017 test](http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar)， [2012 train and val](http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar)。
	- 1.2 全部解压，出现文件夹VOCdevkit，数据都在里面。
	- 1.3 main.py里，data_dir为数据的VOCdevkit路径，如果使用tfrecord，需要在create_tfrecords.py（该文件参考tf官方的[object_detectionAPI](https://github.com/tensorflow/models/blob/1af55e018eebce03fb61bba9959a04672536107d/research/object_detection/dataset_tools/create_pascal_tf_record.py)）里面分别生成train和val的tfrecord文件。如果使用实时地从硬盘读取图片，那么val_set指定验证集（测试集），train_set指定训练集。
- 2.dataet构造。有两种方式，选择第二种更方便。
	- 2.1 第一种，通过生成tfrecord。create_tfrecords.py可以生成tfrecord数据文件。main.py里面，以下代码是相关代码：
		```python
		...
		dset_train_path = 'data/train.tfrecord'
		dset_val_path = 'data/val.tfrecord'
		...
		dset_train = OB_tfrecord_dataset(dset_train_path, batch_size, cfg, shuffle=False)
		dset_val = OB_tfrecord_dataset(dset_val_path, batch_size, cfg, shuffle=True)
		...
		```
	- 2.2 第二种，通过tf.data.Dataset.from_tensor_slices实时地从硬盘读取图片。由于每张图片可能存在多个目标，需要提前读取所有图片的标签。main.py里面，以下是相关代码：
		```python
		val_set = {'VOC2012':['val']}
		train_set = {'VOC2007':['train', 'val', 'test'], 'VOC2012':['train']}
		...
		dset_train = OB_tensor_slices_dataset(data_dir, train_set, batch_size, cfg, shuffle=True)
		dset_val = OB_tensor_slices_dataset(data_dir, val_set, batch_size, cfg, shuffle=False)
		...
		```
- 3.下载官方提供的预训练模型
	- 3.1 [yolov2-voc.weights](https://pjreddie.com/media/files/yolov2-voc.weights)是官方提供的在VOC 2007+2012上训练的标准模型权重，[yolov2-tiny-voc.weights](https://pjreddie.com/media/files/yolov2-tiny-voc.weights)则是使用tiny网络结构。更多相关模型信息可以查看[这里](https://pjreddie.com/darknet/yolov2/)。
	- 3.2 将下载好的weights文件放在main.py中model_weights_path指示的路径。
- 4.训练模型
	- 4.1 main.py中weight_reader是载入模型权重的方法，iflast表示是否载入最后一层卷积层（输出的分类回归层）。
	- 4.2 数据和模型都放好后，并且main.py里面的各个路径设置好后，运行main.py就可以训练了。main.py最下面是模型测试。
- 5.未来
	- 5.1 加入mAP，Recall评价指标，比如[这里](https://github.com/Cartucho/mAP)
- 最新发现，YOLOv2和v3的xywh loss都是在t尺度上计算的，也就是网络原始的输出，还没经过坐标变换【在main.py的yolov2_loss中已更改】。[参考](https://www.zhihu.com/question/357005177)
		
