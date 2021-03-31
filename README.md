## 

****

### train_pospal.py

- **基于research/object_detection/legacy/train.py和trainer.py修改.**

- **把所有主要函数都写在一起, 便于后面修改和调试.**

- **改进点:**
  1)  把tf.contrib.slim.learning.train中训练主函数显式拉出来写, 更直观, 便于修改调试.
  2)  在训练主函数中加入自定义测试指定数据集功能.

- **使用方法**

  ```shell
  CUDA_VISIBLE_DEVICES=$GPU_IDS python train_pospal.py \
  --train_dir $OUTPUT_DIR \
  --addicfg_path $ADDICFG_FILE \
  --pipeline_config_path configs/$CONFIG_FILE \
  --num_clones $NUM_CLONES
  ```

  其中, `$ADDICFG_FILE`是额外的配置文件(`.py`)文件, 如果没有指定, 默认使用与`train_pospal.py`同目录的`addi_config.py`.

- **关于自定义测试功能具体使用, 可以查看`addi_config.py`里面的各个参数.**



Update on 2021-03-31 by HuangYing.
