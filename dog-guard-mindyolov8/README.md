# 基于Mindspore的mindyolo训练Stanford-dogs数据集
# 实例规格
华为云model arts平台使用mindyolo的yolov8m训练自己的数据集

- 实例规格:Ascend: 1*ascend-snt9b1|ARM: 24核 192GB
- 镜像:ms2.7.1-cann8.2rc1:v3
# 配置环境
# 拉取mindyolo
```
git clone https://github.com/mindspore-lab/mindyolo.git
cd mindyolo
```

## 修改requirements.txt文件内容

```
cp /home/ma-user/work/dog-guard-mindyolov8/requirements.txt /home/ma-user/work/mindyolo/requirements.txt
```

## 安装依赖

```
pip install -r /home/ma-user/work/mindyolo/requirements.txt
```

## 如果numpy版本不匹配，执行下面命令

```
pip install numpy==1.26.4 --force-reinstall
```

# 处理数据集

## 进入工作目录

```
cd /home/ma-user/work
```

## 克隆仓库元数据（不下载文件）

```
git clone --filter=blob:none --no-checkout https://github.com/OliLov/Stanford-Dogs-YOLO.git
```

## 进入项目目录

```
cd Stanford-Dogs-YOLO
```

## 启用稀疏检出（只拉取指定路径）

```
git sparse-checkout init --cone
git sparse-checkout set YOLO
```

## 检出文件（开始下载所有图像和标签）20min左右

```
git checkout
```

## 创建目标目录结构

```
DATASET_DIR="/home/ma-user/work/mindyolo/data/stanford_dogs"
mkdir -p $DATASET_DIR/{images,labels}/{train,val,test}
```

## 复制训练集

```
cp /home/ma-user/work/Stanford-Dogs-YOLO/YOLO/Train/*.jpg $DATASET_DIR/images/train/
cp /home/ma-user/work/Stanford-Dogs-YOLO/YOLO/Train/*.txt $DATASET_DIR/labels/train/
```

## 复制验证集

```
cp /home/ma-user/work/Stanford-Dogs-YOLO/YOLO/Validation/*.jpg $DATASET_DIR/images/val/
cp /home/ma-user/work/Stanford-Dogs-YOLO/YOLO/Validation/*.txt $DATASET_DIR/labels/val/
```

## 复制测试集

```
cp /home/ma-user/work/Stanford-Dogs-YOLO/YOLO/Test/*.jpg $DATASET_DIR/images/test/
cp /home/ma-user/work/Stanford-Dogs-YOLO/YOLO/Test/*.txt $DATASET_DIR/labels/test/
```

# 处理相关文件

```
cd /home/ma-user/work/mindyolo
```

## 生成 val.txt 文件：

```
find data/stanford_dogs/images/val -name '*.jpg' > data/stanford_dogs/val.txt
```

## 生成 train.txt 文件：

```
find data/stanford_dogs/images/train -name '*.jpg' > data/stanford_dogs/train.txt
```

## 修改dataset.py代码

```
sed -i 's/self\.imgIds = \[int(Path(im_file)\.stem) for im_file in self\.img_files\]/self.imgIds = list(range(len(self.img_files)))/' mindyolo/data/dataset.py
```

## 替换yaml配置文件和运行文件train.py

```
cp /home/ma-user/work/dog-guard-mindyolov8/train.py /home/ma-user/work/mindyolo/train.py
cp /home/ma-user/work/dog-guard-mindyolov8/yolov8m.yaml /home/ma-user/work/mindyolo/configs/yolov8/yolov8m.yaml
cp /home/ma-user/work/dog-guard-mindyolov8/yolov8-base.yaml /home/ma-user/work/mindyolo/configs/yolov8/yolov8-base.yaml
```


# 运行命令

```
cd /home/ma-user/work/mindyolo
```

```
python train.py \
  --config configs/yolov8/yolov8m.yaml \
  --epochs 10 \
  --per_batch_size 2 \
  --device_target Ascend \
  --ms_mode 0 \
  --run_eval False \
  --save_dir ./outputs/stanford_dogs_yolov8m_outputs \
  --log_interval 50 \
  --keep_checkpoint_max 1
```

 

