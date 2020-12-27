# 2020“华为云杯”人工智能大赛亚军方案分享
我们的方案在初赛的线上精度达0.8386，排名（4/377）
# 方案策略总结：
* 利用滑动窗口对大尺寸影像进行切割;
* 训练过程中使用随机翻转，旋转，缩放，颜色空间变换等数据增强方法;
* 使用Lovasz+Focal做损失函数，有助于模型向iou最佳的方向优化和降低类别不均衡的影响
* 使用ImageNet预训练的EfficientNet-B5做BackBone
* 训练时使用多尺度的图片进行训练
* 测试时使用膨胀预测提高模型对边界的预测效果
* 对预测结果使用0.001作为分类阈值

# 训练与使用
1. 影像的切割
使用cut_data.py对影像进行切割，切割完成后按照9：1划分为训练用的训练集和测试集，并去除切割过程中产生的全黑影像（非标签全黑）。
```python
TARGET_W, TARGET_H = 1024, 1024  #在此处修改滑动窗口的窗口大小
STEP = 512  #在此处修改滑动窗口的步长
```
```python
data_dir = r"E:\Project\Road_split\data"  #修改大尺寸影像的路径
img_name1 = "382.png"                     #修改影像名称
img_name2 = "182.png"
label_name1 = "382_label.png"             #修改标签名称
label_name2 = "182_label.png"
```

2. 训练
使用train.py开始训练
```python
#L22
train_size = [512,480,384,256]    #修改多尺度训练时的不同尺度，可以自行添加或修改，第一个为原图尺寸大小

#L26
DATA_DIR = r'E:\Project\Road_split\RSC_Baseline\data'          #修改训练时的路径
DATA_DIR_train = r'E:\Project\Road_split\RSC_Baseline\data\train'
DATA_DIR_val = r'E:\Project\Road_split\RSC_Baseline\data\val'


#L234
train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        augmentation=get_training_augmentation,
        preprocessing=get_preprocessing(preprocessing_fn),
    )                  #自定义训练数据集，若训练时不使用多尺度，添加mode='val',且在get_training_augmentation后添加()

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        mode = 'val'
    )

#L329
ENCODER = 'efficientnet-b5'  #修改训练时的BackBone


#
preprocessing_fn = {'std': (0.16419388323474826, 0.148104289090804, 0.14687551863170334), 'mean': (0.4447633174830224, 0.4553760430163513, 0.47636535016731446)}
#训练集的RGB三通道的std值和mean值，建议使用自己训练集的std和mean值，可以使用get_std_mean.py计算或者取消注释L353-354行

    
```
训练时会自动下载ImageNet 上的预训练模型
训练过程中可以实时显示iou、Loss等指标
