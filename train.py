import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from catalyst.contrib.nn import optimizers,criterion
from torch.nn.modules.loss import _Loss
from catalyst.contrib.nn.criterion.lovasz import _lovasz_hinge
from catalyst.contrib.nn.criterion.focal import partial
from catalyst.contrib.nn.criterion.dice import BCEDiceLoss
from catalyst import metrics
import random
# from get_std_mean import get_std_mean

train_size = [512,480,384,256]     #多尺度训练的图片尺寸
# DATA_DIR = '/home/ma-user/work/RSC/data'   #数据存放根目录，best_iou.pth和best_loss.pth文件保存路径
# DATA_DIR_train = '/home/ma-user/work/RSC/data/train'  #训练集路径
# DATA_DIR_val = '/home/ma-user/work/RSC/data/val'    #验证集路径
DATA_DIR = r'E:\Project\Road_split\RSC_Baseline\data'
DATA_DIR_train = r'E:\Project\Road_split\RSC_Baseline\data\train'
DATA_DIR_val = r'E:\Project\Road_split\RSC_Baseline\data\val'

x_train_dir = os.path.join(DATA_DIR_train, 'images')
y_train_dir = os.path.join(DATA_DIR_train, 'labels')

x_valid_dir = os.path.join(DATA_DIR_val, 'images')
y_valid_dir = os.path.join(DATA_DIR_val, 'labels')


# result_dir = os.path.join(DATA_DIR, 'result')




#数据集类，管理数据加载与增实时强
class Dataset(BaseDataset):
    CLASSES = ['background', 'road']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            batch_size=2,
            augmentation=None,
            preprocessing=None,
            mode = 'train'
    ):

        self.ids = os.listdir(images_dir)
        self.maskids = []
        self.batch_size = batch_size
        self.batch_input_img_number = 1
        self.mode = mode    #'train' 模式下使用多尺度训练

        self.maskids = [os.path.splitext(f)[0]+'.png' for f in self.ids]
        self.maskids.extend(self.maskids)
        print("normal", len(self.ids))

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.maskids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        # self.class_values = [1]
        print(self.class_values)

        self.temp_aug = augmentation
        self.augmentation = augmentation    #数据增强函数
        self.preprocessing = preprocessing   #数据预处理函数

    def getFileName(self, i):
        return self.ids[i]

    def __getitem__(self, i):

        # 在同一batch下保持图片尺寸相同
        if self.batch_input_img_number%self.batch_size !=0 and self.mode!='val':
            self.augmentation = self.temp_aug(random.choice(train_size))
            self.batch_input_img_number = 0
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        #
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)

#自定义损失函数   Lovasz+FocalLoss
class SelfLoss(_Loss):

    def __init__(self,
                 ignore :int =None,
                 per_image=False,
                 reduced:bool=False,
                 gamma:float=2.0,
                 alpha:float=0.25,
                 threshold:float=0.5,
                 reduction:str='mean'
                 ):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image
        if reduced:
            self.loss_fn = partial(
                metrics.reduced_focal_loss,
                gamma=gamma,
                threshold=threshold,
                reduction=reduction,
            )
        else:
            self.loss_fn = partial(
                metrics.sigmoid_focal_loss,
                gamma=gamma,
                alpha=alpha,
                reduction=reduction,
            )
        pass

    def forward(self, logits, targets):
        lovasz = _lovasz_hinge(
            logits, targets, per_image=self.per_image, ignore=self.ignore
        )
        targets = targets.view(-1)
        logits = logits.view(-1)

        if self.ignore is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = targets != self.ignore
            logits = logits[not_ignored]
            targets = targets[not_ignored]
        focal = self.loss_fn(logits, targets)
        loss = lovasz + focal
        return loss


#图片缩放到0-1之间
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
#数据增强函数
def get_training_augmentation(crop_size = 512):
    if crop_size==train_size[0]:
        train_transform = []
    else :
        train_transform = [A.RandomCrop(height=crop_size, width=crop_size)]

    train_transform += [

        A.OneOf([
            A.HorizontalFlip(p = 1),
            A.VerticalFlip(p = 1),
            A.ShiftScaleRotate(p=1),
            A.GridDistortion(p = 1),
            A.IAAPerspective(p=1),
            A.ElasticTransform(p=1),
            A.NoOp(p = 1)
        ],p = 1),
        A.OneOf([
            A.GaussNoise(p = 1),
            A.Blur(p = 1),
            A.IAASharpen(p = 1),
            A.ISONoise(p=1),
            A.MotionBlur(p = 1),
            A.CoarseDropout(max_height=int(crop_size/25), max_width=int(crop_size/25),p = 0.4),
            A.IAAEmboss(),
            A.CLAHE(),
            A.NoOp(p = 1),
        ],p = 1),
        A.OneOf([
            A.HueSaturationValue(p=1),
            A.ChannelShuffle(p=1),
            A.RGBShift(p=1),
            A.RandomBrightnessContrast(p = 1),
            A.NoOp(p=1)
        ], p=1),


    ]
    return A.Compose(train_transform)

#验证集数据操作
def get_validation_augmentation():
    test_transform = [
        A.PadIfNeeded(train_size[0], train_size[0])   #处理图片尺寸到512大小

    ]
    return A.Compose(test_transform)


def to_tensor(x, **kwargs):
    if len(x.shape) == 3:
        return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(mask=round_clip_0_1),
        A.Normalize(mean=preprocessing_fn['mean'],std=preprocessing_fn['std']),   #RGBNormalize
        A.Lambda(image=to_tensor, mask=to_tensor)
    ]
    return A.Compose(_transform)

#获取训练集和验证集实例，转换到DataLoader
def train_test(CLASSES, preprocessing_fn, BATCH_SIZE=2,  num_workers=4):
    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        batch_size=BATCH_SIZE,
        classes=CLASSES,
        augmentation=get_training_augmentation,
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        mode = 'val'
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=True)
    return train_dataloader, valid_dataloader

#训练
def train(model, CLASSES, preprocessing_fn, BATCH_SIZE,  num_workers,LR):
    train_dataloader, valid_dataloader = train_test(CLASSES, preprocessing_fn,  BATCH_SIZE,
                                                    num_workers)

    loss = SelfLoss()
    loss.__name__='LovaszFocal'

    metrics = [
        smp.utils.metrics.IoU(),
        smp.utils.metrics.Fscore(),
    ]

    #定义优化器
    base_optimizer = optimizers.RAdam([dict(params=model.parameters(), lr=LR), ],betas=(.95, 0.999),eps = 1e-5)
    lookahead = optimizers.Lookahead(optimizer=base_optimizer,k = 5)
    #定义学习率调整
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(lookahead, T_0=6,T_mult=2, eta_min=0)

    #模型训练构建
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=base_optimizer,
        device=DEVICE,
        verbose=True,
    )
    #验证集测试构建
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )


    ious = []
    losses = []
    #写训练过程中的验证集loss和iou
    with open(os.path.join(DATA_DIR,'loss_iou.txt'),'w',encoding='utf-8') as f :
        for i in range(0, EPOCHS):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_dataloader)
            valid_logs = valid_epoch.run(valid_dataloader)

            losses.append(valid_logs['LovaszFocal'])
            ious.append(valid_logs['iou_score'])
            # if min(losses) >= valid_logs['dice_loss']:
            #保存验证集上loss最低的模型
            if min(losses) >= valid_logs['LovaszFocal']:
                torch.save(model, os.path.join(DATA_DIR,'best_loss.pth'))
                print('best loss Model saved!')
            #保存验证集上iou最高的模型
            if max(ious) <=valid_logs['iou_score']:
                torch.save(model, os.path.join(DATA_DIR,'best_iou.pth'))
                print('best iou Model saved!')
            # torch.save(model, os.path.join(DATA_DIR, 'model_val_loss_'+str(valid_logs['LovaszFocal'])+'.pth'))
            lr_scheduler.step()
            print('Now LR is ',base_optimizer.param_groups[0]['lr'])

            f.write(str(i)+' , '+str(losses[i])+' , '+str(ious[i])+'\n')




if __name__ == '__main__':

    #定义backbone
    ENCODER = 'efficientnet-b5'
    #定义预训练模型权重
    ENCODER_WEIGHTS = 'imagenet'
    #定义最后一层激活函数
    ACTIVATION = 'sigmoid'
    DEVICE = 'cuda'
    BATCH_SIZE = 2     #batch大小
    CLASSES = ['background', 'road']
    LR = 0.0008
    EPOCHS = 200
    num_workers = 4
    n_classes = 3


    torch.cuda.empty_cache()
    #模型初始化
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=n_classes,
        activation=ACTIVATION,
        decoder_attention_type='scse'
    )
    #计算RGB三通道的std和mean
    # preprocessing_fn = get_std_mean(x_train_dir)
    # print(preprocessing_fn)
#     preprocessing_fn = {'std': (0.17326175387045314, 0.1576403750961563, 0.15749221318747858), 'mean': (0.42765062173714935, 0.4377896089745295, 0.457827423197792)}
    preprocessing_fn = {'std': (0.16419388323474826, 0.148104289090804, 0.14687551863170334), 'mean': (0.4447633174830224, 0.4553760430163513, 0.47636535016731446)}

    # 训练用的
    train(model,CLASSES, preprocessing_fn,  BATCH_SIZE, num_workers,LR)
    import moxing as mox

    mox.file.copy_parallel('/home/ma-user/work/RSC/data/best_loss.pth',
                           'obs://obs-road-seg/model_checkpoint/best_loss_12_7.pth')
    mox.file.copy_parallel('/home/ma-user/work/RSC/data/best_iou.pth',
                           'obs://obs-road-seg/model_checkpoint/best_iou_12_7.pth')
