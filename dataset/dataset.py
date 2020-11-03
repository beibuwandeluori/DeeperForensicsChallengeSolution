import numpy as np
import os
import time
import sys
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import *
from albumentations.pytorch import ToTensor, ToTensorV2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue, Normalize, RandomBrightnessContrast,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize, ImageCompression, Rotate,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, Cutout, GridDropout,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip, GaussianBlur, CoarseDropout,
    PadIfNeeded, ToGray, FancyPCA)
from catalyst.data.sampler import BalanceClassSampler

try:
    from dataset.distortions import *
except:
    sys.path.append('/data1/cby/py_project/DeeperForensicsChallengeSubmissionExample/dataset')
    from distortions import *


def get_train_transforms(size=300):
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.1),
        GaussianBlur(blur_limit=3, p=0.05),
        HorizontalFlip(),
        Resize(height=size, width=size),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        OneOf([RandomBrightnessContrast(), HueSaturationValue()], p=0.5),  # FancyPCA(),
        OneOf([CoarseDropout(), GridDropout()], p=0.2),
        ToGray(p=0.2),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    )

def get_valid_transforms(size=300):
    return Compose([
            Resize(height=size, width=size, p=1.0),
            PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(p=1.0),
        ], p=1.0)

def one_hot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def get_file(path, format='png'):
    files = os.listdir(path)  # 得到文件夹下的所有文件，包含文件夹名称
    FileList = []
    for name in files:
        if os.path.isdir(os.path.join(path, name)):
            FileList.extend(get_file(os.path.join(path, name), format))  #回调函数，对所有子文件夹进行搜索
        elif os.path.isfile(os.path.join(path, name)):
            if format.lower() in name.lower():
                FileList.append(os.path.join(path, name))
        else:
            print("未知文件:%s", name)
    return FileList

def load_image_file_paths(real_root_path='/raid/chenby/DeepForensics/face_images/source_images',
                          real_target_root_path='/raid/chenby/DeepForensics/face_images/target_images/',
                          fake_root_path='/raid/chenby/DeepForensics/face_images/manipulated_images',
                          split_root_path='/raid/chenby/DeepForensics/face_images/lists/splits',
                          data_type='train'):
    txt_path = os.path.join(split_root_path, data_type + '.txt')
    f = open(txt_path, "r")
    lines = f.readlines()
    print(len(lines))
    real_video_paths = []
    real_target_video_paths = []
    fake_video_paths = []

    for line in lines:
        line = line.strip().replace("\n", "")
        fake_subsets = os.listdir(fake_root_path)
        for fake_subset in fake_subsets:
            fake = os.path.join(fake_root_path, fake_subset, line)
            fake_video_paths.append(fake)

        real_id_target = line.split('.')[0].split('_')[0] + '.mp4'
        real_target_video_paths.append(os.path.join(real_target_root_path, 'c23', real_id_target))
        real_target_video_paths.append(os.path.join(real_target_root_path, 'c40', real_id_target))

        real_id = line.split('.')[0].split('_')[1]
        real = os.path.join(real_root_path, real_id)
        real_video_paths.append(real)

    # print('fake:', len(fake_video_paths), 'real:', len(real_video_paths), real_video_paths[0])
    real_video_paths = set(real_video_paths)
    real_target_video_paths = set(real_target_video_paths)
    fake_video_paths = set(fake_video_paths)

    fake_image_paths = []
    for fake_video_path in tqdm(fake_video_paths):
        images = os.listdir(fake_video_path)
        images = [os.path.join(fake_video_path, image) for image in images]
        fake_image_paths += images

    real_image_paths = []
    for real_video_path in tqdm(real_video_paths):
        images = get_file(real_video_path, format='png')
        real_image_paths += images
    print(len(real_image_paths))

    for target_path in tqdm(real_target_video_paths):
        images = sample_real_target_images(path=target_path, num_samples=4)
        real_image_paths += images

    print('Images fake:', len(fake_image_paths), 'real:', len(real_image_paths))

    real_image_paths = np.array(real_image_paths)
    fake_image_paths = np.array(fake_image_paths)
    # np.save('split_npy/real_' + data_type + '.npy', real_image_paths)
    # np.save('split_npy/fake_' + data_type + '.npy', fake_image_paths)

    # return real_image_paths, fake_image_paths

# 用 ff++ 中的real c23 每个视频间隔4帧进行采样
def sample_real_target_images(path, num_samples=4):
    images = sorted(os.listdir(path))
    sample_images = []
    for img in images:
        id = int(img.split('.')[0])
        if id % num_samples == 0:
            sample_images.append(os.path.join(path, img))

    return sample_images


def load_images_from_npy(data_type='train', npy_root='/data1/cby/py_project/DeeperForensicsChallengeSubmissionExample/dataset/split_npy/'):
    real_image_paths = np.load(npy_root + 'real_' + data_type + '.npy')
    fake_image_paths = np.load(npy_root + 'fake_' + data_type + '.npy')

    return real_image_paths, fake_image_paths


def load_images_from_npys(real_npys=[], fake_npys=[]):
    for i, real in enumerate(real_npys):
        if i == 0:
            real_image_paths = np.load(real)
        else:
            real_image_paths = np.concatenate([real_image_paths, np.load(real)], axis=0)

    for i, fake in enumerate(fake_npys):
        if i == 0:
            fake_image_paths = np.load(fake)
        else:
            fake_image_paths = np.concatenate([fake_image_paths, np.load(fake)], axis=0)

    return real_image_paths, fake_image_paths


class DeeperForensicsDataset(Dataset):

    def __init__(self, data_type='train', is_one_hot=True, transforms=None, classes_num=2):
        super().__init__()
        self.classes_num = classes_num
        self.data_type = data_type
        self.transforms = transforms
        self.is_one_hot = is_one_hot
        real_img_paths, fake_img_paths = load_images_from_npy(data_type=data_type)
        print('real:', real_img_paths.shape, 'fake:', fake_img_paths.shape)
        self.images = []
        self.labels = []
        for p in real_img_paths:
            self.images.append(p)
            self.labels.append(0)
        for p in fake_img_paths:
            self.images.append(p)
            self.labels.append(1)

    def __getitem__(self, index: int):
        label = self.labels[index]
        image_name = self.images[index]
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        if self.is_one_hot:
            label = one_hot(self.classes_num, label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)

    def get_labels(self):
        return list(self.labels)


class DeeperForensicsDatasetNew(Dataset):

    def __init__(self, real_npys, fake_npys, is_one_hot=False, transforms=None, classes_num=2, data_type='train'):
        super().__init__()
        self.classes_num = classes_num
        self.transforms = transforms
        self.is_one_hot = is_one_hot
        self.data_type = data_type
        real_img_paths, fake_img_paths = load_images_from_npys(real_npys, fake_npys)
        print('real:', real_img_paths.shape, 'fake:', fake_img_paths.shape)
        self.images = []
        self.labels = []
        for p in real_img_paths:
            self.images.append(p)
            self.labels.append(0)
        for p in fake_img_paths:
            self.images.append(p)
            self.labels.append(1)

    def __getitem__(self, index: int):
        label = self.labels[index]
        image_name = self.images[index]
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)

        if self.transforms:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        else:
            if self.data_type == 'train':
                image = my_augmentation(image)
            image = cv2.resize(image, dsize=(224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255
            image = np.transpose(image, (2, 0, 1))
        if self.is_one_hot:
            label = one_hot(self.classes_num, label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)

    def get_labels(self):
        return list(self.labels)

# 自定义数据增强
def my_augmentation(img):
    type_list = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG']
    if random.random() > 0.2:
        # level_list = ['1', '2', '3', '4', '5']
        type_id = random.randint(0, 5)
        dist_type = type_list[type_id]
        dist_level = random.randint(1, 5)
        # get distortion parameter
        dist_param = get_distortion_parameter(dist_type, dist_level)
        # get distortion function
        dist_function = get_distortion_function(dist_type)
        img = dist_function(img, dist_param)
    else:
        # mixed aug
        for dist_type in type_list:
            if random.random() > 0.5:
                dist_level = random.randint(1, 5)
                dist_param = get_distortion_parameter(dist_type, dist_level)
                # get distortion function
                dist_function = get_distortion_function(dist_type)
                img = dist_function(img, dist_param)

    return img


if __name__ == '__main__':
    # load_image_file_paths(data_type='test')
    # real_image_paths, fake_image_paths = load_images_from_npy(data_type='val')
    # print(real_image_paths.shape, fake_image_paths.shape)
    # print(real_image_paths[0], fake_image_paths[0])

    # sample = sample_real_target_images(path='/raid/chenby/DeepForensics/face_images/target_images/c23/000.mp4', num_samples=4)
    # print('sample:', len(sample), sample[:5])
    real_npys = ['/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/Celeb-DF-v1_mtcnn/new/real_60frames_train.npy']
    fake_npys = ['/data1/cby/py_project/FaceForensics/dataset/splits/images_npy/Celeb-DF-v1_mtcnn/new/fake_30frames_train.npy']

    start = time.time()
    xdl = DeeperForensicsDatasetNew(real_npys=real_npys, fake_npys=fake_npys, transforms=None,  # get_train_transforms(size=224)
                                    is_one_hot=False, classes_num=2)
    # xdl = DeeperForensicsDataset(data_type='train', transforms=get_train_transforms(), is_one_hot=False)
    print('length:', len(xdl))
    train_loader = DataLoader(xdl, batch_size=16, shuffle=False, num_workers=4,
                              sampler=BalanceClassSampler(labels=xdl.get_labels(), mode="downsampling"))
    for i, (img, label) in enumerate(train_loader):
        print(i, img.shape, label.shape, label)
        if i == 10:
            break
    end = time.time()
    print('end iterate')
    print('DataLoader total time: %fs' % (end - start))
    pass
