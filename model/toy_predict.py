import sys

sys.path.append('..')

from eval_kit.detector import DeeperForensicsDetector
from model.models import model_selection, get_efficientnet
import torch
import time
import glob
from PIL import Image
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, extract_face
import torch.nn as nn
from model.face_detector import DetectionPipeline
from albumentations.pytorch import ToTensorV2
from albumentations import Compose, Normalize, Resize, PadIfNeeded
import cv2
import numpy as np

def get_valid_transforms(size=300):
    return Compose([
        Resize(height=size, width=size, p=1.0),
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # Resize(height=224, width=224, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


resnet_default_data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class Test_time_agumentation(object):

    def __init__(self, is_rotation=True):
        self.is_rotation = is_rotation

    def __rotation(self, img):
        """
        clockwise rotation 90 180 270
        """
        img90 = img.rot90(-1, [2, 3]) # 1 逆时针； -1 顺时针
        img180 = img.rot90(-1, [2, 3]).rot90(-1, [2, 3])
        img270 = img.rot90(1, [2, 3])
        return [img90, img180, img270]

    def __inverse_rotation(self, img90, img180, img270):
        """
        anticlockwise rotation 90 180 270
        """
        img90 = img90.rot90(1, [2, 3]) # 1 逆时针； -1 顺时针
        img180 = img180.rot90(1, [2, 3]).rot90(1, [2, 3])
        img270 = img270.rot90(-1, [2, 3])
        return img90, img180, img270

    def __flip(self, img):
        """
        Flip vertically and horizontally
        """
        return [img.flip(2), img.flip(3)]

    def __inverse_flip(self, img_v, img_h):
        """
        Flip vertically and horizontally
        """
        return img_v.flip(2), img_h.flip(3)

    def tensor_rotation(self, img):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__rotation(img)

    def tensor_inverse_rotation(self, img_list):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_rotation(img_list[0], img_list[1], img_list[2])

    def tensor_flip(self, img):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__flip(img)

    def tensor_inverse_flip(self, img_list):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_flip(img_list[0], img_list[1])


# 9 times
def TTA(model_, img, activation=nn.Softmax(dim=1)):
    # original 1
    outputs = activation(model_(img))
    tta = Test_time_agumentation()
    # 水平翻转 + 垂直翻转 2
    flip_imgs = tta.tensor_flip(img)
    for flip_img in flip_imgs:
        outputs += activation(model_(flip_img))
    # 2*3=6
    for flip_img in [img, flip_imgs[0]]:
        rot_flip_imgs = tta.tensor_rotation(flip_img)
        for rot_flip_img in rot_flip_imgs:
            outputs += activation(model_(rot_flip_img))

    outputs /= 9

    return outputs

def preprocess_image(images, device):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    #     print(images.shape)
    preprocessed_images = None
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = resnet_default_data_transforms['test']
    for image in images:
        preprocessed_image = preprocess(Image.fromarray(image))
        # Add first dimension as the network expects a batch
        preprocessed_image = preprocessed_image.unsqueeze(0)
        if preprocessed_images is None:
            preprocessed_images = preprocessed_image
        else:
            preprocessed_images = torch.cat([preprocessed_images, preprocessed_image], 0)

    preprocessed_images = preprocessed_images.to(device)
    return preprocessed_images

def preprocess_image_2(images, device):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    #     print(images.shape)
    preprocessed_images = None
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = get_valid_transforms()
    for image in images:
        sample = {'image': image}
        sample = preprocess(**sample)
        preprocessed_image = sample['image']
        # Add first dimension as the network expects a batch
        preprocessed_image = preprocessed_image.unsqueeze(0)
        if preprocessed_images is None:
            preprocessed_images = preprocessed_image
        else:
            preprocessed_images = torch.cat([preprocessed_images, preprocessed_image], 0)

    preprocessed_images = preprocessed_images.to(device)
    return preprocessed_images

def preprocess_image_3(images, device):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    #     print(images.shape)
    preprocessed_images = None
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    for image in images:
        image = cv2.resize(image, dsize=(224, 224)).astype(np.float32)
        image /= 255
        image = np.transpose(image, (2, 0, 1))
        preprocessed_image = torch.from_numpy(image)
        # Add first dimension as the network expects a batch
        preprocessed_image = preprocessed_image.unsqueeze(0)
        if preprocessed_images is None:
            preprocessed_images = preprocessed_image
        else:
            preprocessed_images = torch.cat([preprocessed_images, preprocessed_image], 0)

    preprocessed_images = preprocessed_images.to(device)
    return preprocessed_images


def predict_with_model(image, model, post_function=nn.Softmax(dim=1), device='cpu', is_tta=False):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image
    :param model: torch model with linear layer at the end
    :param post_function: e.g., softmax
    :param cuda: enables cuda, must be the same parameter as the model
    :return: prediction (1 = fake, 0 = real)
    """
    # Preprocess
    # preprocessed_image = preprocess_image(image, device)
    # preprocessed_image = preprocess_image_2(image, device)
    preprocessed_image = preprocess_image_3(image, device)

    # Model prediction
    model.eval()
    with torch.no_grad():
        if is_tta:
            output = TTA(model, preprocessed_image, activation=post_function)
        else:
            output = model(preprocessed_image)
            output = post_function(output)

    # Cast to desired
    _, prediction = torch.max(output, 1)  # argmax
    prediction = prediction.cpu().numpy()

    return prediction, output.cpu().numpy()


def clip_pred(val, threshold=0.2):
    if val < threshold:
        val = threshold
    elif val > (1 - threshold):
        val = 1 - threshold

    return val


class ToyPredictor(DeeperForensicsDetector):
    def __init__(self):
        super(ToyPredictor, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_ensamble = False
        if not self.is_ensamble:
            # model, _, *_ = model_selection('se_resnext101_32x4d', num_out_classes=2, dropout=0.5)
            model = get_efficientnet(model_name='efficientnet-b0', num_classes=2, pretrained=False)
            model_path = './weight/output_my_aug/efn-b0_LS_27_loss_0.2205.pth'
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Load model in:', model_path)
            self.model = model.to(self.device)
        else:
            self.models = self.load_models(model_names=['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2'],
                                           model_paths=['./weight/output_my_aug/efn-b0_LS_27_loss_0.2205.pth',
                                                        './weight/output_my_aug/efn-b1_LS_6_loss_0.1756.pth',
                                                        './weight/output_my_aug/efn-b2_LS_12_loss_0.1728.pth'])
        self.mtcnn = MTCNN(margin=14, keep_all=True, factor=0.6, device=self.device).eval()

    def load_models(self, model_names, model_paths):
        models = []
        for i in range(len(model_names)):
            model = get_efficientnet(model_name=model_names[i], num_classes=2, pretrained=False)
            model_path = model_paths[i]
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Load model ', i, 'in:', model_path)
            model.to(self.device)
            models.append(model)

        return models

    def predict(self, video_frames):
        # Here, we just simply return possibility of 0.5
        # Define face detection pipeline
        detection_pipeline = DetectionPipeline(detector=self.mtcnn)
        # Load frames and find faces
        faces = detection_pipeline(video_frames)

        if faces.shape == (0,):  # 排除没有提取到人脸的视频
            print('No face detect in video!')
            pred = 0.5
        else:
            if not self.is_ensamble:
                prediction, output = predict_with_model(faces, self.model, device=self.device, is_tta=False)
                pred = output[:, 1]
                pred = sum(pred) / len(pred)
                pred = clip_pred(pred, threshold=0.01)
            else:
                pred = []
                for model in self.models:
                    prediction, output = predict_with_model(faces, model, device=self.device, is_tta=False)
                    pred_i = output[:, 1]
                    pred_i = sum(pred_i) / len(pred_i)
                    pred.append(pred_i)

                pred = sum(pred) / len(pred)
                pred = clip_pred(pred, threshold=0.01)
        return pred
