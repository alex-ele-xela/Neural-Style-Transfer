import cv2 as cv
import numpy as np
import os

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    img = transform(img).unsqueeze(0)

    return img


def save_image(optimizing_img, name, cnt, dump_path):
    out_img = optimizing_img.squeeze(axis=0).detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)
    out_img_name = f"{name}_{cnt}.png"

    dump_img = np.copy(out_img)
    dump_img += np.array(np.multiply(IMAGENET_MEAN, 255)).reshape((1, 1, 3))
    dump_img = np.clip(dump_img, 0, 255).astype('uint8')
    cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])


def gram_matrix(x, should_normalize=True):
    (b, channels, height, width) = x.size()
    features = x.view(b, channels, width * height)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= channels * height * width
    return gram

