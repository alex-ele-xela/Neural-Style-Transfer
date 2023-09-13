import cv2 as cv
import numpy as np
import os
import torch

from torchvision import transforms

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]

    img = img.astype(np.float32)
    img /= 255.0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(img).unsqueeze(0)

    return img


def save_image(optimizing_img, cnt, dump_path):
    out_img = optimizing_img.squeeze(axis=0).detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)
    out_img_name = str(cnt).zfill(4) + ".png"

    dump_img = np.copy(out_img)
    dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
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


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


def create_video(dump_path):
    import moviepy.video.io.ImageSequenceClip as ISC

    fps=25
    images = [img for img in os.listdir(dump_path) if img.endswith(".png")]
    images.sort()
    image_files = [os.path.join(dump_path,img) for img in images]

    clip = ISC.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(os.path.join(dump_path, 'out.mp4'), verbose=False, logger=None)