import cv2 as cv
import numpy as np
import os
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Sampler

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


class SequentialSubsetSampler(Sampler):
    def __init__(self, data_source, subset_size):
        assert isinstance(data_source, datasets.ImageFolder)
        self.data_source = data_source

        if subset_size is None:
            subset_size = len(data_source)
        assert 0 < subset_size <= len(data_source), f'Subset size should be between (0, {len(data_source)}].'
        self.subset_size = subset_size

    def __iter__(self):
        return iter(range(self.subset_size))

    def __len__(self):
        return self.subset_size


def get_training_data_loader(training_config, should_normalize=True, is_255_range=False):
    transform_list = [transforms.Resize(training_config['image_size']),
                      transforms.CenterCrop(training_config['image_size']),
                      transforms.ToTensor()]
    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x.mul(255)))
    if should_normalize:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL) if is_255_range else transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.ImageFolder(training_config['dataset_path'], transform)
    sampler = SequentialSubsetSampler(train_dataset, training_config['subset_size'])
    training_config['subset_size'] = len(sampler)  # update in case it was None
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], sampler=sampler, drop_last=True)
    print(f'Using {len(train_loader)*training_config["batch_size"]*training_config["num_of_epochs"]} datapoints ({len(train_loader)*training_config["num_of_epochs"]} batches) (MS COCO images) for transformer network training.')
    return train_loader