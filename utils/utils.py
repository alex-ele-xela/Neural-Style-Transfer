"""
Module contains several functions that help in performing Neural Style Transfer
"""

import cv2 as cv
import numpy as np
import os
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Sampler

# known Imagnet quantities
IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def logger(path, text):
    """
    Function to write text logs to the given file

    Args:
        path (string): log file location
        text (string): log text to write to log file
    """

    with open(path, 'a') as f:
        f.write(text)



def load_image(img_path):
    """
    Loads image and pre-processes to return a tensor ready to be processed by the neural network
    
    Args:
        img_path (string): path of the image to be loaded

    Raises:
        Exception: raised if the given image path does not exist

    Returns:
        Tensor: tensor of loaded image to be loaded into neural net
    """

    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    
    # read image in
    img = cv.imread(img_path)[:, :, ::-1]

    # convert image data type to float and normalize to range [0,1]
    img = img.astype(np.float32)
    img /= 255.0

    # define transform composition to take image and normalize it using standard Imagenet mean and deviation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    # adding 1 axis at 0th position for passing tensor through CNN
    img = transform(img).unsqueeze(0)

    return img


def save_image(optimizing_img, name, dump_path):
    """
    Post-processes image to a readablee format and saves it as png image in the given path

    Args:
        optimizing_img (Tensor): image being modified using the neural nets and optimization, which is to be saved
        cnt (int): count of the iteration of the optimization process
        dump_path (string): path of the folder to save image
    """

    # post-processing optimizing image tensor to RGB format
    out_img = optimizing_img.squeeze(axis=0).detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)

    # creating output image name
    out_img_name = name + ".png"

    # converting pixel vales into 0 to 255 range
    dump_img = np.copy(out_img)
    dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    dump_img = np.clip(dump_img, 0, 255).astype('uint8')

    # saving image to dump path
    cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])


def gram_matrix(x, should_normalize=True):
    """
    Calculate the Gram matrix from the given feature map

    Args:
        x (Tensor): feature map tensor
        should_normalize (bool, optional): normalize by the number of values present in the feature map. Defaults to True.

    Returns:
        Tensor: tensor describing the gram matrix for the given feature map
    """

    # squeezing width and heeight dimensions into one dimension and making transpose
    (b, channels, height, width) = x.size()
    features = x.view(b, channels, width * height)
    features_t = features.transpose(1, 2)

    # performing batch matrix multiplication
    gram = features.bmm(features_t)

    # normalizing if required
    if should_normalize:
        gram /= channels * height * width

    return gram


def total_variation(y):
    """
    Calculate total variation loss
    This acts as a regularization term which controls the smoothness of the image
    It calculates the width-wise and height-wise differences between adjacent pixel values

    Args:
        y (Tensor): tensor of image being optimized

    Returns:
        Tensor: tensor describing the total variation loss
    """

    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


def create_video(dump_path):
    """
    Create a video showing the progress of the optimization process from the output images available in the given path

    Args:
        dump_path (string): path of the folder which conatins the output images
    """

    import moviepy.video.io.ImageSequenceClip as ISC

    fps=25

    # extract all images in order
    images = [img for img in os.listdir(dump_path) if img.endswith(".png")]
    images.sort()
    image_files = [os.path.join(dump_path,img) for img in images]

    # create video
    clip = ISC.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(os.path.join(dump_path, 'out.mp4'), verbose=False, logger=None)




# Code for training Johnson neural net


def prepare_img(img_path, batch_size=1):
    img = load_image(img_path)  
    img = img.repeat(batch_size, 1, 1, 1)

    return img

def batch_total_variation(img_batch):
    batch_size = img_batch.shape[0]
    return (torch.sum(torch.abs(img_batch[:, :, :, :-1] - img_batch[:, :, :, 1:])) +
            torch.sum(torch.abs(img_batch[:, :, :-1, :] - img_batch[:, :, 1:, :]))) / batch_size

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


def get_header(training_config):
    header = ""
    header += f'Learning the style of {training_config["style_img_name"]} style image.\n'
    header += '*' * 80 + "\n"
    header += f'Hyperparams: content_weight={training_config["content_weight"]}, style_weight={training_config["style_weight"]} and tv_weight={training_config["tv_weight"]} \n'
    header += '*' * 80 + "\n"

    if training_config["console_log_freq"]:
        header += f'Logging to console every {training_config["console_log_freq"]} batches. \n'
    else:
        header += f'Console logging disabled. Change console_log_freq if you want to use it. \n'

    if training_config["checkpoint_freq"]:
        header += f'Saving checkpoint models every {training_config["checkpoint_freq"]} batches. \n'
    else:
        header += f'Checkpoint models saving disabled. \n'
    header += '*' * 80 +"\n"


def get_training_metadata(training_config):
    num_of_datapoints = training_config['subset_size'] * training_config['num_of_epochs']
    training_metadata = {
        "content_weight": training_config['content_weight'],
        "style_weight": training_config['style_weight'],
        "tv_weight": training_config['tv_weight'],
        "num_of_datapoints": num_of_datapoints
    }
    return training_metadata