"""
Module that performs Neural Style Transfer using Gatys optimization based on a set configuration
"""

import cv2 as cv
import numpy as np
import torch
import os
import json

from models import vgg_nets
from torch.optim import Adam, LBFGS
from utils import utils


def get_loss(neural_net, optimizing_img, target_content, target_gram_matrices, content_weight, style_weight, tv_weight):
    """
    Function that calculates the loss values according to the given weights

    Args:
        neural_net (Module): custom trained CNN that will be used to compare images to perform Style Transfer optimization
        optimizing_img (Tensor): image that will be modified using optimization to obtain the desired result
        target_content (Tensor): target content feature map
        target_gram_matrices (Tensor): gram matrtices of style feature maps
        content_weight (float): weight(importance) for the content loss
        style_weight (float): weight(importance) for the style loss
        tv_weight (float): weight(importance) for the total variation loss

    Returns:
        (Tensor, Tensor, Tensor, Tensor): tuple containing  the total loss, content loss, style loss and total variation loss
    """

    # evaluate the feature maps of the current optmimizing image
    current_feature_maps = neural_net(optimizing_img)

    # calculate content loss
    current_content = current_feature_maps[3].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content, current_content)

    # calculate style loss
    style_loss = 0.0
    current_gram_matrices = [utils.gram_matrix(x) for x in current_feature_maps]
    for target, current in zip(target_gram_matrices, current_gram_matrices):
        style_loss += torch.nn.MSELoss(reduction='sum')(target[0], current[0])
    style_loss /= len(target_gram_matrices)

    # calculate total variation loss
    tv_loss = utils.total_variation(optimizing_img)

    # calculate total loss
    total_loss = (content_weight * content_loss) + (style_weight * style_loss) + (tv_weight * tv_loss)

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_content, target_gram_matrices, content_weight, style_weight, tv_weight):
    """
    Build the function that will take the optimizing image, calculate the loss and update the optimizing image accordingly

    Args:
        neural_net (Module): custom trained CNN that will be used to compare images to perform Style Transfer optimization
        optimizer (Optimizer): thee optimizer that is being used for the process of Style Transfer optimization
        target_content (Tensor): target content feature map
        target_gram_matrices (Tensor): gram matrtices of style feature maps
        content_weight (float): weight(importance) for the content loss
        style_weight (float): weight(importance) for the style loss
        tv_weight (float): weight(importance) for the total variation loss

    Returns:
        (Tensor, Tensor, Tensor, Tensor): tuple containing  the total loss, content loss, style loss and total variation loss
    """

    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = get_loss(neural_net, optimizing_img, target_content, target_gram_matrices, content_weight, style_weight, tv_weight)
        
        # Computes gradients
        total_loss.backward()
        
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()

        return total_loss, content_loss, style_loss, tv_loss

    return tuning_step



def image_style_transfer(config):
    """
    Main function to perform the task of Neurla style transfer on the given image

    Args:
        config (dict): contains all required configuration to perform the Neural style transfer task
    """

    # creating the log file in the dump path
    log_file = os.path.join(config["dump_path"], "log.txt")
    utils.logger(log_file, f"Using config: {str(config)}\n\n")

    # loading the content and style images
    content_img = utils.load_image(config["content_img_path"])
    style_img = utils.load_image(config["style_img_path"])
    utils.logger(log_file, "Loaded images\n\n")

    # initializing the optimizing image as an image of random noise or the content image, according to the given configuration
    if config["init_img"] == "random":
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img)
        optimizing_img = init_img.clone().detach().requires_grad_(True)
    elif config["init_img"] == "content":
        optimizing_img = content_img.clone().detach().requires_grad_(True)
    utils.logger(log_file, "Initialized Optimizing image\n\n")

    # initalizing the custom trained CNN as per the given configuration and setting it to eval mode to prevent gradient calculation and updation to weights
    if config["neural_net"] == "VGG19":
        neural_net = vgg_nets.Vgg19().eval()
    else:
        neural_net = vgg_nets.Vgg16().eval()

    # passing the content and style image through the neural net and saving the feature maps
    content_feature_maps = neural_net(content_img)
    style_feature_maps = neural_net(style_img)
    utils.logger(log_file, "Generated feature maps\n\n")
    
    # saving the relevant content feature map and style gram matrices
    target_content = content_feature_maps[3].squeeze(axis=0)
    target_gram_matrices = [utils.gram_matrix(x) for x in style_feature_maps]
    
    # saving initialized optimizing image as iteration 0 of the optimization process
    utils.save_image(optimizing_img, str(0).zfill(4), config["dump_path"])

    # setting max number of iterations for each optimizer
    iter = {
        "Adam": 3000,
        "LBFGS": 500
    }

    # performing optimization using the optimizer specified in the configuration
    if config["optimizer"] == "Adam":
        utils.logger(log_file, "Starting Adam Optimization\n\n")

        # initializing the optimizer
        optimizer = Adam((optimizing_img,), lr=1e1)

        # building function to calculate loss at each iteration
        tuning_step = make_tuning_step(neural_net, optimizer, target_content, target_gram_matrices, config["content_weight"], config["style_weight"], config["tv_weight"])
        
        # performing style transfer using optimization
        for i in range(iter["Adam"]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)

            # providing output and logging with no gradient computations
            with torch.no_grad():
                text = f'Adam | iteration: {i:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}'
                print(text)
                utils.logger(log_file, text+"\n")

                # saving image every 50 iterations
                if i==2999 or ((i+1)%50 == 0):
                    utils.save_image(optimizing_img, str(i+1).zfill(4), config["dump_path"])


    elif config["optimizer"] == "LBFGS":
        utils.logger(log_file, "Starting LBFGS Optimization\n\n")

        # initializing the optimizer
        optimizer = LBFGS((optimizing_img,), max_iter=iter["LBFGS"], tolerance_grad=-1, line_search_fn='strong_wolfe')

        # initializing counter for number of iterations of optimizer
        cnt = 1

        # bulding closure function for LBFGS optimizer
        def closure():
            nonlocal cnt

            # setting gradients to zero
            if torch.is_grad_enabled():
                optimizer.zero_grad()

            # calculate loss
            total_loss, content_loss, style_loss, tv_loss = get_loss(neural_net, optimizing_img, target_content, target_gram_matrices, config["content_weight"], config["style_weight"], config["tv_weight"])
            
            # backward propagation of gradients
            if total_loss.requires_grad:
                total_loss.backward()

            # providing output and logging with no gradient computations
            with torch.no_grad():
                text = f'LBFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}'
                print(text)
                utils.logger(log_file, text+"\n")
                del text

                # saving image every 4 iterations
                if cnt%4 == 0:
                    utils.save_image(optimizing_img, str(cnt).zfill(4), config["dump_path"])
            cnt += 1

            return total_loss
        
        # performing LBFGS optimization
        optimizer.step(closure)

        # restarting LBFGS optimizer with greater learning rate if the earlier one stopped before reaching 100 iterations due to Early stopping
        if cnt < 100:
            optimizer = LBFGS((optimizing_img,), lr=5, max_iter=iter["LBFGS"]-cnt, tolerance_grad=-1, line_search_fn='strong_wolfe')
            optimizer.step(closure)
    
    # creaeting video of the optimizing images
    utils.create_video(config["dump_path"])
    utils.logger(log_file, "\nGenerated video clip")


def get_config(file) -> dict:
    """
    Function to pull Style Transfer configuration from the json file

    Args:
        file (string): json configuration  file path

    Returns:
        dict: dictionary containing all required configuration to perform the Neural style transfer task
    """

    # setting important directory location
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_img_dir = os.path.join(default_resource_dir, 'content_images')
    style_img_dir = os.path.join(default_resource_dir, 'style_images')
    output_img_dir = os.path.join(default_resource_dir, 'output_images', 'optimization')

    # loading json file
    config = dict() 
    config = json.load(open(file))

    # setting location of content and style images
    config["content_img_path"] = os.path.join(content_img_dir, config["content_img_name"])
    config["style_img_path"] = os.path.join(style_img_dir, config["style_img_name"])

    # create dump path location
    output_dir_name = config["content_img_name"].split(".")[0] + " styled as " + config["style_img_name"].split(".")[0]
    weight_dir_name = f'{int(config["content_weight"])} {int(config["style_weight"])} {int(config["tv_weight"])}'
    dump_path = os.path.join(os.path.join(output_img_dir, output_dir_name), weight_dir_name)
    os.makedirs(dump_path, exist_ok=True)
    del output_dir_name, weight_dir_name

    config["dump_path"] = dump_path

    return config



if __name__ == "__main__":
    config = get_config('image_style_transfer_config.json')

    image_style_transfer(config)