import cv2 as cv
import numpy as np
import torch
import os
import json

from models import vgg_nets
from torch.optim import Adam, LBFGS
from utils import utils


def get_loss(neural_net, optimizing_img, target_content, target_gram_matrices, content_weight, style_weight, tv_weight):
    current_feature_maps = neural_net(optimizing_img)

    current_content = current_feature_maps[3].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content, current_content)

    style_loss = 0.0
    current_gram_matrices = [utils.gram_matrix(x) for x in current_feature_maps]
    for target, current in zip(target_gram_matrices, current_gram_matrices):
        style_loss += torch.nn.MSELoss(reduction='sum')(target[0], current[0])
    style_loss /= len(target_gram_matrices)

    tv_loss = utils.total_variation(optimizing_img)

    total_loss = (content_weight * content_loss) + (style_weight * style_loss) + (tv_weight * tv_loss)

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_content, target_gram_matrices, content_weight, style_weight, tv_weight):
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


def logger(path, text):
    with open(path, 'a') as f:
        f.write(text)


def image_style_transfer(config):
    log_file = os.path.join(config["dump_path"], "log.txt")
    logger(log_file, f"Using config: {str(config)}\n\n")

    content_img = utils.load_image(config["content_img_path"])
    style_img = utils.load_image(config["style_img_path"])
    logger(log_file, "Loaded images\n\n")


    if config["init_img"] == "random":
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img)
        optimizing_img = init_img.clone().detach().requires_grad_(True)
    elif config["init_img"] == "content":
        optimizing_img = content_img.clone().detach().requires_grad_(True)
    
    logger(log_file, "Initialized Optimizing image\n\n")

    if config["neural_net"] == "VGG19":
        neural_net = vgg_nets.Vgg19().eval()
    else:
        neural_net = vgg_nets.Vgg16().eval()

    content_feature_maps = neural_net(content_img)
    style_feature_maps = neural_net(style_img)
    logger(log_file, "Generated feature maps\n\n")
    
    target_content = content_feature_maps[3].squeeze(axis=0)
    target_gram_matrices = [utils.gram_matrix(x) for x in style_feature_maps]
    
    utils.save_image(optimizing_img, 0, dump_path)

    iter = {
        "Adam": 3000,
        "LBFGS": 500
    }

    if config["optimizer"] == "Adam":
        logger(log_file, "Starting Adam Optimization\n\n")
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_content, target_gram_matrices, config["content_weight"], config["style_weight"], config["tv_weight"])
        for i in range(iter["Adam"]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                text = f'Adam | iteration: {i:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}'
                print(text)
                logger(log_file, text+"\n")
                if i==2999 or ((i+1)%50 == 0):
                    utils.save_image(optimizing_img, i, config["dump_path"])


    elif config["optimizer"] == "LBFGS":
        logger(log_file, "Starting LBFGS Optimization\n\n")

        optimizer = LBFGS((optimizing_img,), max_iter=iter["LBFGS"], tolerance_grad=-1, line_search_fn='strong_wolfe')
        cnt = 1

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = get_loss(neural_net, optimizing_img, target_content, target_gram_matrices, config["content_weight"], config["style_weight"], config["tv_weight"])
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                text = f'LBFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}'
                print(text)
                logger(log_file, text+"\n")
                del text

                if cnt%4 == 0:
                    utils.save_image(optimizing_img, cnt, dump_path)
            cnt += 1

            return total_loss
        
        optimizer.step(closure)

        if cnt < 100:
            optimizer = LBFGS((optimizing_img,), lr=5, max_iter=iter["LBFGS"]-cnt, tolerance_grad=-1, line_search_fn='strong_wolfe')
            optimizer.step(closure)
    
    utils.create_video(config["dump_path"])
    logger(log_file, "\nGenerated video clip")


if __name__ == "__main__":
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_img_dir = os.path.join(default_resource_dir, 'content_images')
    style_img_dir = os.path.join(default_resource_dir, 'style_images')
    output_img_dir = os.path.join(default_resource_dir, 'output_images')

    config = dict() 
    config = json.load(open('config.json'))   

    # Use this part if you want to loop through all images
    # content_img_names = os.listdir(content_img_dir)
    # style_img_names = os.listdir(style_img_dir)

    # for content_img_name in content_img_names:
    #     config["content_img_path"] = os.path.join(content_img_dir, content_img_name)

    #     for style_img_name in style_img_names:
    #         config["style_img_path"] = os.path.join(style_img_dir, style_img_name)

    #         output_dir_name = content_img_name.split(".")[0] + " styled as " + style_img_name.split(".")[0]
    #         weight_dir_name = f'{int(config["content_weight"])} {int(config["style_weight"])} {int(config["tv_weight"])}'
    #         dump_path = os.path.join(os.path.join(output_img_dir, output_dir_name), weight_dir_name)
    #         os.makedirs(dump_path, exist_ok=True)
    #         print("Made dir")

    #         config["dump_path"] = dump_path

    #         image_style_transfer(config)

    # Use this part if you want to use config.json file
    config["content_img_path"] = os.path.join(content_img_dir, config["content_img_name"])
    config["style_img_path"] = os.path.join(style_img_dir, config["style_img_name"])

    output_dir_name = config["content_img_name"].split(".")[0] + " styled as " + config["style_img_name"].split(".")[0]
    weight_dir_name = f'{int(config["content_weight"])} {int(config["style_weight"])} {int(config["tv_weight"])}'
    dump_path = os.path.join(os.path.join(output_img_dir, output_dir_name), weight_dir_name)
    os.makedirs(dump_path, exist_ok=True)
    del output_dir_name, weight_dir_name

    config["dump_path"] = dump_path

    image_style_transfer(config)