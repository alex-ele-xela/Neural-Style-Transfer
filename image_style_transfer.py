import cv2 as cv
import numpy as np
import torch
import os

from matplotlib import pyplot as plt
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

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def image_style_transfer():
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    content_img_dir = os.path.join(default_resource_dir, 'content_images')
    style_img_dir = os.path.join(default_resource_dir, 'style_images')
    output_img_dir = os.path.join(default_resource_dir, 'output_images')

    content_img_name = "tubingen.png"
    style_img_name = "vg_starry_night.jpg"

    content_img_path = os.path.join(content_img_dir, content_img_name)
    style_img_path = os.path.join(style_img_dir, style_img_name)

    content_img = utils.load_image(content_img_path)
    # print(content_img.size())
    # cv.imshow("Content Image", content_img.squeeze(axis=0).numpy().size())
    style_img = utils.load_image(style_img_path)
    # cv.imshow("Style Image", style_img.squeeze)

    # Random noise image
    # gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
    # init_img = torch.from_numpy(gaussian_noise_img)
    # optimizing_img = init_img.clone().detach().requires_grad_(True)

    optimizing_img = content_img.clone().detach().requires_grad_(True)

    # neural_net = vgg_nets.Vgg16().eval()
    neural_net = vgg_nets.Vgg19().eval()
    content_feature_maps = neural_net(content_img)
    style_feature_maps = neural_net(style_img)
    
    target_content = content_feature_maps[3].squeeze(axis=0)
    target_gram_matrices = [utils.gram_matrix(x) for x in style_feature_maps]

    content_weight = 1e5
    style_weight_dic = {
        1e1: "1e1",
        1e2: "1e2",
        1e3: "1e3",
        1e4: "1e4"
    }
    tv_weight = 1e0

    style_weight = 1e4

    optimized_img_name = "NewImg"
    output_dir_name = content_img_name.split(".")[0] + " styled as " + style_img_name.split(".")[0]
    weight_dir_name = "1e5 " + style_weight_dic[style_weight] + " 1e0"
    dump_path = os.path.join(os.path.join(output_img_dir, output_dir_name), weight_dir_name)
    os.makedirs(dump_path, exist_ok=True)

    
    utils.save_image(optimizing_img, optimized_img_name, 0, dump_path)

    # optimizer = Adam((optimizing_img,), lr=1e1)
    # tuning_step = make_tuning_step(neural_net, optimizer, target_content, target_gram_matrices, content_weight, style_weight)
    # for i in range(3000):
    #     total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
    #     with torch.no_grad():
    #         print(f'Adam | iteration: {i:03}, total loss={total_loss.item():12.4f}, content_loss={content_weight * content_loss.item():12.4f}, style loss={style_weight * style_loss.item():12.4f}, tv loss={tv_weight * tv_loss.item():12.4f}')
    #         if i==2999 or ((i+1)%50 == 0):
    #             utils.save_image(optimizing_img, optimized_img_name, i, dump_path)

    optimizer = LBFGS((optimizing_img,), max_iter=1000, line_search_fn='strong_wolfe')
    cnt = 0
    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style_loss, tv_loss = get_loss(neural_net, optimizing_img, target_content, target_gram_matrices, content_weight, style_weight, tv_weight)
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            print(f'LBFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={content_weight * content_loss.item():12.4f}, style loss={style_weight * style_loss.item():12.4f}, tv loss={tv_weight * tv_loss.item():12.4f}')
            if cnt==2999 or ((cnt+1)%10 == 0):
                utils.save_image(optimizing_img, optimized_img_name, cnt, dump_path)
        cnt += 1
        return total_loss
    optimizer.step(closure)


image_style_transfer()