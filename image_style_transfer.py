import cv2 as cv
import numpy as np
import torch

from matplotlib import pyplot as plt
from models import vgg_nets
from torch.optim import Adam
from utils import utils


def get_loss(neural_net, optimizing_img, target_content, target_gram_matrices, content_weight, style_weight):
    current_feature_maps = neural_net(optimizing_img)

    current_content = current_feature_maps[3].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content, current_content)

    style_loss = 0.0
    current_gram_matrices = [utils.gram_matrix(x) for x in current_feature_maps]
    for target, current in zip(target_gram_matrices, current_gram_matrices):
        style_loss += torch.nn.MSELoss(reduction='sum')(target[0], current[0])
    style_loss /= len(target_gram_matrices)

    total_loss = (content_weight * content_loss) + (style_weight * style_loss)

    return total_loss, content_loss, style_loss


def make_tuning_step(neural_net, optimizer, target_content, target_gram_matrices, content_weight, style_weight):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss = get_loss(neural_net, optimizing_img, target_content, target_gram_matrices, content_weight, style_weight)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def image_style_transfer():
    content_img_path = "content.png"
    style_img_path = "style.jpg"

    content_img = utils.load_image(content_img_path)
    # print(content_img.size())
    # cv.imshow("Content Image", content_img.squeeze(axis=0).numpy().size())
    style_img = utils.load_image(style_img_path)
    # cv.imshow("Style Image", style_img.squeeze)

    # Random noise image
    gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
    init_img = torch.from_numpy(gaussian_noise_img)
    optimizing_img = init_img.clone().detach().requires_grad_(True)

    neural_net = vgg_nets.Vgg16().eval()
    content_feature_maps = neural_net(content_img)
    style_feature_maps = neural_net(style_img)
    
    target_content = content_feature_maps[3].squeeze(axis=0)
    target_gram_matrices = [utils.gram_matrix(x) for x in style_feature_maps]

    content_weight = 1e5
    style_weight = 3e4

    optimizer = Adam((optimizing_img,), lr=1e1)
    tuning_step = make_tuning_step(neural_net, optimizer, target_content, target_gram_matrices, content_weight, style_weight)
    for i in range(3000):
        total_loss, content_loss, style_loss = tuning_step(optimizing_img)
        with torch.no_grad():
            print(f'Adam | iteration: {i:03}, total loss={total_loss.item():12.4f}, content_loss={content_weight * content_loss.item():12.4f}, style loss={style_weight * style_loss.item():12.4f}')

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        out_img = optimizing_img.squeeze(axis=0).detach().numpy()
        out_img = np.copy(np.moveaxis(out_img, 0, 2))
        out_img += np.array(np.multiply(IMAGENET_MEAN, 255)).reshape((1, 1, 3))
        out_img = np.clip(out_img, 0, 255).astype('uint8')

        plt.imshow(out_img)

        # if i%20 == 0:
        #     cv.imshow(f"Iteration {i}", optimizing_img.clone().detach().numpy())
    
    plt.imshow(optimizing_img)

    # neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    # print(f'Using {config["model"]} in the optimization procedure.')

    # content_img_set_of_feature_maps = neural_net(content_img)
    # style_img_set_of_feature_maps = neural_net(style_img)

    # target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    # target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    # target_representations = [target_content_representation, target_style_representation]

    # # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    # num_of_iterations = {
    #     "lbfgs": 1000,
    #     "adam": 3000,
    # }

    # #
    # # Start of optimization procedure
    # #
    # if config['optimizer'] == 'adam':
    #     optimizer = Adam((optimizing_img,), lr=1e1)
    #     tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
    #     for cnt in range(num_of_iterations[config['optimizer']]):
    #         total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
    #         with torch.no_grad():
    #             print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
    #             utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    # elif config['optimizer'] == 'lbfgs':
    #     # line_search_fn does not seem to have significant impact on result
    #     optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
    #     cnt = 0

    #     def closure():
    #         nonlocal cnt
    #         if torch.is_grad_enabled():
    #             optimizer.zero_grad()
    #         total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
    #         if total_loss.requires_grad:
    #             total_loss.backward()
    #         with torch.no_grad():
    #             print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
    #             utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)

    #         cnt += 1
    #         return total_loss

    #     optimizer.step(closure)

    # return dump_path


image_style_transfer()