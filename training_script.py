import os
import argparse
import time

import numpy as np
import torch
import json
from torch.optim import Adam

from models.vgg_nets import Vgg19 as PerceptualLossNet
from models.transformer_net import TransformerNet
import utils.utils as utils


def train(training_config):
    # creating the log file in the dump path
    log_file = os.path.join(training_config['model_binaries_path'],
                            f"{training_config['style_img_name'].split('.')[0]}.txt")
    utils.logger(log_file, f"Using config: {str(training_config)}\n\n")

    # prepare data loader
    train_loader = utils.get_training_data_loader(training_config)

    # prepare neural networks
    transformer_net = TransformerNet().train()
    perceptual_loss_net = PerceptualLossNet(requires_grad=False)

    # initialize Adam optimizer for the Transformer net params
    optimizer = Adam(transformer_net.parameters())

    utils.logger(log_file, "Initialized neural nets and optimizer\n\n")

    # Calculate style image's Gram matrices
    style_img = utils.prepare_img(training_config['style_img_path'], batch_size=training_config['batch_size'])
    style_img_set_of_feature_maps = perceptual_loss_net(style_img)
    target_style = [utils.gram_matrix(x) for x in style_img_set_of_feature_maps]

    utils.logger(log_file, "Calculated style feature maps\n\n")

    # Printing details of training
    header = utils.get_header(training_config)
    print(header)
    utils.logger(log_file, header)
    del header
    
    # Storing and aggregating losses
    acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]

    # storing starting time
    ts = time.time()

    utils.logger(log_file, "Starting training:\n")
    for epoch in range(training_config['num_of_epochs']):
        for batch_id, (content_batch, _) in enumerate(train_loader):
            # Feed content batch through transformer net
            content_batch = content_batch
            stylized_batch = transformer_net(content_batch)

            # Extracting feature maps of stylized batch through Perceptual Loss Net (VGG19)
            stylized_feature_maps = perceptual_loss_net(stylized_batch)

            # Calculate content representations and content loss
            target_content = perceptual_loss_net(content_batch).conv4_2
            current_content = stylized_feature_maps.conv4_2
            content_loss = torch.nn.MSELoss(reduction='mean')(target_content, current_content)

            # step4: Calculate style representation and style loss
            style_loss = 0.0
            current_style = [utils.gram_matrix(x) for x in stylized_feature_maps]
            for gram_gt, gram_hat in zip(target_style, current_style):
                style_loss += torch.nn.MSELoss(reduction='mean')(gram_gt, gram_hat)
            style_loss /= len(target_style)

            # Calculate total variation loss
            tv_loss = utils.batch_total_variation(stylized_batch)

            # Calculate total loss and Backpropagate
            total_loss = (training_config["content_weight"] * content_loss) + (training_config["style_weight"] * style_loss) + (training_config["tv_weight"] * tv_loss)
            total_loss.backward()

            # Updating weights and then setting gradients to zero
            optimizer.step()
            optimizer.zero_grad()

            # storing aggregated loss
            acc_content_loss += content_loss.item()
            acc_style_loss += style_loss.item()
            acc_tv_loss += tv_loss.item()

            # logging to console and log file
            if training_config['console_log_freq'] is not None and batch_id % training_config['console_log_freq'] == 0:
                text = f'Epoch={epoch + 1} | Batch=[{batch_id + 1}/{len(train_loader)}] | Content-loss={acc_content_loss / training_config["console_log_freq"]} | Style-loss={acc_style_loss / training_config["console_log_freq"]} | TV-loss={acc_tv_loss / training_config["console_log_freq"]} | Total-loss={(acc_content_loss + acc_style_loss + acc_tv_loss) / training_config["console_log_freq"]} | Time elapsed={(time.time()-ts)/60:.2f}min'
                print(text)
                utils.logger(log_file, text + "\n")
                acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]

            # saving intermdediate checkpoints of the model
            if training_config['checkpoint_freq'] is not None and (batch_id + 1) % training_config['checkpoint_freq'] == 0:
                training_state = utils.get_training_metadata(training_config)
                training_state["state_dict"] = transformer_net.state_dict()
                training_state["optimizer_state"] = optimizer.state_dict()
                ckpt_model_name = f"ckpt_style_{training_config['style_img_name'].split('.')[0]}_cw_{str(training_config['content_weight'])}_sw_{str(training_config['style_weight'])}_tw_{str(training_config['tv_weight'])}_epoch_{epoch}_batch_{batch_id}.pth"
                torch.save(training_state, os.path.join(training_config['checkpoints_path'], ckpt_model_name))

    # saving final model with additional metadata
    training_state = utils.get_training_metadata(training_config)
    training_state["state_dict"] = transformer_net.state_dict()
    training_state["optimizer_state"] = optimizer.state_dict()
    model_name = f"{training_config['style_img_name'].split('.')[0]}_datapoints_{training_state['num_of_datapoints']}_cw_{str(training_config['content_weight'])}_sw_{str(training_config['style_weight'])}_tw_{str(training_config['tv_weight'])}.pth"
    torch.save(training_state, os.path.join(training_config['model_binaries_path'], model_name))


def get_config(file) -> dict:
    """
    Function to pull Style Transfer configuration from the json file

    Args:
        file (string): json configuration  file path

    Returns:
        dict: dictionary containing all required configuration to perform the Neural style transfer task
    """

    # setting important directory locations
    default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
    style_img_dir = os.path.join(default_resource_dir, 'style_images')
    dataset_path = os.path.join(default_resource_dir, 'mscoco')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    checkpoints_root_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')

    # loading json file
    config = dict() 
    config = json.load(open(file))

    # setting location of style image
    config["style_img_path"] = os.path.join(style_img_dir, config["style_img_name"])

    # setting and creating checkpoints folder for the specific style
    checkpoints_path = os.path.join(checkpoints_root_path, config["style_img_name"].split('.')[0])
    os.makedirs(checkpoints_path, exist_ok=True)
        
    config['dataset_path'] = dataset_path
    config['model_binaries_path'] = model_binaries_path
    config['checkpoints_path'] = checkpoints_path

    return config


if __name__ == "__main__":
    training_config = get_config('training_script_config.json')

    train(training_config)
