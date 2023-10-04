import os
import json

import torch
from torch.utils.data import DataLoader


from utils import utils
from models.transformer_net import TransformerNet


def stylize_image(styling_config):
    # Prepare the model - load the weights and put the model into evaluation mode
    stylization_model = TransformerNet()
    training_state = torch.load(styling_config["model_path"])
    state_dict = training_state["state_dict"]
    stylization_model.load_state_dict(state_dict, strict=True)
    stylization_model.eval()

    if styling_config['verbose']:
        utils.print_model_metadata(training_state)

    with torch.no_grad():
        content_image = utils.prepare_img(styling_config['content_img_path'])
        stylized_img = stylization_model(content_image).numpy()[0]
        utils.save_image(stylized_img, "name_here", styling_config['dump_path'])


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
    output_img_dir = os.path.join(default_resource_dir, 'output_images', 'perceptual_loss')

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
    styling_config = get_config('pre_trained_style_transfer_config.json')

    stylize_image(styling_config)
