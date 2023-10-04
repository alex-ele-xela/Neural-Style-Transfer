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
    #
    # Fixed args - don't change these unless you have a good reason
    #
    content_images_path = os.path.join(os.path.dirname(__file__), 'data', 'content-images')
    output_images_path = os.path.join(os.path.dirname(__file__), 'data', 'output-images')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')

    assert utils.dir_contains_only_models(model_binaries_path), f'Model directory should contain only model binaries.'
    os.makedirs(output_images_path, exist_ok=True)

    #
    # Modifiable args - feel free to play with these
    #
    parser = argparse.ArgumentParser()
    # Put image name or directory containing images (if you'd like to do a batch stylization on all those images)
    parser.add_argument("--content_input", type=str, help="Content image(s) to stylize", default='taj_mahal.jpg')
    parser.add_argument("--batch_size", type=int, help="Batch size used only if you set content_input to a directory", default=5)
    parser.add_argument("--img_width", type=int, help="Resize content image to this width", default=500)
    parser.add_argument("--model_name", type=str, help="Model binary to use for stylization", default='mosaic_4e5_e2.pth')

    # Less frequently used arguments
    parser.add_argument("--should_not_display", action='store_false', help="Should display the stylized result")
    parser.add_argument("--verbose", action='store_true', help="Print model metadata (how the model was trained) and where the resulting stylized image was saved")
    parser.add_argument("--redirected_output", type=str, help="Overwrite default output dir. Useful when this project is used as a submodule", default=None)
    args = parser.parse_args()

    # if redirected output is not set when doing batch stylization set to default image output location
    if os.path.isdir(args.content_input) and args.redirected_output is None:
        args.redirected_output = output_images_path

    # Wrapping inference configuration into a dictionary
    styling_config = dict()
    for arg in vars(args):
        styling_config[arg] = getattr(args, arg)
    styling_config['content_images_path'] = content_images_path
    styling_config['output_images_path'] = output_images_path
    styling_config['model_binaries_path'] = model_binaries_path

    stylize_static_image(styling_config)
