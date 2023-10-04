import zipfile
from torch.hub import download_url_to_file
import os

# If the link is broken you can download the MS COCO 2014 dataset manually from http://cocodataset.org/#download
MS_COCO_2014_TRAIN_DATASET_PATH = r'http://images.cocodataset.org/zips/train2014.zip' 


if __name__ == '__main__':
    # step1: download the resource to local filesystem
    remote_resource_path = MS_COCO_2014_TRAIN_DATASET_PATH
    print(f'Downloading from {remote_resource_path}')
    resource_tmp_path = 'mscoco.zip'
    download_url_to_file(remote_resource_path, resource_tmp_path)

    # step2: unzip the resource
    print(f'Started unzipping...')
    with zipfile.ZipFile(resource_tmp_path) as zf:
        local_resource_path = os.path.join(os.path.dirname(__file__), 'data', 'mscoco')
        os.makedirs(local_resource_path, exist_ok=True)
        zf.extractall(path=local_resource_path)
    print(f'Unzipping to: {local_resource_path} finished.')

    # step3: remove the temporary resource file
    os.remove(resource_tmp_path)
    print(f'Removing tmp file {resource_tmp_path}.')
