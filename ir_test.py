#!/usr/bin/env python
import os
from glob import glob
import sys
import cv2
import numpy as np
sys.path.append('cut')
from cut.options.base_options import BaseOptions
from cut.options.test_options import TestOptions
from cut.models import create_model
from cut.data import create_dataset

try:
    import gdown
except ModuleNotFoundError:
    raise AssertionError('This example requires `gdown` to be installed. '
                         'Please install using `pip install gdown`')

DATASET_FOLDER = './datasets/test_imgs'

def download_file_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?export=download&confirm=t&id={file_id}"
        gdown.download(url, output_path, quiet=False)

def download_folder_from_gdrive(id, output_path):
    if not os.path.exists(output_path):
        gdown.download_folder(id=id, output=output_path, quiet=False)

def tensor2numpy(tensor_img):
    numpy_img = tensor_img.data[0].clamp(-1.0, 1.0).cpu().float().numpy()
    numpy_img = (np.transpose(numpy_img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return numpy_img.astype(np.uint8)


if __name__ == '__main__':
    testoptions = TestOptions()

    # test parameters for cut network
    opt = testoptions.gather_options()
    opt.dataroot = DATASET_FOLDER
    opt.gpu_ids = ''
    opt.num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True
    opt.display_id = -1
    opt.name = ''
    opt.isTrain = False
    testoptions.print_options(opt)

    print("Fetching pretrained model...")
    PRETRAINED_NET_D_GDRIVE_ID = '1G0G8Dkn167CP_zJNp3hzNASMfy4MT2ks'
    download_file_from_gdrive(PRETRAINED_NET_D_GDRIVE_ID, opt.checkpoints_dir + '/pretrained_net_D.pth')
    PRETRAINED_NET_G_GDRIVE_ID = '1LEwjzkBXq4gO-_46hrSlvl7i0ECr9u66'
    download_file_from_gdrive(PRETRAINED_NET_G_GDRIVE_ID, opt.checkpoints_dir + '/pretrained_net_G.pth')

    print("Fetching test dataset...")
    TEST_IMGS_A_GDRIVE_ID = '1yGODL5zZyUYLzKTFjA5yF_2s0pDK0YlR'
    download_folder_from_gdrive(TEST_IMGS_A_GDRIVE_ID, DATASET_FOLDER + '/testA')
    TEST_IMGS_B_GDRIVE_ID = '1jsmuUUGeW_IFFHyKarwF63rsKy4erNeR'
    download_folder_from_gdrive(TEST_IMGS_B_GDRIVE_ID, DATASET_FOLDER + '/testB')

    ckpt = glob(opt.checkpoints_dir + '/*.pth')
    opt.epoch = ckpt[0].split("/")[-1].split("_")[0]

    dataset = create_dataset(opt)
    model = create_model(opt)

    if not os.path.exists(opt.results_dir):
        os.mkdir(opt.results_dir)

    inference_result_v = None

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # model setup
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()

        output_cut = visuals['fake_B']
        output_cut_numpy = tensor2numpy(output_cut)
        orig_input = tensor2numpy(data['A'])

        inference_plot = np.hstack((orig_input, output_cut_numpy))

        if inference_result_v is None:
            inference_result_v = inference_plot
        else:
            inference_result_v = np.vstack((inference_result_v, inference_plot))

    cv2.imwrite(opt.results_dir + 'inference_result_v.png', inference_result_v)
    print("Inference complete, output can be found in results/ folder.")

