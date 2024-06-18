"""
Author: Arpit Aggarwal
Summary: Given the H&E image, it generates its corresponding CD163 IHC image
Output: it saves the cd163 brown segmentation mask for the H&E image as input in the 'results/' directory within the local folder
Command: python3 test.py --dataroot 'dataset_path' --name cd163 --model pix2pix
"""


# header files
import os
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import cv2

# run code on a h&e patch
if __name__ == '__main__':
    # set test options
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1

    # create dataset
    dataset = create_dataset(opt)

    # set model
    model = create_model(opt)
    model.setup(opt)

    # run and save binary mask
    print("Running model...")
    model.eval()
    for i, data in enumerate(dataset):
        print("Done with..." + str(i))
        im = data['A']
        A_paths = data['A_paths']
        brown_mask = np.zeros((int(im.shape[2]), int(im.shape[3])))
        ihc_image = np.zeros((int(im.shape[2]), int(im.shape[3]), 3))
        for index1 in range(0, int(im.shape[2])-249, 250):
            for index2 in range(0, int(im.shape[3])-249, 250):
                s_in1 = index1
                e_in1 = index1+256
                s_in2 = index2
                e_in2 = index2+256
                if e_in1 > int(im.shape[2]):
                    s_in1 = index1-6
                    e_in1 = index1+250
                if e_in2 > int(im.shape[3]):
                    s_in2 = index2-6
                    e_in2 = index2+250

                # read image and apply model
                A = im[:,:,s_in1:e_in1, s_in2:e_in2]
                dic = {'A': A, 'B': A, 'C': A, 'D': A, 'A_paths': A_paths}
                model.set_input(dic)
                model.test()
                visuals = model.get_current_visuals()
                img_path = model.get_image_paths()

                # cd163 ihc image
                fakeB_img = visuals['fake_B_1']
                fakeB_img = (fakeB_img*0.5+0.5)*255
                fakeB_img = np.array(fakeB_img[0,:,:,:].cpu())
                fakeB_img = np.transpose(fakeB_img, (1, 2, 0))
                fakeB_img = cv2.cvtColor(fakeB_img, cv2.COLOR_RGB2BGR)
                ihc_image[s_in1:e_in1, s_in2:e_in2, :] = fakeB_img
                cv2.imwrite("results/"+img_path[0].split("/")[-1], fakeB_img)

                # generate brown mask
                get_fake = cv2.imread("results/"+img_path[0].split("/")[-1])
                get_fake = cv2.cvtColor(get_fake, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(get_fake, (10, 75, 100), (20, 125, 175))
                brown_mask[s_in1:e_in1, s_in2:e_in2] = mask
        cv2.imwrite("results/"+img_path[0].split("/")[-1], brown_mask)
