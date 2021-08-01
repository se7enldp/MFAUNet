import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os, nibabel
import sys, getopt
import PIL
from PIL import Image
import imageio
import scipy.misc
import numpy as np
import glob
from torch.utils import data
import torch
import random
from data.augmentations import Compose, RandomRotate, PaddingCenterCrop
from skimage import transform

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt
from collections import OrderedDict
import shutil
import os
import pickle
import json
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from data.augmentations import Compose, RandomSizedCrop, AdjustContrast, AdjustBrightness, RandomVerticallyFlip, RandomHorizontallyFlip, RandomRotate, PaddingCenterCrop
import os, glob
import argparse
from PIL import Image

def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats=False):
    if invert_image:
        data_sample = - data_sample
    if not per_channel:
        if retain_stats:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats:
            data_sample = data_sample - data_sample.mean() + mn
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
    else:
        for c in range(data_sample.shape[0]):
            if retain_stats:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats:
                data_sample[c] = data_sample[c] - data_sample[c].mean() + mn
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
    if invert_image:
        data_sample = - data_sample
    return data_sample


def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


class RV12Data(data.Dataset):

    def __init__(self,
                 root,
                 split='train',
                 augmentations=None,
                 target_size=(224, 256)
                 ):
        self.target_size = target_size
        self.ROOT_PATH = root
        self.split = split
        self.augmentations = augmentations
        self.TRAIN_IMG_PATH = os.path.join(root, 'train', 'img')
        self.TRAIN_SEG_PATH = os.path.join(root, 'train', 'gt')
        self.VAL_IMG_PATH = os.path.join(root, 'val', 'img')
        self.VAL_SEG_PATH = os.path.join(root, 'val', 'gt')
        self.train_num = len(os.listdir(self.TRAIN_IMG_PATH))
        self.val_num = len(os.listdir(self.VAL_IMG_PATH))
        self.full_img_path = []
        self.full_seg_path = []
        if self.split == 'train':
            self.full_img_path = self.TRAIN_IMG_PATH
            self.full_seg_path = self.TRAIN_SEG_PATH
        elif self.split == 'val':
            self.full_img_path = self.VAL_IMG_PATH
            self.full_seg_path = self.VAL_SEG_PATH
        # print('full_img_path:', full_img_path )
        # print('full_seg_path:', full_seg_path)
        self.images_names = os.listdir(self.full_img_path)
        self.labels_names = os.listdir(self.full_seg_path)

    def __len__(self):
        if self.split == "train":
            return self.train_num
        else:
            return self.val_num

    def __getitem__(self, i): # i is index

        img = Image.open(self.full_img_path + "/" + self.images_names[i])
        seg = Image.open(self.full_seg_path + "/" + self.labels_names[i])
        img_arr = np.asarray(img, dtype="float32")
        seg_arr = np.asarray(seg, dtype="float32")
        # image = img_arr
        # label = seg_arr
        # import matplotlib.pyplot as plt
        # # print(label.dtype)
        # # print(image.dtype)
        # plt.imshow(label, cmap='gray', clim=(0, 255))
        # # plt.imshow(image, cmap='gray', clim=(0, 255))
        #
        # plt.axis('off')
        # plt.show()

        # data[:, :, :] = img_arr
        # data[:, :, :] = seg_arr
        img = np.asarray(img_arr)[:, :, None]
        seg = np.asarray(seg_arr)[:, :, None]
        # print('img2.shape:', img.shape)
        # print('seg2.shape:', seg.shape)
        if self.augmentations is not None:
            img = img.transpose(2, 0, 1)
            seg = seg.transpose(2, 0, 1)
            # print('img3.shape:', img.shape)
            # print('seg3.shape:', seg.shape)
            for x in range(len(img)):
                # label = seg[x]
                image = img[x]
                import matplotlib.pyplot as plt
                # print(label.dtype)
                print(image.dtype)
                # plt.imshow(label, cmap='gray', clim=(0, 255))
                plt.imshow(image, cmap='gray', clim=(0, 255))

                plt.axis('off')
                plt.show()

            img_c = np.zeros((img.shape[0], self.target_size[0], self.target_size[1]))
            seg_c = np.zeros((seg.shape[0], self.target_size[0], self.target_size[1]))
            # print('img_c.shape:', img_c.shape)
            # print('seg_c.shape:', seg_c.shape)

            for z in range(img.shape[0]):
                if img[z].min() > 0:
                    img[z] -= img[z].min()

                img_tmp, seg_tmp = self.augmentations(img[z].astype(np.uint32), seg[z].astype(np.uint8))
                img_tmp = augment_gamma(img_tmp)
                # print('img_temp.shape:', img_tmp.shape)
                # print('seg_temp.shape:', seg_tmp.shape)

                mu = img_tmp.mean()
                sigma = img_tmp.std()
                img_tmp = (img_tmp - mu) / (sigma+1e-10)
                # print('img3.shape:', img_tmp.shape)
                # print('seg3.shape:', seg_tmp.shape)
                img_c[z] = img_tmp
                seg_c[z] = seg_tmp
                # img_c[z] = crop_or_pad_slice_to_size(img_tmp, 256, 256)
                # seg_c[z] = crop_or_pad_slice_to_size(seg_tmp, 256, 256)
                # print('img_c1.shape:', img_c.shape)
                # print('seg_c1.shape:', seg_c.shape)

            img = img_c.transpose(1, 2, 0)
            seg = seg_c.transpose(1, 2, 0)
            # print('img.shape:', img.shape)
            # print('seg.shape:', seg.shape)

        img = torch.from_numpy(img).float()
        seg = torch.from_numpy(seg).long()
        # print('img_tensor.shape:', img.shape)
        # print('seg_tensor.shape:', seg.shape)
        data_dict = {
            "image": img,
            "mask": seg,
        }
        # print('data_dict:', data_dict)
        return data_dict
        # return img, seg

    def _transform(self, img, mask):
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img, img), axis=0)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        # print('image4.size(): ', img.size())
        # print('mask4.size(): ', mask.size())
        return img, mask


class RV12_2DLoad():
    def __init__(self, dataset, split='train', deform=True):
        super(RV12_2DLoad, self).__init__()
        self.data = []
        # self.data_image = []
        # self.data_mask = []
        self.split = split
        self.deform = deform

        for x in range(dataset.__len__()):
            d = dataset[x]
            entry = {}
            entry["image"] = d["image"]
            entry["mask"] = d["mask"]
            self.data.append(entry)
            # image = dataset[x]['image']
            # label = dataset[x]['mask']
            #
            # self.data_image.append(image)
            # self.data_mask.append(label)

    def __getitem__(self, i):
        #return self.data[i]
        # print('dataset.len:', len(self.data))
        # print('index:', i)
        # img = self.data_image[i]
        # seg = self.data_mask[i]
        img = self.data[i]['image']
        seg = self.data[i]['mask']
        # print('img.shape:', img.shape)
        # print('seg.shape:', seg.shape)
        img = np.squeeze(img)
        seg = np.squeeze(seg)
        # print('squeeze_img.shape:', img.shape)
        # print('squeeze_seg.shape:', seg.shape)
        if self.split == 'train': #50% chance of deformation
            img = img.double().numpy()
            seg = seg.double().numpy()

            if random.uniform(0, 1.0) <= 0.5 and self.deform==True:
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
                seg = np.expand_dims(seg, axis=2)
                stacked = np.concatenate((img, seg), axis=2)
                red = self.random_elastic_deformation(stacked, alpha=500, sigma=20).transpose(2,0,1)
                img, seg = red[0], red[1]
            # print('img_train11.shape:', img.shape)
            # print('seg_train11.shape:', seg.shape)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
                img = np.concatenate((img, img, img), axis=0)
            # End Random Elastic Deformation
            #
            # print('img_train.shape:', img.shape)
            # print('seg_train.shape:', seg.shape)
            d = {"image": torch.from_numpy(img).float(),
                 # "mask": (torch.from_numpy(seg),
                 #          self.mask_to_edges(seg)),
                 "mask": torch.from_numpy(seg)}#########
            # print('d_train:', d)
            return d
            # image = torch.from_numpy(img).float()
            #
            # mask = torch.from_numpy(seg)
            # # print('img_tensor.shape111:', image.shape)
            # # print('seg_tensor.shape111:', mask.shape)
            # return image, mask

        elif self.split == 'val' or self.deform == False:
            img = img.unsqueeze(0)
            img = torch.cat([img, img, img], 0)
            # print('img_val.shape:', img.shape)
            # print('seg_val.shape:', seg.shape)
            d = {"image": img.float(),
                 # "mask": (seg, self.mask_to_edges(seg)),
                 "mask": seg
                 }
            return d
            # image = img.float()
            # mask = seg
            #
            # return image, mask

    def __len__(self):
        return len(self.data)

    def random_elastic_deformation(self, image, alpha, sigma, mode='nearest',
                                   random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
    ..  [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        assert len(image.shape) == 3

        if random_state is None:
            random_state = np.random.RandomState(None)

        height, width, channels = image.shape

        dx = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        indices = (np.repeat(np.ravel(x+dx), channels),
                np.repeat(np.ravel(y+dy), channels),
                np.tile(np.arange(channels), height*width))

        values = map_coordinates(image, indices, order=1, mode=mode)
        # print('value.shape:', values.shape)
        return values.reshape((height, width, channels))


if __name__ == '__main__':

    DATA_DIR = "G:\\python_project\\datasets\\RVSC"
    # augs = Compose([PaddingCenterCrop(256)])
    augs = Compose([PaddingCenterCrop([216, 256]),  RandomHorizontallyFlip(), RandomVerticallyFlip(), RandomRotate(180)])

    dataset = RV12Data(DATA_DIR, split='val', augmentations=augs)

    rv12 = RV12_2DLoad(dataset, split='val')
    dloader = torch.utils.data.DataLoader(rv12, batch_size=1)
    for idx, batch in enumerate(dloader):
        img, mask = batch['image'], batch['mask']
        print('mask:', mask.shape)
        print('img:', img.shape)

        print(idx)#, mask[0].max(), mask.min(), img.max(), img.min())
    # for idx, data in enumerate(dloader):
    #     image = data[0]
    #     mask = data[1]
    #     # print('mask.shape:', mask[0].shape, mask[1].shape)
    #     print('dloader.image.shape:', image.shape, mask.shape)
    #     print('idx:', idx)
