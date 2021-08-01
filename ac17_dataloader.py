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


class AC17Data(data.Dataset):

    def __init__(self,
                 root,
                 split='train',
                 augmentations=None,
                 img_norm=True,
                 k=5,
                 k_split=1,
                 target_size=(256, 256)
                 ):
        self.target_size = target_size
        self.ROOT_PATH = root
        self.split = split
        self.k = k
        self.split_len = int(200/self.k)
        self.k_split = int(k_split)
        self.augmentations = augmentations
        self.TRAIN_IMG_PATH = os.path.join(root, 'train', 'img')
        self.TRAIN_SEG_PATH = os.path.join(root, 'train', 'seg')
        self.VAL_IMG_PATH = os.path.join(root, 'val', 'img')
        self.VAL_SEG_PATH = os.path.join(root, 'val', 'seg')
        self.list = self.read_files()

    def read_files(self):
        d = []
        txt_file = os.path.join('G:\\python_project\\(windows)shape-attentive-unet-master\\data\\data_series.txt')
        split_range = list(range((self.k_split-1)*self.split_len, self.k_split*self.split_len))
        with open(txt_file, 'r') as f:
            for i, line in enumerate(f):
                l = line.split(' ')
                key, val = int(l[0]), int(l[1])
                # append as train set
                if self.split == 'train' and i not in split_range:
                    d.append([key, val])

                #append as val set
                elif self.split == 'val' and i in split_range:
                    d.append([key, val])
                    if i > split_range[-1]: #do not need to iterate through rest of file
                        return d
        print('d', d)
        return d

    def __len__(self):
        if self.split == "train":
            return 200 - self.split_len
        else:
            return self.split_len

    def __getitem__(self, i): # i is index
        # print('self.list:', self.list[i])
        filename = "patient%03d_frame%02d" % (self.list[i][0], self.list[i][1])
        # print('filename:', filename)
        full_img_path = []
        full_seg_path = []
        if self.split == 'train':
            full_img_path = os.path.join(self.TRAIN_IMG_PATH, filename)
            full_seg_path = os.path.join(self.TRAIN_SEG_PATH, filename)
        elif self.split == 'val':
            full_img_path = os.path.join(self.VAL_IMG_PATH, filename)
            full_seg_path = os.path.join(self.VAL_SEG_PATH, filename)
        # print('full_img_path:', full_img_path )
        # print('full_seg_path:', full_seg_path)
        img = nibabel.load(full_img_path+".nii.gz")
        seg = nibabel.load(full_seg_path+"_gt.nii.gz").get_data()

        pix_dim = img.header.structarr['pixdim'][1]
        # pixel_dim = (img.header.structarr['pixdim'][1],#####################################
        #              img.header.structarr['pixdim'][2],
        #              img.header.structarr['pixdim'][3])
        img = np.array(img.get_data())
        seg = np.array(seg)
        # print('img1.shape:', img.shape)
        # print('seg1.shape:', seg.shape)
        pre_shape = img.shape[:-1]
        ratio = float(pix_dim/1.25)
        scale_vector = [ratio, ratio, 1]
        # target_resolution = (1.25, 1.25, 5)
        # scale_vector = [pixel_dim[0] / target_resolution[0],
        #                 pixel_dim[1] / target_resolution[1],
        #                 pixel_dim[2] / target_resolution[2]]
        img = transform.rescale(img,
                                scale_vector,
                                order=1,
                                preserve_range=True,
                                multichannel=False,
                                mode='constant')
        seg = transform.rescale(seg,
                                scale_vector,
                                order=0,
                                preserve_range=True,
                                multichannel=False,
                                mode='constant')
        # print('img2.shape:', img.shape)
        # print('seg2.shape:', seg.shape)
        if self.augmentations is not None:
            img = img.transpose(2, 0, 1)
            seg = seg.transpose(2, 0, 1)
            # print('img3.shape:', img.shape)
            # print('seg3.shape:', seg.shape)
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
        print('img_tensor.shape:', img.shape)
        print('seg_tensor.shape:', seg.shape)

        data_dict = {
            "name": filename,
            "image": img,
            "mask": seg,
        }
        # print('data_dict:', data_dict)
        return data_dict

    def _transform(self, img, mask):
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
            img = np.concatenate((img, img, img), axis=0)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        # print('image4.size(): ', img.size())
        # print('mask4.size(): ', mask.size())
        return img, mask


class AC17_2DLoad():
    def __init__(self, dataset, split='train', deform=True):
        super(AC17_2DLoad, self).__init__()
        self.data = []
        self.split = split
        self.deform = deform

        for i in range(dataset.__len__()):
            d = dataset[i]
            for x in range(d["image"].shape[-1]):
                entry = {}
                entry["image"] = d["image"].permute(2, 0, 1)[x]
                entry["mask"] = d["mask"].permute(2, 0, 1)[x]
                entry["name"] = d["name"] + "_z"+str(x)
                self.data.append(entry)
        # print('entry:', self.data)

    def __getitem__(self, i):
        # print('dataset.len:', len(self.data))
        # print('i:', i)
        #return self.data[i]
        img = self.data[i]["image"]
        seg = self.data[i]["mask"]
        # print('img.shape:', img.shape)
        # print('seg.shape:', seg.shape)

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
                 "mask": torch.from_numpy(seg), #########
                 "name": self.data[i]["name"]}
            # print('d_train:', d)
            return d

        elif self.split == 'val' or self.deform == False:
            img = img.unsqueeze(0)
            img = torch.cat([img, img, img], 0)
            # print('img_val.shape:', img.shape)
            # print('seg_val.shape:', seg.shape)
            d = {"image": img.float(),
                 # "mask": (seg, self.mask_to_edges(seg)),
                 "mask": seg,

                 "name": self.data[i]["name"]}
            # print('d_val:', d)
            return d

    def __len__(self):
        return len(self.data)

    def mask_to_onehot(self, mask, num_classes=3):
        _mask = [mask == i for i in range(1, num_classes+1)]
        _mask = [np.expand_dims(x, 0) for x in _mask]
        return np.concatenate(_mask, 0)

    def onehot_to_binary_edges(self, mask, radius=2, num_classes=3):
        if radius < 0:
            return mask

        # We need to pad the borders for boundary conditions
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

        edgemap = np.zeros(mask.shape[1:])

        for i in range(num_classes):
            dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap

    def mask_to_edges(self, mask):
        _edge = mask
        _edge = self.mask_to_onehot(_edge)
        _edge = self.onehot_to_binary_edges(_edge)
        return torch.from_numpy(_edge).float()

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

    DATA_DIR = "G:\\python_project\\datasets\\ACDC2017"
    # augs = Compose([PaddingCenterCrop(256)])
    augs = Compose([PaddingCenterCrop([216, 256]),  RandomHorizontallyFlip(), RandomVerticallyFlip(), RandomRotate(180)])

    dataset = AC17Data(DATA_DIR, split='train', augmentations=augs)

    ac17 = AC17_2DLoad(dataset, split='train')
    dloader = torch.utils.data.DataLoader(ac17, batch_size=1)
    for idx, batch in enumerate(dloader):
        img, mask = batch['image'], batch['mask']
        # print('mask:', mask.shape)
        # print('img:', img.shape)

        # print(idx)#, mask[0].max(), mask.min(), img.max(), img.min())

