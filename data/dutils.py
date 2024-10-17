'''
dutils.py
A utility library for customized data loading functions
The "d" in dutils is just to have a different naming to the other utils files
'''

import os
import gzip
import numpy as np
import pandas as pd

import os
import cv2
from typing import List, Union, Dict, Sequence
import numpy as np
import numpy.random as nprand
import datetime
import pandas as pd
import h5py
import torch
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d
import random
from torchvision import transforms as T
from torchvision import datasets

from PIL import Image

# =====================================================================================
# Moving MNIST
# =====================================================================================
# code from SimVP

def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'moving_mnist/train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)
    return mnist


def load_fixed_set(root):
    # Load the fixed dataset
    filename = 'moving_mnist/mnist_test_seq.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingMNIST(torch.utils.data.Dataset):
    def __init__(self, root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2],
                 transform=None):
        super(MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_fixed_set(root)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

        self.mean = 0
        self.std = 1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_,
                         self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(
                    data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        r = 1
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(
            0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        return input, output

    def __len__(self):
        return self.length


def load_mmnist_data(batch_size, val_batch_size, data_root, num_workers):

    train_set = MovingMNIST(root=data_root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2])
    test_set = MovingMNIST(root=data_root, is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, 

# =====================================================================================
# Motion Moving MNIST - Just a small change from MMNIST classes before (get random_tracjectory function, that is has 60% to move one more pixel, 30% to move 2 more pixel )
# =====================================================================================
def load_mmfixed_set(root):
    filename = 'moving_mnist/motion_mnist_seq.npy' # Gaussian Noise
    path = os.path.join(root, filename)
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset
    
class Motion_MovingMNIST(torch.utils.data.Dataset):
    def __init__(self, root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2],
                 transform=None):
        super(Motion_MovingMNIST, self).__init__()

        self.dataset = None
        if is_train:
            self.mnist = load_mnist(root)
        else:
            if num_objects[0] != 2:
                self.mnist = load_mnist(root)
            else:
                self.dataset = load_mmfixed_set(root)
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.is_train = is_train
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        # For generating data
        self.image_size_ = 64
        self.digit_size_ = 28
        self.step_length_ = 0.1

        self.mean = 0
        self.std = 1

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.            
            y += v_y * self.step_length_+random.gauss(0, 1/self.image_size_)
            x += v_x * self.step_length_+random.gauss(0, 1/self.image_size_)

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_,
                         self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total)
            ind = random.randint(0, self.mnist.shape[0] - 1)
            digit_image = self.mnist[ind]
            for i in range(self.n_frames_total):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size_
                right = left + self.digit_size_
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(
                    data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects[0] != 2:
            # Sample number of objects
            num_digits = random.choice(self.num_objects)
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            images = self.dataset[:, idx, ...]

        r = 1
        w = int(64 / r)
        images = images.reshape((length, w, r, w, r)).transpose(
            0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output / 255.0).contiguous().float()
        input = torch.from_numpy(input / 255.0).contiguous().float()
        return input, output

    def __len__(self):
        return self.length

def load_motion_mmnist_data(batch_size, val_batch_size, data_root, num_workers):

    train_set = Motion_MovingMNIST(root=data_root, is_train=True, n_frames_input=10, n_frames_output=10, num_objects=[2])
    test_set = Motion_MovingMNIST(root=data_root, is_train=False, n_frames_input=10, n_frames_output=10, num_objects=[2])

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std


# =====================================================================================
# KTH Dataset
# =====================================================================================
# Code from OpenSTL

class KTHDataset(torch.utils.data.Dataset):
    """KTH Action <https://ieeexplore.ieee.org/document/1334462>`_ Dataset"""

    def __init__(self, datas, indices, pre_seq_length, aft_seq_length, use_augment=False):
        super(KTHDataset,self).__init__()
        self.datas = datas.swapaxes(2, 3).swapaxes(1,2)
        self.indices = indices
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1

    def _augment_seq(self, imgs, crop_scale=0.95):
        """Augmentations for video"""
        _, _, h, w = imgs.shape  # original shape, e.g., [10, 3, 128, 128]
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = imgs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = imgs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            imgs = torch.flip(imgs, dims=(3, ))  # horizontal flip
        return imgs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end1 = begin + self.pre_seq_length
        end2 = begin + self.pre_seq_length + self.aft_seq_length
        data = torch.tensor(self.datas[begin:end1, ::]).float()
        labels = torch.tensor(self.datas[end1:end2, ::]).float()
        if self.use_augment:
            imgs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.95)
            data = imgs[:self.pre_seq_length, ...]
            labels = imgs[self.pre_seq_length:self.pre_seq_length+self.aft_seq_length, ...]
        return data, labels


class InputHandle(object):
    """Class for handling dataset inputs."""

    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        """Gets a mini-batch."""
        if self.no_batch_left():
            print(
                'There is no batch left in %s.'
                'Use iterators.begin() to rescan from the beginning.',
                self.name)
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width,
            self.image_width, 1)).astype(self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch


class DataProcess(object):
    """Class for preprocessing dataset inputs."""

    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.category_1 = ['boxing', 'handclapping', 'handwaving', 'walking']
        self.category_2 = ['jogging', 'running']
        self.category = self.category_1 + self.category_2
        self.image_width = input_param['image_width']

        self.train_person = [
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
            '13', '14', '15', '16'
        ]
        self.test_person = ['17', '18', '19', '20', '21', '22', '23', '24', '25']

        self.input_param = input_param
        self.seq_len = input_param['seq_length']

    def load_data(self, path, mode='train'):
        """Loads the dataset.
        Args:
            path: action_path.
            mode: Training or testing.
        Returns:
            A dataset and indices of the sequence.
        """
        # path = paths[0]
        if mode == 'train':
            person_id = self.train_person
        elif mode == 'test':
            person_id = self.test_person
        else:
            print('ERROR!')
        print('begin load data: ' + str(path))

        frames_np = []
        frames_file_name = []
        frames_person_mark = []
        frames_category = []
        person_mark = 0

        c_dir_list = self.category
        frame_category_flag = -1
        for c_dir in c_dir_list:  # handwaving
            if c_dir in self.category_1:
                frame_category_flag = 1  # 20 step
            elif c_dir in self.category_2:
                frame_category_flag = 2  # 3 step
            else:
                print('category error!!!')

            c_dir_path = os.path.join(path, c_dir)
            p_c_dir_list = os.listdir(c_dir_path)
            # p_c_dir_list.sort() # for date seq

            for p_c_dir in p_c_dir_list:  # person01_handwaving_d1_uncomp
                # print(p_c_dir)
                if p_c_dir[6:8] not in person_id:
                    continue
                person_mark += 1

                dir_path = os.path.join(c_dir_path, p_c_dir)
                filelist = os.listdir(dir_path)
                filelist.sort()  # tocheck
                for cur_file in filelist:  # image_0257
                    if not cur_file.startswith('image'):
                        continue

                    frame_im = Image.open(os.path.join(dir_path, cur_file))
                    frame_np = np.array(frame_im)  # (1000, 1000) numpy array
                    # print(frame_np.shape)
                    frame_np = frame_np[:, :, 0]  #
                    frames_np.append(frame_np)
                    frames_file_name.append(cur_file)
                    frames_person_mark.append(person_mark)
                    frames_category.append(frame_category_flag)

        # is it a begin index of sequence
        indices = []
        index = len(frames_person_mark) - 1
        while index >= self.seq_len - 1:
            if frames_person_mark[index] == frames_person_mark[index - self.seq_len + 1]:
                end = int(frames_file_name[index][6:10])
                start = int(frames_file_name[index - self.seq_len + 1][6:10])
                # TODO(yunbo): mode == 'test'
                if end - start == self.seq_len - 1:
                    indices.append(index - self.seq_len + 1)
                    if frames_category[index] == 1:
                        index -= self.seq_len - 1
                    elif frames_category[index] == 2:
                        index -= 2
                    else:
                        print('category error 2 !!!')
            index -= 1

        frames_np = np.asarray(frames_np)
        data = np.zeros((frames_np.shape[0], self.image_width, self.image_width, 1))
        for i in range(len(frames_np)):
            temp = np.float32(frames_np[i, :, :])
            data[i, :, :, 0] = cv2.resize(temp, (self.image_width, self.image_width)) / 255
        print('there are ' + str(data.shape[0]) + ' pictures')
        print('there are ' + str(len(indices)) + ' sequences')
        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)


def load_kth_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=20, in_shape=[10, 1, 128, 128],
              distributed=False, use_augment=False, use_prefetcher=False):

    img_width = in_shape[-1] if in_shape is not None else 128
    # pre_seq_length, aft_seq_length = 10, 10
    input_param = {
        'paths': os.path.join(data_root, 'kth'),
        'image_width': img_width,
        'minibatch_size': batch_size,
        'seq_length': (pre_seq_length + aft_seq_length),
        'input_data_type': 'float32',
        'name': 'kth'
    }
    input_handle = DataProcess(input_param)
    train_input_handle = input_handle.get_train_input_handle()
    test_input_handle = input_handle.get_test_input_handle()

    train_set = KTHDataset(train_input_handle.datas,
                           train_input_handle.indices,
                           pre_seq_length,
                           aft_seq_length, use_augment=use_augment)
    test_set = KTHDataset(test_input_handle.datas,
                          test_input_handle.indices,
                          pre_seq_length,
                          aft_seq_length, use_augment=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

# =====================================================================================
# HKO-7 data
# =====================================================================================
def pixel_to_dBZ_nonlinear(img):
    '''
    [0, 255] OR [0, 1] pixel => [0, 80] dBZ
    '''
    if img.mean() > 1.0:
        img = img / 255.0
    ashift = 31.0
    afact = 4.0
    atan_dBZ_min = -1.482
    atan_dBZ_max = 1.412
    tan_pix = np.tan(img * (atan_dBZ_max - atan_dBZ_min) + atan_dBZ_min) 
    return tan_pix * afact + ashift

def dbZ_to_pixel_nonlinear(dbZ):
    '''
    [0, 80] dBZ => [0, 255] OR [0, 1] pixel
    '''
    ashift = 31.0
    afact = 4.0
    atan_dBZ_min = -1.482
    atan_dBZ_max = 1.412
    dbZ_adjusted = (dbZ - ashift) / afact
    return (np.arctan(dbZ_adjusted) - atan_dBZ_min) / (atan_dBZ_max - atan_dBZ_min)

def dbZ_to_pixel(dbZ):
    '''
    [0, 80] dbZ => [0, 1] pixel
    '''
    return np.floor((dbZ + 10) * 255 / 70 + 0.5) / 255.0

def pixel_to_dBZ(pixel):
    '''
    [0, 255] (or [0, 1]) pixel => [0, 80] dBZ
    '''
    if pixel.mean() > 1.0:
        pixel = pixel / 255.0
    return (70 * pixel) - 10

def nonlinear_to_linear(im):
    return dbZ_to_pixel(pixel_to_dBZ_nonlinear(im))

def nonlinear_to_linear_batched(seq, datetime):
    seq_linear = np.zeros_like(seq)
    for i, (seq_b, dt_b) in enumerate(zip(seq, datetime)):
        if dt_b[0].year >= 2016:
            seq_linear[i] = nonlinear_to_linear(seq_b)
        else:
            seq_linear[i] = seq_b
    seq_linear = np.clip(seq_linear, 0.0, 1.0)
    return seq_linear

def linear_to_nonlinear(im):
    return dbZ_to_pixel_nonlinear(pixel_to_dBZ(im))

def linear_to_nonlinear_batched(seq, datetime):
    seq_nonlinear = np.zeros_like(seq)
    for i, (seq_b, dt_b) in enumerate(zip(seq, datetime)):
        if dt_b[0].year < 2016:
            seq_nonlinear[i] = linear_to_nonlinear(seq_b)
        else:
            seq_nonlinear[i] = seq_b
    seq_nonlinear = np.clip(seq_nonlinear, 0.0, 1.0)
    return seq_nonlinear

# Crop Radar Image in NE Coordinate
# Input Shape : [Batch_size, ImageY, ImageX, Seq_len, channel_no]
# Output Shape : [Batch_size, ImageY, ImageX, Seq_len, channel_no]
def crop_radar(rad_dataBatch, convertNE=False, radius=64): 
    '''
    To crop the 2radius x 2radius centered at the radar
    '''
    batch_size, ly, lx, seq_len, channel_no = rad_dataBatch.shape
    cx = lx//2
    cy = ly//2

    if convertNE:
        output = []
        for batch in rad_dataBatch:
            batch_ch = []
            for chno in range(channel_no):
                convertBatch = [ConvertRadImagetoNEcoordinate(batch[:,:,i,chno], xsize=radius*2, ysize=radius*2) for i in range(seq_len)]
                batch_ch.append(convertBatch)
            output.append(batch_ch)
        output = np.array(output)
        output = output.transpose((0, 3, 4, 2, 1))
    else:
        output = rad_dataBatch[:, cy-radius:cy+radius, cx-radius:cx+radius]
    return output

def find_closest(arr, val):
    idx = np.abs(arr - val).argmin()
    return arr[idx], idx


# =====================================================================================
# SEVIR data
# Code is adapted from https://github.com/MIT-AI-Accelerator/neurips-2020-sevir. Their license is MIT License.
# (From Earthformer's implementation)
# =====================================================================================

# SEVIR Dataset constants
SEVIR_DATA_TYPES = ['vis', 'ir069', 'ir107', 'vil', 'lght']
SEVIR_RAW_DTYPES = {'vis': np.int16,
                    'ir069': np.int16,
                    'ir107': np.int16,
                    'vil': np.uint8,
                    'lght': np.int16}
LIGHTING_FRAME_TIMES = np.arange(- 120.0, 125.0, 5) * 60
SEVIR_DATA_SHAPE = {'lght': (48, 48), }
PREPROCESS_SCALE_SEVIR = {'vis': 1,  # Not utilized in original paper
                          'ir069': 1 / 1174.68,
                          'ir107': 1 / 2562.43,
                          'vil': 1 / 47.54,
                          'lght': 1 / 0.60517}
PREPROCESS_OFFSET_SEVIR = {'vis': 0,  # Not utilized in original paper
                           'ir069': 3683.58,
                           'ir107': 1552.80,
                           'vil': - 33.44,
                           'lght': - 0.02990}
PREPROCESS_SCALE_01 = {'vis': 1,
                       'ir069': 1,
                       'ir107': 1,
                       'vil': 1 / 255,  # currently the only one implemented
                       'lght': 1}
PREPROCESS_OFFSET_01 = {'vis': 0,
                        'ir069': 0,
                        'ir107': 0,
                        'vil': 0,  # currently the only one implemented
                        'lght': 0}

# sevir
SEVIR_ROOT_DIR = "data/sevir"
SEVIR_CATALOG = os.path.join(SEVIR_ROOT_DIR, "CATALOG.csv")
SEVIR_DATA_DIR = os.path.join(SEVIR_ROOT_DIR, "data")
SEVIR_RAW_SEQ_LEN = 49

SEVIR_TRAIN_VAL_SPLIT_DATE = datetime.datetime(2019, 1, 1)
SEVIR_TRAIN_TEST_SPLIT_DATE = datetime.datetime(2019, 6, 1)

def change_layout_np(data,
                     in_layout='NHWT', out_layout='NHWT',
                     ret_contiguous=False):
    # first convert to 'NHWT'
    if in_layout == 'NHWT':
        pass
    elif in_layout == 'NTHW':
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NWHT':
        data = np.transpose(data,
                            axes=(0, 2, 1, 3))
    elif in_layout == 'NTCHW':
        data = data[:, :, 0, :, :]
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NTHWC':
        data = data[:, :, :, :, 0]
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NTWHC':
        data = data[:, :, :, :, 0]
        data = np.transpose(data,
                            axes=(0, 3, 2, 1))
    elif in_layout == 'TNHW':
        data = np.transpose(data,
                            axes=(1, 2, 3, 0))
    elif in_layout == 'TNCHW':
        data = data[:, :, 0, :, :]
        data = np.transpose(data,
                            axes=(1, 2, 3, 0))
    else:
        raise NotImplementedError

    if out_layout == 'NHWT':
        pass
    elif out_layout == 'NTHW':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
    elif out_layout == 'NWHT':
        data = np.transpose(data,
                            axes=(0, 2, 1, 3))
    elif out_layout == 'NTCHW':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
        data = np.expand_dims(data, axis=2)
    elif out_layout == 'NTHWC':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
        data = np.expand_dims(data, axis=-1)
    elif out_layout == 'NTWHC':
        data = np.transpose(data,
                            axes=(0, 3, 2, 1))
        data = np.expand_dims(data, axis=-1)
    elif out_layout == 'TNHW':
        data = np.transpose(data,
                            axes=(3, 0, 1, 2))
    elif out_layout == 'TNCHW':
        data = np.transpose(data,
                            axes=(3, 0, 1, 2))
        data = np.expand_dims(data, axis=2)
    else:
        raise NotImplementedError
    if ret_contiguous:
        data = data.ascontiguousarray()
    return data

def change_layout_torch(data,
                        in_layout='NHWT', out_layout='NHWT',
                        ret_contiguous=False):
    # first convert to 'NHWT'
    if in_layout == 'NHWT':
        pass
    elif in_layout == 'NTHW':
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'NTCHW':
        data = data[:, :, 0, :, :]
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'NTHWC':
        data = data[:, :, :, :, 0]
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'TNHW':
        data = data.permute(1, 2, 3, 0)
    elif in_layout == 'TNCHW':
        data = data[:, :, 0, :, :]
        data = data.permute(1, 2, 3, 0)
    else:
        raise NotImplementedError

    if out_layout == 'NHWT':
        pass
    elif out_layout == 'NTHW':
        data = data.permute(0, 3, 1, 2)
    elif out_layout == 'NTCHW':
        data = data.permute(0, 3, 1, 2)
        data = torch.unsqueeze(data, dim=2)
    elif out_layout == 'NTHWC':
        data = data.permute(0, 3, 1, 2)
        data = torch.unsqueeze(data, dim=-1)
    elif out_layout == 'TNHW':
        data = data.permute(3, 0, 1, 2)
    elif out_layout == 'TNCHW':
        data = data.permute(3, 0, 1, 2)
        data = torch.unsqueeze(data, dim=2)
    else:
        raise NotImplementedError
    if ret_contiguous:
        data = data.contiguous()
    return data

class SEVIRDataLoader:
    r"""
    DataLoader that loads SEVIR sequences, and spilts each event
    into segments according to specified sequence length.

    Event Frames:
        [-----------------------raw_seq_len----------------------]
        [-----seq_len-----]
        <--stride-->[-----seq_len-----]
                    <--stride-->[-----seq_len-----]
                                        ...
    """
    def __init__(self,
                 data_types: Sequence[str] = None,
                 seq_len: int = 49,
                 raw_seq_len: int = 49,
                 sample_mode: str = 'sequent',
                 stride: int = 12,
                 batch_size: int = 1,
                 layout: str = 'NHWT',
                 num_shard: int = 1,
                 rank: int = 0,
                 split_mode: str = "uneven",
                 sevir_catalog: Union[str, pd.DataFrame] = None,
                 sevir_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 datetime_filter=None,
                 catalog_filter='default',
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type=np.float32,
                 preprocess: bool = True,
                 rescale_method: str = '01',
                 downsample_dict: Dict[str, Sequence[int]] = None,
                 verbose: bool = False):
        r"""
        Parameters
        ----------
        data_types
            A subset of SEVIR_DATA_TYPES.
        seq_len
            The length of the data sequences. Should be smaller than the max length raw_seq_len.
        raw_seq_len
            The length of the raw data sequences.
        sample_mode
            'random' or 'sequent'
        stride
            Useful when sample_mode == 'sequent'
            stride must not be smaller than out_len to prevent data leakage in testing.
        batch_size
            Number of sequences in one batch.
        layout
            str: consists of batch_size 'N', seq_len 'T', channel 'C', height 'H', width 'W'
            The layout of sampled data. Raw data layout is 'NHWT'.
            valid layout: 'NHWT', 'NTHW', 'NTCHW', 'TNHW', 'TNCHW'.
        num_shard
            Split the whole dataset into num_shard parts for distributed training.
        rank
            Rank of the current process within num_shard.
        split_mode: str
            if 'ceil', all `num_shard` dataloaders have the same length = ceil(total_len / num_shard).
            Different dataloaders may have some duplicated data batches, if the total size of datasets is not divided by num_shard.
            if 'floor', all `num_shard` dataloaders have the same length = floor(total_len / num_shard).
            The last several data batches may be wasted, if the total size of datasets is not divided by num_shard.
            if 'uneven', the last datasets has larger length when the total length is not divided by num_shard.
            The uneven split leads to synchronization error in dist.all_reduce() or dist.barrier().
            See related issue: https://github.com/pytorch/pytorch/issues/33148
            Notice: this also affects the behavior of `self.use_up`.
        sevir_catalog
            Name of SEVIR catalog CSV file.
        sevir_data_dir
            Directory path to SEVIR data.
        start_date
            Start time of SEVIR samples to generate.
        end_date
            End time of SEVIR samples to generate.
        datetime_filter
            function
            Mask function applied to time_utc column of catalog (return true to keep the row).
            Pass function of the form   lambda t : COND(t)
            Example:  lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21)  # Generate only day-time events
        catalog_filter
            function or None or 'default'
            Mask function applied to entire catalog dataframe (return true to keep row).
            Pass function of the form lambda catalog:  COND(catalog)
            Example:  lambda c:  [s[0]=='S' for s in c.id]   # Generate only the 'S' events
        shuffle
            bool, If True, data samples are shuffled before each epoch.
        shuffle_seed
            int, Seed to use for shuffling.
        output_type
            np.dtype, dtype of generated tensors
        preprocess
            bool, If True, self.preprocess_data_dict(data_dict) is called before each sample generated
        downsample_dict:
            dict, downsample_dict.keys() == data_types. downsample_dict[key] is a Sequence of (t_factor, h_factor, w_factor),
            representing the downsampling factors of all dimensions.
        verbose
            bool, verbose when opening raw data files
        """
        super(SEVIRDataLoader, self).__init__()
        if sevir_catalog is None:
            sevir_catalog = SEVIR_CATALOG
        if sevir_data_dir is None:
            sevir_data_dir = SEVIR_DATA_DIR
        if data_types is None:
            data_types = SEVIR_DATA_TYPES
        else:
            assert set(data_types).issubset(SEVIR_DATA_TYPES)

        # configs which should not be modified
        self._dtypes = SEVIR_RAW_DTYPES
        self.lght_frame_times = LIGHTING_FRAME_TIMES
        self.data_shape = SEVIR_DATA_SHAPE

        self.raw_seq_len = raw_seq_len
        assert seq_len <= self.raw_seq_len, f'seq_len must not be larger than raw_seq_len = {raw_seq_len}, got {seq_len}.'
        self.seq_len = seq_len
        assert sample_mode in ['random', 'sequent'], f'Invalid sample_mode = {sample_mode}, must be \'random\' or \'sequent\'.'
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        valid_layout = ('NHWT', 'NTHW', 'NTCHW', 'NTHWC', 'TNHW', 'TNCHW')
        if layout not in valid_layout:
            raise ValueError(f'Invalid layout = {layout}! Must be one of {valid_layout}.')
        self.layout = layout
        self.num_shard = num_shard
        self.rank = rank
        valid_split_mode = ('ceil', 'floor', 'uneven')
        if split_mode not in valid_split_mode:
            raise ValueError(f'Invalid split_mode: {split_mode}! Must be one of {valid_split_mode}.')
        self.split_mode = split_mode
        self._samples = None
        self._hdf_files = {}
        self.data_types = data_types
        if isinstance(sevir_catalog, str):
            self.catalog = pd.read_csv(sevir_catalog, parse_dates=['time_utc'], low_memory=False)
        else:
            self.catalog = sevir_catalog
        self.sevir_data_dir = sevir_data_dir
        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = start_date
        self.end_date = end_date
        self.shuffle = shuffle
        self.shuffle_seed = int(shuffle_seed)
        self.output_type = output_type
        self.preprocess = preprocess
        self.downsample_dict = downsample_dict
        self.rescale_method = rescale_method
        self.verbose = verbose

        if self.start_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date]
        if self.end_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if self.datetime_filter:
            self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]

        if self.catalog_filter is not None:
            if self.catalog_filter == 'default':
                self.catalog_filter = lambda c: c.pct_missing == 0
            self.catalog = self.catalog[self.catalog_filter(self.catalog)]

        self._compute_samples()
        self._open_files(verbose=self.verbose)
        self.reset()

    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets self._samples
        """
        # locate all events containing colocated data_types
        imgt = self.data_types
        imgts = set(imgt)
        filtcat = self.catalog[ np.logical_or.reduce([self.catalog.img_type==i for i in imgt]) ]
        # remove rows missing one or more requested img_types
        filtcat = filtcat.groupby('id').filter(lambda x: imgts.issubset(set(x['img_type'])))
        # If there are repeated IDs, remove them (this is a bug in SEVIR)
        # TODO: is it necessary to keep one of them instead of deleting them all
        filtcat = filtcat.groupby('id').filter(lambda x: x.shape[0]==len(imgt))
        self._samples = filtcat.groupby('id').apply(lambda df: self._df_to_series(df,imgt) )
        if self.shuffle:
            self.shuffle_samples()

    def shuffle_samples(self):
        self._samples = self._samples.sample(frac=1, random_state=self.shuffle_seed)

    def _df_to_series(self, df, imgt):
        d = {}
        df = df.set_index('img_type')
        for i in imgt:
            s = df.loc[i]
            idx = s.file_index if i != 'lght' else s.id
            d.update({f'{i}_filename': [s.file_name],
                      f'{i}_index': [idx]})

        return pd.DataFrame(d)

    def _open_files(self, verbose=True):
        """
        Opens HDF files
        """
        imgt = self.data_types
        hdf_filenames = []
        for t in imgt:
            hdf_filenames += list(np.unique( self._samples[f'{t}_filename'].values ))
        self._hdf_files = {}
        for f in hdf_filenames:
            if verbose:
                print('Opening HDF5 file for reading', f)
            self._hdf_files[f] = h5py.File(self.sevir_data_dir + '/' + f, 'r')

    def close(self):
        """
        Closes all open file handles
        """
        for f in self._hdf_files:
            self._hdf_files[f].close()
        self._hdf_files = {}

    @property
    def num_seq_per_event(self):
        return 1 + (self.raw_seq_len - self.seq_len) // self.stride

    @property
    def total_num_seq(self):
        """
        The total number of sequences within each shard.
        Notice that it is not the product of `self.num_seq_per_event` and `self.total_num_event`.
        """
        return int(self.num_seq_per_event * self.num_event)

    @property
    def total_num_event(self):
        """
        The total number of events in the whole dataset, before split into different shards.
        """
        return int(self._samples.shape[0])

    @property
    def start_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx >= start_event_idx
        """
        return self.total_num_event // self.num_shard * self.rank

    @property
    def end_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx < end_event_idx

        """
        if self.split_mode == 'ceil':
            _last_start_event_idx = self.total_num_event // self.num_shard * (self.num_shard - 1)
            _num_event = self.total_num_event - _last_start_event_idx
            return self.start_event_idx + _num_event
        elif self.split_mode == 'floor':
            return self.total_num_event // self.num_shard * (self.rank + 1)
        else:  # self.split_mode == 'uneven':
            if self.rank == self.num_shard - 1:  # the last process
                return self.total_num_event
            else:
                return self.total_num_event // self.num_shard * (self.rank + 1)

    @property
    def num_event(self):
        """
        The number of events split into each rank
        """
        return self.end_event_idx - self.start_event_idx

    def _read_data(self, row, data):
        """
        Iteratively read data into data dict. Finally data[imgt] gets shape (batch_size, height, width, raw_seq_len).

        Parameters
        ----------
        row
            A series with fields IMGTYPE_filename, IMGTYPE_index, IMGTYPE_time_index.
        data
            Dict, data[imgt] is a data tensor with shape = (tmp_batch_size, height, width, raw_seq_len).

        Returns
        -------
        data
            Updated data. Updated shape = (tmp_batch_size + 1, height, width, raw_seq_len).
        """
        imgtyps = np.unique([x.split('_')[0] for x in list(row.keys())])
        for t in imgtyps:
            fname = row[f'{t}_filename']
            idx = row[f'{t}_index']
            t_slice = slice(0, None)
            # Need to bin lght counts into grid
            if t == 'lght':
                lght_data = self._hdf_files[fname][idx][:]
                data_i = self._lght_to_grid(lght_data, t_slice)
            else:
                data_i = self._hdf_files[fname][t][idx:idx + 1, :, :, t_slice]
            data[t] = np.concatenate((data[t], data_i), axis=0) if (t in data) else data_i

        return data

    def _lght_to_grid(self, data, t_slice=slice(0, None)):
        """
        Converts Nx5 lightning data matrix into a 2D grid of pixel counts
        """
        # out_size = (48,48,len(self.lght_frame_times)-1) if isinstance(t_slice,(slice,)) else (48,48)
        out_size = (*self.data_shape['lght'], len(self.lght_frame_times)) if t_slice.stop is None else (*self.data_shape['lght'], 1)
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # filter out points outside the grid
        x, y = data[:, 3], data[:, 4]
        m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
        data = data[m, :]
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # Filter/separate times
        t = data[:, 0]
        if t_slice.stop is not None:  # select only one time bin
            if t_slice.stop > 0:
                if t_slice.stop < len(self.lght_frame_times):
                    tm = np.logical_and(t >= self.lght_frame_times[t_slice.stop - 1],
                                        t < self.lght_frame_times[t_slice.stop])
                else:
                    tm = t >= self.lght_frame_times[-1]
            else:  # special case:  frame 0 uses lght from frame 1
                tm = np.logical_and(t >= self.lght_frame_times[0], t < self.lght_frame_times[1])
            # tm=np.logical_and( (t>=FRAME_TIMES[t_slice],t<FRAME_TIMES[t_slice+1]) )

            data = data[tm, :]
            z = np.zeros(data.shape[0], dtype=np.int64)
        else:  # compute z coordinate based on bin location times
            z = np.digitize(t, self.lght_frame_times) - 1
            z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

        x = data[:, 3].astype(np.int64)
        y = data[:, 4].astype(np.int64)

        k = np.ravel_multi_index(np.array([y, x, z]), out_size)
        n = np.bincount(k, minlength=np.prod(out_size))
        return np.reshape(n, out_size).astype(np.int16)[np.newaxis, :]

    def _old_save_downsampled_dataset(self, save_dir, downsample_dict, verbose=True):
        """
        This method does not save .h5 dataset correctly. There are some batches missed due to unknown error.
        E.g., the first converted .h5 file `SEVIR_VIL_RANDOMEVENTS_2017_0501_0831.h5` only has batch_dim = 1414,
        while it should be 1440 in the original .h5 file.
        """
        import os
        from skimage.measure import block_reduce
        assert not os.path.exists(save_dir), f"save_dir {save_dir} already exists!"
        os.makedirs(save_dir)
        sample_counter = 0
        for index, row in self._samples.iterrows():
            if verbose:
                print(f"Downsampling {sample_counter}-th data item.", end='\r')
            for data_type in self.data_types:
                fname = row[f'{data_type}_filename']
                idx = row[f'{data_type}_index']
                t_slice = slice(0, None)
                if data_type == 'lght':
                    lght_data = self._hdf_files[fname][idx][:]
                    data_i = self._lght_to_grid(lght_data, t_slice)
                else:
                    data_i = self._hdf_files[fname][data_type][idx:idx + 1, :, :, t_slice]
                # Downsample t
                t_slice = [slice(None, None), ] * 4
                t_slice[-1] = slice(None, None, downsample_dict[data_type][0])  # layout = 'NHWT'
                data_i = data_i[tuple(t_slice)]
                # Downsample h, w
                data_i = block_reduce(data_i,
                                      block_size=(1, *downsample_dict[data_type][1:], 1),
                                      func=np.max)
                # Save as new .h5 file
                new_file_path = os.path.join(save_dir, fname)
                if not os.path.exists(new_file_path):
                    if not os.path.exists(os.path.dirname(new_file_path)):
                        os.makedirs(os.path.dirname(new_file_path))
                    # Create dataset
                    with h5py.File(new_file_path, 'w') as hf:
                        hf.create_dataset(
                            data_type, data=data_i,
                            maxshape=(None, *data_i.shape[1:]))
                else:
                    # Append
                    with h5py.File(new_file_path, 'a') as hf:
                        hf[data_type].resize((hf[data_type].shape[0] + data_i.shape[0]), axis=0)
                        hf[data_type][-data_i.shape[0]:] = data_i

            sample_counter += 1

    def save_downsampled_dataset(self, save_dir, downsample_dict, verbose=True):
        """
        Parameters
        ----------
        save_dir
        downsample_dict:    Dict[Sequence[int]]
            Notice that this is different from `self.downsample_dict`, which is used during runtime.
        """
        import os
        from skimage.measure import block_reduce
        from ...utils.utils import path_splitall
        assert not os.path.exists(save_dir), f"save_dir {save_dir} already exists!"
        os.makedirs(save_dir)
        for fname, hdf_file in self._hdf_files.items():
            if verbose:
                print(f"Downsampling data in {fname}.")
            data_type = path_splitall(fname)[0]
            if data_type == 'lght':
                # TODO: how to get idx?
                raise NotImplementedError
                # lght_data = self._hdf_files[fname][idx][:]
                # t_slice = slice(0, None)
                # data_i = self._lght_to_grid(lght_data, t_slice)
            else:
                data_i = self._hdf_files[fname][data_type]
            # Downsample t
            t_slice = [slice(None, None), ] * 4
            t_slice[-1] = slice(None, None, downsample_dict[data_type][0])  # layout = 'NHWT'
            data_i = data_i[tuple(t_slice)]
            # Downsample h, w
            data_i = block_reduce(data_i,
                                  block_size=(1, *downsample_dict[data_type][1:], 1),
                                  func=np.max)
            # Save as new .h5 file
            new_file_path = os.path.join(save_dir, fname)
            if not os.path.exists(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))
            # Create dataset
            with h5py.File(new_file_path, 'w') as hf:
                hf.create_dataset(
                    data_type, data=data_i,
                    maxshape=(None, *data_i.shape[1:]))

    @property
    def sample_count(self):
        """
        Record how many times self.__next__() is called.
        """
        return self._sample_count

    def inc_sample_count(self):
        self._sample_count += 1

    @property
    def curr_event_idx(self):
        return self._curr_event_idx

    @property
    def curr_seq_idx(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self._curr_seq_idx

    def set_curr_event_idx(self, val):
        self._curr_event_idx = val

    def set_curr_seq_idx(self, val):
        """
        Used only when self.sample_mode == 'sequent'
        """
        self._curr_seq_idx = val

    def reset(self, shuffle: bool = None):
        self.set_curr_event_idx(val=self.start_event_idx)
        self.set_curr_seq_idx(0)
        self._sample_count = 0
        if shuffle is None:
            shuffle = self.shuffle
        if shuffle:
            self.shuffle_samples()

    def __len__(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self.total_num_seq // self.batch_size

    @property
    def use_up(self):
        """
        Check if dataset is used up in 'sequent' mode.
        """
        if self.sample_mode == 'random':
            return False
        else:   # self.sample_mode == 'sequent'
            # compute the remaining number of sequences in current event
            curr_event_remain_seq = self.num_seq_per_event - self.curr_seq_idx
            all_remain_seq = curr_event_remain_seq + (
                        self.end_event_idx - self.curr_event_idx - 1) * self.num_seq_per_event
            if self.split_mode == "floor":
                # This approach does not cover all available data, but avoid dealing with masks
                return all_remain_seq < self.batch_size
            else:
                return all_remain_seq <= 0

    def _load_event_batch(self, event_idx, event_batch_size):
        """
        Loads a selected batch of events (not batch of sequences) into memory.

        Parameters
        ----------
        idx
        event_batch_size
            event_batch[i] = all_type_i_available_events[idx:idx + event_batch_size]
        Returns
        -------
        event_batch
            list of event batches.
            event_batch[i] is the event batch of the i-th data type.
            Each event_batch[i] is a np.ndarray with shape = (event_batch_size, height, width, raw_seq_len)
        """
        event_idx_slice_end = event_idx + event_batch_size
        pad_size = 0
        if event_idx_slice_end > self.end_event_idx:
            pad_size = event_idx_slice_end - self.end_event_idx
            event_idx_slice_end = self.end_event_idx
        pd_batch = self._samples.iloc[event_idx:event_idx_slice_end]
        data = {}
        for index, row in pd_batch.iterrows():
            data = self._read_data(row, data)
        if pad_size > 0:
            event_batch = []
            for t in self.data_types:
                pad_shape = [pad_size, ] + list(data[t].shape[1:])
                data_pad = np.concatenate((data[t].astype(self.output_type),
                                           np.zeros(pad_shape, dtype=self.output_type)),
                                          axis=0)
                event_batch.append(data_pad)
        else:
            event_batch = [data[t].astype(self.output_type) for t in self.data_types]
        return event_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_mode == 'random':
            self.inc_sample_count()
            ret_dict = self._random_sample()
        else:
            if self.use_up:
                raise StopIteration
            else:
                self.inc_sample_count()
                ret_dict = self._sequent_sample()
        ret_dict = self.data_dict_to_tensor(data_dict=ret_dict,
                                            data_types=self.data_types)
        if self.preprocess:
            ret_dict = self.preprocess_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 layout=self.layout,
                                                 rescale=self.rescale_method)
        if self.downsample_dict is not None:
            ret_dict = self.downsample_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 factors_dict=self.downsample_dict,
                                                 layout=self.layout)
        return ret_dict

    def __getitem__(self, index):
        data_dict = self._idx_sample(index=index)
        return data_dict

    @staticmethod
    def preprocess_data_dict(data_dict, data_types=None, layout='NHWT', rescale='01'):
        """
        Parameters
        ----------
        data_dict:  Dict[str, Union[np.ndarray, torch.Tensor]]
        data_types: Sequence[str]
            The data types that we want to rescale. This mainly excludes "mask" from preprocessing.
        layout: str
            consists of batch_size 'N', seq_len 'T', channel 'C', height 'H', width 'W'
        rescale:    str
            'sevir': use the offsets and scale factors in original implementation.
            '01': scale all values to range 0 to 1, currently only supports 'vil'
        Returns
        -------
        data_dict:  Dict[str, Union[np.ndarray, torch.Tensor]]
            preprocessed data
        """
        if rescale == 'sevir':
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == '01':
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f'Invalid rescale option: {rescale}.')
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, np.ndarray):
                    data = scale_dict[key] * (
                            data.astype(np.float32) +
                            offset_dict[key])
                    data = change_layout_np(data=data,
                                            in_layout='NHWT',
                                            out_layout=layout)
                elif isinstance(data, torch.Tensor):
                    data = scale_dict[key] * (
                            data.float() +
                            offset_dict[key])
                    data = change_layout_torch(data=data,
                                               in_layout='NHWT',
                                               out_layout=layout)
                data_dict[key] = data
        return data_dict

    @staticmethod
    def process_data_dict_back(data_dict, data_types=None, rescale='01'):
        """
        Parameters
        ----------
        data_dict
            each data_dict[key] is a torch.Tensor.
        rescale
            str:
                'sevir': data are scaled using the offsets and scale factors in original implementation.
                '01': data are all scaled to range 0 to 1, currently only supports 'vil'
        Returns
        -------
        data_dict
            each data_dict[key] is the data processed back in torch.Tensor.
        """
        if rescale == 'sevir':
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == '01':
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f'Invalid rescale option: {rescale}.')
        if data_types is None:
            data_types = data_dict.keys()
        for key in data_types:
            data = data_dict[key]
            data = data.float() / scale_dict[key] - offset_dict[key]
            data_dict[key] = data
        return data_dict

    @staticmethod
    def data_dict_to_tensor(data_dict, data_types=None):
        """
        Convert each element in data_dict to torch.Tensor (copy without grad).
        """
        ret_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, torch.Tensor):
                    ret_dict[key] = data.detach().clone()
                elif isinstance(data, np.ndarray):
                    ret_dict[key] = torch.from_numpy(data)
                else:
                    raise ValueError(f"Invalid data type: {type(data)}. Should be torch.Tensor or np.ndarray")
            else:   # key == "mask"
                ret_dict[key] = data
        return ret_dict

    @staticmethod
    def downsample_data_dict(data_dict, data_types=None, factors_dict=None, layout='NHWT'):
        """
        Parameters
        ----------
        data_dict:  Dict[str, Union[np.array, torch.Tensor]]
        factors_dict:   Optional[Dict[str, Sequence[int]]]
            each element `factors` is a Sequence of int, representing (t_factor, h_factor, w_factor)

        Returns
        -------
        downsampled_data_dict:  Dict[str, torch.Tensor]
            Modify on a deep copy of data_dict instead of directly modifying the original data_dict
        """
        if factors_dict is None:
            factors_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        downsampled_data_dict = SEVIRDataLoader.data_dict_to_tensor(
            data_dict=data_dict,
            data_types=data_types)    # make a copy
        for key, data in data_dict.items():
            factors = factors_dict.get(key, None)
            if factors is not None:
                downsampled_data_dict[key] = change_layout_torch(
                    data=downsampled_data_dict[key],
                    in_layout=layout,
                    out_layout='NTHW')
                # downsample t dimension
                t_slice = [slice(None, None), ] * 4
                t_slice[1] = slice(None, None, factors[0])
                downsampled_data_dict[key] = downsampled_data_dict[key][tuple(t_slice)]
                # downsample spatial dimensions
                downsampled_data_dict[key] = avg_pool2d(
                    input=downsampled_data_dict[key],
                    kernel_size=(factors[1], factors[2]))

                downsampled_data_dict[key] = change_layout_torch(
                    data=downsampled_data_dict[key],
                    in_layout='NTHW',
                    out_layout=layout)

        return downsampled_data_dict

    def _random_sample(self):
        """
        Returns
        -------
        ret_dict
            dict. ret_dict.keys() == self.data_types.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        num_sampled = 0
        event_idx_list = nprand.randint(low=self.start_event_idx,
                                        high=self.end_event_idx,
                                        size=self.batch_size)
        seq_idx_list = nprand.randint(low=0,
                                      high=self.num_seq_per_event,
                                      size=self.batch_size)
        seq_slice_list = [slice(seq_idx * self.stride,
                                seq_idx * self.stride + self.seq_len)
                          for seq_idx in seq_idx_list]
        ret_dict = {}
        while num_sampled < self.batch_size:
            event = self._load_event_batch(event_idx=event_idx_list[num_sampled],
                                           event_batch_size=1)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event[imgt_idx][[0, ], :, :, seq_slice_list[num_sampled]]  # keep the dim of batch_size for concatenation
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq),
                                                    axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})
        return ret_dict

    def _sequent_sample(self):
        """
        Returns
        -------
        ret_dict:   Dict
            `ret_dict.keys()` contains `self.data_types`.
            `ret_dict["mask"]` is a list of bool, indicating if the data entry is real or padded.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        assert not self.use_up, 'Data loader used up! Reset it to reuse.'
        event_idx = self.curr_event_idx
        seq_idx = self.curr_seq_idx
        num_sampled = 0
        sampled_idx_list = []   # list of (event_idx, seq_idx) records
        while num_sampled < self.batch_size:
            sampled_idx_list.append({'event_idx': event_idx,
                                     'seq_idx': seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]['event_idx']
        event_batch_size = sampled_idx_list[-1]['event_idx'] - start_event_idx + 1

        event_batch = self._load_event_batch(event_idx=start_event_idx,
                                             event_batch_size=event_batch_size)
        ret_dict = {"mask": []}
        all_no_pad_flag = True
        for sampled_idx in sampled_idx_list:
            batch_slice = [sampled_idx['event_idx'] - start_event_idx, ]  # use [] to keepdim
            seq_slice = slice(sampled_idx['seq_idx'] * self.stride,
                              sampled_idx['seq_idx'] * self.stride + self.seq_len)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq),
                                                    axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})
            # add mask
            no_pad_flag = sampled_idx['event_idx'] < self.end_event_idx
            if not no_pad_flag:
                all_no_pad_flag = False
            ret_dict["mask"].append(no_pad_flag)
        if all_no_pad_flag:
            # if there is no padded data items at all, set `ret_dict["mask"] = None` for convenience.
            ret_dict["mask"] = None
        # update current idx
        self.set_curr_event_idx(event_idx)
        self.set_curr_seq_idx(seq_idx)
        return ret_dict

    def _idx_sample(self, index):
        """
        Parameters
        ----------
        index
            The index of the batch to sample.
        Returns
        -------
        ret_dict
            dict. ret_dict.keys() == self.data_types.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        event_idx = (index * self.batch_size) // self.num_seq_per_event
        seq_idx = (index * self.batch_size) % self.num_seq_per_event
        num_sampled = 0
        sampled_idx_list = []  # list of (event_idx, seq_idx) records
        while num_sampled < self.batch_size:
            sampled_idx_list.append({'event_idx': event_idx,
                                     'seq_idx': seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]['event_idx']
        event_batch_size = sampled_idx_list[-1]['event_idx'] - start_event_idx + 1

        event_batch = self._load_event_batch(event_idx=start_event_idx,
                                             event_batch_size=event_batch_size)
        ret_dict = {}
        for sampled_idx in sampled_idx_list:
            batch_slice = [sampled_idx['event_idx'] - start_event_idx, ]  # use [] to keepdim
            seq_slice = slice(sampled_idx['seq_idx'] * self.stride,
                              sampled_idx['seq_idx'] * self.stride + self.seq_len)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq),
                                                    axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})

        ret_dict = self.data_dict_to_tensor(data_dict=ret_dict,
                                            data_types=self.data_types)
        if self.preprocess:
            ret_dict = self.preprocess_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 layout=self.layout,
                                                 rescale=self.rescale_method)

        if self.downsample_dict is not None:
            ret_dict = self.downsample_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 factors_dict=self.downsample_dict,
                                                 layout=self.layout)
        return ret_dict


class SEVIRDataIterator():
    '''
    A wrapper s.t. it implements the function sample().
    Every arguments in this class will be redirected to the inner SEVIRDataLoader object.
    If you expect a pythonic iterator, use SEVIRDataLoader instead.
    '''
    def __init__(self, **kwargs):
        self.loader = SEVIRDataLoader(**kwargs)
        self.sample_mode = kwargs['sample_mode'] if 'sample_mode' in kwargs else 'random'
    
    def reset(self):
        self.loader.reset()
    
    def sample(self, batch_size=None):
        '''
        The input param batch_size here is not used
        '''
        out = next(self.loader, None)
        if out is None and self.sample_mode == 'random':
            self.loader.reset()
            out = next(self.loader, None)
        return out

# =====================================================================================
# MeteoNet data
# Reshape it to 256x256, with in_len=4, out_len=12
# https://meteofrance.github.io/meteonet/
# downloaded from https://meteonet.umr-cnrm.fr/dataset/data/NW/radar/reflectivity_old_product/
# =====================================================================================

MeteoMM_IDX = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] # Only for 2016 and 2017
MM2018_IDX = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'] # For year 2018 --> Test Set
METEO_TRAIN = 370*1000//20*4
METEO_VALID = 1024
class MeteoNetDataset(torch.utils.data.Dataset):
    def __init__(self, in_len, out_len, stride, months=MeteoMM_IDX, reshape_size=256, start_year=2016, end_year=2018, is_train=True):
        super(MeteoNetDataset, self).__init__()
        data_path = 'data/meteonet/{}/reflectivity-old-NW-{}-{}/reflectivity_old_NW_{}_{}.{}.npz'
        self.file_paths = []
        for yy in range(start_year, end_year+1):
            if yy == 2018 and not set(['11', '12']).isdisjoint(set(months)):
                months = MM2018_IDX
            self.file_paths += [data_path.format(str(yy), str(yy), mm, str(yy), mm, '1') for mm in months]
            self.file_paths += [data_path.format(str(yy), str(yy), mm, str(yy), mm, '2') for mm in months]
            self.file_paths += [data_path.format(str(yy), str(yy), mm, str(yy), mm, '3') for mm in months]
        self.file_paths.sort()
        self.in_len = in_len
        self.total_len = out_len + in_len
        self.stride = stride
        self.reshape_size = reshape_size
        self.file_idx = 0
        self.data_idx = 0
        self.dataset_len = None
        self.data_open = None
        
        self.is_train = is_train
        if is_train:
            if start_year==2018:
                self.dataset_len = METEO_VALID
                # self.dataset_len = int(test_no*0.5) # For thresholding
            else:
                self.dataset_len = METEO_TRAIN
                # self.dataset_len = int(train_no*0.5) # For thresholding

    def __len__(self):
        if self.dataset_len is not None:
            return self.dataset_len
        len = 0
        for f in self.file_paths:
            file_len = np.load(f)['data'].shape[0]
            data_len = (file_len-self.total_len)//self.stride
            len += data_len
        self.dataset_len = len
        return len

    def reset(self):
        self.data_open = None

    def preprocess(self, seq):
        # seq[seq==-1] = 0
        # seq = seq/100.
        # seq = (seq/200)**(1/1.6) # T, H, W
        seq[seq==255] = 0
        seq = seq/70 # T, H, W
        seq = seq.unsqueeze(1) # T, C, H, W
        seq = F.interpolate(seq, size=self.reshape_size, mode='bilinear', align_corners=False)
        seq = seq.clamp(0,1)
        return seq
    
    def __getitem__(self, i):
        assert i < self.dataset_len
        if self.data_open is None:
            self.file_idx = 0
            self.data_open = torch.tensor(np.load(self.file_paths[self.file_idx])['data']).float() # T, H, W
            self.data_idx = 0

        with torch.no_grad():
            try:
                if self.data_idx + self.total_len >= len(self.data_open):
                    raise Exception()
                X = self.data_open[self.data_idx: self.data_idx + self.total_len]
            except:
                self.file_idx += 1
                if self.file_idx >= len(self.file_paths):
                    self.file_idx = 0
                    
                self.data_open = torch.tensor(np.load(self.file_paths[self.file_idx])['data']).float() # T, H, W
                self.data_idx = 0
                X = self.data_open[self.data_idx: self.data_idx + self.total_len]
            
            self.data_idx += self.stride
            X = self.preprocess(X)
            x, y = X[:self.in_len], X[self.in_len:]
            
        return x, y

def load_MeteoNet_data(batch_size, val_batch_size, in_len, out_len, stride, split_year=2018, reshape_size=256, train=False, num_workers=0):
    if train:
        train_set = MeteoNetDataset(in_len, out_len, stride, reshape_size=reshape_size, end_year=split_year)
        valid_set = MeteoNetDataset(in_len, out_len, stride, months=MM2018_IDX, reshape_size=reshape_size, start_year=split_year)
        dataloader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
        dataloader_valid = torch.utils.data.DataLoader(valid_set, batch_size=val_batch_size, pin_memory=True, num_workers=num_workers)
        return dataloader_train, dataloader_valid
    else:
        test_set = MeteoNetDataset(in_len, out_len, stride, months=MM2018_IDX, reshape_size=reshape_size, start_year=split_year, is_train=False)
        dataloader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
        return None, dataloader_test
