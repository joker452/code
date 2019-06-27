import os
import cv2
import sys
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
# may change to skimage?
from scipy.misc import imresize
from skimage.io import imread
from torchvision import transforms
from threading import Thread, Lock
from queue import Queue


def mkdir(dir_name):
    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError:
            print('Can not make directory for {}'.format(dir_name))
            raise OSError
        else:
            print("Make directory for {}".format(dir_name))
    else:
        print("{} already exists".format(dir_name))


def create_db(images, db_file, max_shape):
    print("begin create h5 data file")
    lock = Lock()
    q = Queue()
    f = h5py.File(db_file, 'w')
    image_dset = f.create_dataset('images', (len(images), 1, max_shape[0], max_shape[1]), dtype=np.uint8)
    f.create_dataset('image_mean', data=calculate_mean(images))
    for i, img in enumerate(images):
        q.put((i, img))

    def worker():
        while True:
            i, img = q.get()
            lock.acquire()
            h, w = img.shape
            image_dset[i, :, :h, :w] = img
            lock.release()
            q.task_done()

    print('adding images to hdf5.... (this might take a while)')
    num_workers = 8
    for i in range(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()
    q.join()
    f.close()


class RcnnDataset(Dataset):

    def __init__(self, opt, use_dtp, is_train, logger, data_dir, json_name, db_file):
        super(Dataset, self).__init__()
        self.use_dtp = use_dtp
        self.logger = logger
        self.is_train = is_train
        self.parent_dir = os.path.abspath(os.path.join(data_dir, os.pardir))
        # all image data, memory for speed
        self.target_image_size = float(opt.image_size)
        # note this also set self.images
        self.data = self.load_data(data_dir, json_name, db_file)
        db = h5py.File(os.path.join(self.parent_dir, 'cache', db_file), 'r')
        images = db.get('images').value
        self.images = []
        self.image_mean = db.get('image_mean').value
        for i, ims in enumerate(images):
            h, w = self.data[i]['h'], self.data[i]['w']
            img = ims[0, :h, :w]
            scale = self.target_image_size / max(h, w)
            img = imresize(img, scale)
            img = img[:, :, np.newaxis]

            self.images.append(img)
        self.image_number = len(self.images)
        self.print_info(False, "total images:{}".format(self.image_number))
        self.print_info(False, "image_mean:{}".format(self.image_mean))
        # this change region_proposal from list of list to ndarray
        self.filter_region_proposals()
        # after these, boxes change to ndarray, and xc, yc, w, h
        self.encode_boxes('region_proposals')
        self.encode_boxes('gt_boxes')
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((self.image_mean / 255,), (1,))])

    def load_data(self, data_dir, json_name, db_file):

        if not os.path.isfile(os.path.join(self.parent_dir, 'cache', json_name)):
            mkdir(os.path.join(self.parent_dir, 'cache'))
            self.print_info(False, "json file doesn't exist, load data from scratch")
            image_paths = [data_dir + entry.name for entry in os.scandir(data_dir)
                           if entry.name.endswith('.jpg')]
            image_paths.sort(key=lambda item: (len(item), item))
            data = []
            images = []
            images_shape = []
            for image_path in image_paths:
                label_path = image_path[: -3] + 'txt'
                if not os.path.exists(label_path):
                    self.print_info(False, "{} doesn't exist".format(label_path))
                    sys.exit(1)
                with open(label_path, 'r') as f:
                    box_lines = f.readlines()
                img = imread(image_path, cv2.IMREAD_GRAYSCALE)
                images.append(img)
                # h, w
                images_shape.append((img.shape[0], img.shape[1]))
                gt_boxes = []
                # x1, y1, x2, y2
                for box_line in box_lines:
                    box = box_line.split()
                    box = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    gt_boxes.append(box)
                datum = {}
                datum['id'] = image_path
                datum['h'], datum['w'] = img.shape[0], img.shape[1]
                datum['gt_boxes'] = gt_boxes
                data.append(datum)

            max_shape = np.array(images_shape).max(0)
            for datum in data:
                try:
                    if self.is_train:
                        datum['region_proposals'] = \
                            np.load('npz/train/' + datum['id'].split('/')[-1].split('.')[0] + '_dtp.npz')[
                                'regions'].tolist()
                    else:
                        datum['region_proposals'] = \
                            np.load('npz/test/' + datum['id'].split('/')[-1].split('.')[0] + '_dtp.npz')[
                                'regions'].tolist()
                except:
                    if self.is_train:
                        print('npz/train/' + datum['id'].split('/')[-1].split('.')[0] + "_dtp.npz doesn't exist!")
                    else:
                        print('npz/test/' + datum['id'].split('/')[-1].split('.')[0] + "_dtp.npz doesn't exist!")

            # self.filter_ground_truth_boxes(images, data)
            if not os.path.exists(os.path.join(self.parent_dir, 'cache', db_file)):
                create_db(images, os.path.join(self.parent_dir, 'cache', db_file), max_shape)

            with open(os.path.join(self.parent_dir, 'cache', json_name), "w") as f:
                json.dump(data, f)
        else:
            with open(os.path.join(self.parent_dir, 'cache', json_name), "r") as f:
                data = json.load(f)

        return data

    def filter_region_proposals(self):
        """
        Remove duplicate region proposals when downsampled to the roi-pooling size
        First it's the image scaling preprocessing then it's the downsampling in
        the network.
        """
        for i, datum in enumerate(self.data):
            H, W = datum['h'], datum['w']
            scale = self.target_image_size / max(H, W)

            scale /= 8
            okay = []
            for box in datum['region_proposals']:
                w, h = box[2] - box[0], box[3] - box[1]
                w, h = round(scale * w), round(scale * h)
                if w > 0 and h > 0:
                    okay.append(box)

            # Only keep unique proposals in downsampled coordinate system, i.e., remove aliases
            # self.print_info(True, "in utils.filter_region_proposals, before unique_boxes {}".
            # format(len(datum['region_proposals'])))
            region_proposals = np.unique(okay, axis=0)
            # self.print_info(True, "in utils.filter_region_proposals, after unique_boxes {}".format(len(okay)))
            datum['region_proposals'] = region_proposals

    def filter_ground_truth_boxes(self, images, data):
        """
        Remove too small ground truth boxes when downsampled to the roi-pooling size
        First it's the image scaling preprocessing then it's the downsampling in
        the network.
        """

        for i, img in enumerate(images):
            H, W = img.shape[0], img.shape[1]
            scale = self.target_image_size / max(H, W)
            # Since we downsample the image 8 times before the roi-pooling,
            # divide scaling by 8. Hardcoded per network architecture.
            scale /= 8
            okay_gt = []
            datum = data[i]

            for gt in datum['gt_boxes']:
                self.print_info(False, "in filter_ground_truth, before {}".format(len(datum['gt_boxes'])))
                x, y, w, h = gt[0], gt[1], gt[2] - gt[0], gt[3] - gt[1]

                x, y = round(scale * (x - 1) + 1), round(scale * (y - 1) + 1)
                w, h = round(scale * w), round(scale * h)
                if w > 1 and h > 1 and x > 0 and y > 0:
                    okay_gt.append(gt)
                self.print_info(False, "in filter_ground_truth, after {}".format(len(okay_gt)))

            datum['gt_boxes'] = okay_gt

    def pad_proposals(self, proposals, im_shape, pad=10):
        '''
        for dtp, enlarge the proposal and limit it within the image
        '''
        props = []
        for p in proposals:
            pp = [max(0, p[0] - pad), max(0, p[1] - pad), min(im_shape[1], p[2] + pad), min(im_shape[0], p[3] + pad)]
            props.append(pp)
        return np.array(props)

    def encode_boxes(self, box_type):
        """
        rescales boxes and ensures they are inside the image
        """
        for i, datum in enumerate(self.data):
            H, W = datum['h'], datum['w']
            scale = self.target_image_size / max(H, W)
            # Needed for not so tightly labeled datasets, like washington
            # move boxes to close to edges
            # if box_type == 'region_proposals':
            #     datum[box_type] = self.pad_proposals(datum[box_type], (H, W), 10)
            if box_type == 'gt_boxes':
                datum[box_type] = np.array(datum[box_type])
            scaled_boxes = np.round(scale * (datum[box_type] + 1) - 1)
            x1 = scaled_boxes[:, 0]
            y1 = scaled_boxes[:, 1]
            x2 = scaled_boxes[:, 2]
            y2 = scaled_boxes[:, 3]

            xc = (x1 + x2) // 2
            yc = (y1 + y2) // 2
            w = x2 - x1
            h = y2 - y1
            datum[box_type] = np.stack([xc, yc, w, h], 1)

    def print_info(self, is_debug, message):
        if is_debug:
            self.logger.debug(message)
        else:
            self.logger.info(message)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ind = index % self.image_number
        img = self.images[ind]
        img = self.transforms(img)
        gt_boxes = self.data[ind]['gt_boxes']
        out = (img, gt_boxes)
        H, W = self.data[ind]['h'], self.data[ind]['w']
        if self.use_dtp:
            proposals = self.data[ind]['region_proposals']
            out += (proposals, (H, W))
        return out


class RandomSampler(Sampler):
    """Samples num_iters times the idxes from class
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, dataset, num_iters):
        super(RandomSampler, self).__init__(dataset)
        self.num_iters = num_iters

    def __iter__(self):
        return iter(torch.randperm(self.num_iters))

    def __len__(self):
        return self.num_iters


def calculate_mean(images):
    mean = 0.0
    counter = 0
    for i, image in enumerate(images):
        mean = mean + image.mean()
        counter += 1

    return mean / counter
