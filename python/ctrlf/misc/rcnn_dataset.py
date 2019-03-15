import os
import cv2
import sys
import json
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
# may change to skimage?
from scipy.misc import imresize
from skimage.io import imread
from torchvision import transforms
from threading import Thread, Lock
from queue import Queue

from PIL import Image, ImageDraw

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

    def __init__(self, opt, is_train, logger, data_dir, json_path, db_file):
        super(Dataset, self).__init__()
        self.opt = opt
        # TODO: change this
        self.opt.dtp_train = True
        self.is_train = is_train
        self.logger = logger
        # all image data, memory for speed
        self.target_image_size = float(opt.image_size)
        # note this also set self.images
        self.data = self.load_data(data_dir, json_path, db_file)
        db = h5py.File(db_file, 'r')
        images = db.get('images').value
        self.images = []
        self.image_mean = db.get('image_mean').value
        # maybe get h, w from data?
        for i, ims in enumerate(images):
            h, w = self.data[i]['h'], self.data[i]['w']
            img = ims[0, :h, :w]
            scale = self.target_image_size / max(h, w)
            img = imresize(img, scale)
            img = img[:, :, np.newaxis]

            self.images.append(img)
        self.image_number = len(self.images)
        self.print_info(True, "total images:{}".format(self.image_number))
        self.print_info(True, "image_mean:{}".format(self.image_mean))
        # this change region_proposal from list of list to ndarray
        self.filter_region_proposals()
        # after these box change to ndarray, and xc, yc, w, h
        self.encode_boxes('region_proposals')
        self.encode_boxes('gt_boxes')
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((self.image_mean / 255, ), (1, ))])

    def load_data(self, data_dir, json_path, db_file):
        if not os.path.isfile(json_path):
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
                    # better performance?
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
            self.print_info(False, "extract DTP")
            c_range = list(range(1, 40, 3))  # horizontal range
            r_range = list(range(1, 40, 3))  # vertical range
            w_range = (33, 284)
            h_range = (44, 310)
            self.extract_dtp(data, c_range, r_range, w_range, h_range)
            self.print_info(False, "extract DTP done")
            for datum in data:
                try:
                    datum['region_proposals'] = np.load('npz/' + datum['id'].split('/')[-1][: -4] + '_dtp.npz')[
                        'regions'].tolist()
                except:
                    print('npz/' + datum['id'].split('/')[-1][: -4] + '_dtp.npz')
            # self.filter_ground_truth_boxes(images, data)
            if not os.path.exists(db_file):
                create_db(images, db_file, max_shape)

            with open(json_path, "w") as f:
                json.dump(data, f)
        else:
            with open(json_path, "r") as f:
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
            #self.print_info(True, "in utils.filter_region_proposals, before unique_boxes {}".
            #format(len(datum['region_proposals'])))
            region_proposals = np.unique(okay, axis=0)
            #self.print_info(True, "in utils.filter_region_proposals, after unique_boxes {}".format(len(okay)))
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

    def extract_regions(self, t_img, c_range, r_range, w_range, h_range):
        all_boxes = []
        min_w, max_w = w_range
        min_h, max_h = h_range
        for r in r_range:
            for c in c_range:
                s_img = cv2.morphologyEx(t_img, cv2.MORPH_CLOSE, np.ones((r, c), dtype=np.ubyte))
                n, l_img, stats, centroids = cv2.connectedComponentsWithStats(s_img, connectivity=4)
                boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in stats
                         if min_w <= b[2] <= max_w and min_h <= b[3] <= max_h]
                all_boxes += boxes
        return all_boxes

    def find_regions(self, img, threshold_range, c_range, r_range, w_range, h_range):
        """
        Extracts DTP from an image using different thresholds and morphology kernels
        """

        ims = []
        for t in threshold_range:
            ims.append((img < t).astype(np.ubyte))

        ab = []
        for t_img in ims:
            ab += self.extract_regions(t_img, c_range, r_range, w_range, h_range)

        return ab

    def extract_dtp(self, data, c_range, r_range, w_range, h_range, multiple_thresholds=True):
        q = Queue()
        for i, datum in enumerate(data):
            q.put((i, datum))

        def worker():
            while True:
                i, datum = q.get()
                file_name = datum['id']
                # change to a parameter?
                proposal_file = 'npz/' + file_name.split('/')[-1][:-4] + '_dtp.npz'
                if not os.path.exists(proposal_file):
                    img = imread(file_name)
                    # extract regions
                    m = img.mean()
                    if multiple_thresholds:
                        threshold_range = np.arange(0.7, 1.01, 0.1) * m
                    else:
                        threshold_range = np.array([0.9]) * m
                    region_proposals = self.find_regions(img, threshold_range, c_range, r_range, w_range, h_range)
                    region_proposals = self.unique_boxes(region_proposals)
                    np.savez_compressed(proposal_file, regions=region_proposals)
                else:
                    self.print_info(False, "dtp.npz for {} already exists, skip it".format(file_name))
                q.task_done()

        num_workers = 8
        for i in range(num_workers):
            t = Thread(target=worker)
            t.daemon = True
            t.start()
        q.join()

    def print_info(self, is_debug, message):
        if is_debug:
            self.logger.debug(message)
        else:
            self.logger.info(message)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # TODO: change dtp_train
        ind = index % self.image_number
        img = self.images[ind]
        img = self.transforms(img)
        gt_boxes = self.data[ind]['gt_boxes']
        out = (img, gt_boxes)
        if self.opt.dtp_train:
            proposals = self.data[ind]['region_proposals']
            out += (proposals,)

        return out


class RandomSampler(object):
    """Samples num_iters times the idxes from class
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, num_iters, is_train):
        self.num_iters = num_iters
        self.is_train = is_train

    def __iter__(self):
        # TODO: chage this to right split
        if self.is_train:
            return iter(torch.arange(self.num_iters, dtype=torch.int64))
        else:
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
