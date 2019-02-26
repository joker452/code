#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:26:36 2017

@author: tomas
"""
import os
import copy
import numpy as np
import torch.utils.data as t_data
from scipy.misc import imresize
from skimage.io import imread
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import skimage.filters as fi
import torch
import h5py
from queue import Queue
from threading import Thread, Lock
from . import dataset_loader as dl
from . import box_utils

from . import utils


class Dataset(t_data.Dataset):
    def __init__(self, opt, split, logger, alphabet=None, root='data/', ):
        self.logger = logger
        super(Dataset, self).__init__()
        # fix: delete these
        if alphabet == None:
            self.alphabet = dl.default_alphabet
        else:
            self.alphabet = alphabet

        self.train = split == 'train'
        self.opt = opt
        if self.train:
            self.dataset = opt.dataset
        else:
            self.dataset = opt.val_dataset

        self.iam = self.dataset == 'iam'
        # note: extract dtp
        self.data = getattr(dl, 'load_%s' % self.dataset)(fold=self.opt.fold, alphabet=self.alphabet)
        db_file = root + '%s/data_fold%d.h5' % (self.dataset, self.opt.fold)

        if self.opt.ghosh and self.split == 'test':
            for datum in self.data:
                datum['split'] = 'test'

        self.data_split = [d for d in self.data if d['split'] == split]
        self.N = len(self.data_split)
        self.split = split
        self.image_size = opt.image_size
        self.augment = opt.augment
        self.augment_mode = opt.augment_mode
        # all unique word label
        self.vocab = utils.build_vocab(self.data)
        self.vocab_size = len(self.vocab)
        self.itow = {i: w for i, w in enumerate(self.vocab)}
        self.wtoi = {w: i for i, w in enumerate(self.vocab)}
        self.aug_ratio = opt.aug_ratio
        self.max_words = 256

        self.dtp_train = opt.dtp_train
        if self.dtp_train:
            self.aug_ratio = 1.0  # Only inplace augmentation for now

        self.resolution = 3
        # boils down to whether or not the all embeddings should match their label
        self.bins = len(self.alphabet) * 2
        self.ngrams = 2
        self.unigram_levels = list(range(1, 6))
        self.args = (self.resolution, self.alphabet)

        # logger.debug("db_file path {}".format(db_file))
        # calculate mean, note this save data
        if not os.path.exists(db_file):
            create_db(self.data, db_file)
        db = h5py.File(db_file)
        # note this the image data
        ims = db.get('images').value
        ow = db.get('image_widths').value
        oh = db.get('image_heights').value
        self.image_mean = db.get('image_mean').value
        db.close()
        self.max_shape = (ims.shape[2], ims.shape[3])
        # note this the image data in this split
        self.images = []
        self.original_heights = []
        self.original_widths = []
        self.medians = []
        for j, datum in enumerate(self.data):
            if datum['split'] == self.split:
                h, w = oh[j], ow[j]
                img = ims[j, 0, :h, :w]
                self.images.append(img)
                self.original_heights.append(h)
                self.original_widths.append(w)
                self.medians.append(np.median(img))
        # remove boxes that are too small after resize and downsample in the network
        utils.filter_region_proposals(self.data_split, self.original_heights, self.original_widths, opt.image_size)
        # note: this resize gt_boxes and region_proposals
        self.encode_boxes('gt_boxes')
        self.encode_boxes('region_proposals')

        # use in augment
        self.tparams = {}
        self.tparams['shear'] = (-5, 30)
        self.tparams['order'] = 1  # bilinear
        self.tparams['selem_size'] = (3, 4)  # max size for square selem for erosion, dilation

        self.init_queries()

        # For full page augmentation
        # list of a list, each is all the image that is the same character
        # note this saves data!
        # use in argument
        self.words_by_label = [[] for i in range(self.vocab_size)]
        for i, datum in enumerate(self.data_split):
            img = self.images[i]
            for r in datum['regions']:
                x1, y1, x2, y2 = r['x'], r['y'], r['x'] + r['width'], r['y'] + r['height']
                word = img[y1:y2, x1:x2]
                ind = self.wtoi[r['label']]
                self.words_by_label[ind].append(word)

    def dataset_query_filter(self, features, gt_targets, tensorize=False):
        if self.iam:
            assert hasattr(self, 'stopwords'), 'no stopwords loaded'
            e, l = [], []
            for datum, embeddings, labels in zip(self.data_split, features, gt_targets):
                for r, embedding, label in zip(datum['regions'], embeddings, labels):
                    if r['status'] == 'ok':
                        if self.itow[label] not in self.stopwords:
                            e.append(embedding)
                            l.append(label)

            e, l = np.array(e), np.array(l)
        else:
            e, l = np.concatenate(features, 0), np.concatenate(gt_targets, 0)

        if tensorize:
            e, l = torch.from_numpy(e), torch.from_numpy(l)

        return e, l

    def init_queries(self):
        labels = []
        for datum in self.data_split:
            e, l = [], []
            for r in datum['regions']:
                word = r['label']
                l.append(self.wtoi[word])
            labels.append(l)

        # queries are embeddings
        # self.queries = np.concatenate(embeddings, 0)
        self.qtargets = np.concatenate(labels, 0)

        self.qtargets, inds = np.unique(self.qtargets, return_index=True)
        # self.queries = self.queries[inds, :]

    def get_queries(self, tensorize=False):
        queries = self.queries
        qtargets = self.qtargets
        if tensorize:
            queries = torch.from_numpy(queries)
            qtargets = torch.from_numpy(qtargets)

        return queries, qtargets

    def encode_boxes(self, box_type='gt_boxes'):
        """
        rescales boxes and ensures they are inside the image
        """
        for i, datum in enumerate(self.data_split):
            H, W = self.original_heights[i], self.original_widths[i]
            scale = float(self.image_size) / max(H, W)

            # Needed for not so tightly labeled datasets, like washington
            # move boxes to close to edges
            if box_type == 'region_proposals' and self.dataset.find('washington') > -1:
                datum[box_type] = utils.pad_proposals(datum[box_type], (H, W), 10)

            boxes = np.array(datum[box_type])
            # fix: no need to convert between numpy and torch?
            scaled_boxes = np.round(scale * (boxes + 1) - 1)
            x1 = scaled_boxes[:, 0]
            y1 = scaled_boxes[:, 1]
            x2 = scaled_boxes[:, 2]
            y2 = scaled_boxes[:, 3]

            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            datum[box_type] = np.stack([xc, yc, w, h], 1)

    def __len__(self):
        return self.N

    def __name__(self):
        return self.dataset

    def get_vocab(self):
        return self.vocab

    # fix: duplicate with self.vocab_size
    def get_vocab_size(self):
        return len(self.vocab)

    def normalize(self, e):
        a = np.linalg.norm(e)
        if a != 0.0:
            return e / a
        else:
            return e

    def prep(self, img):
        H, W = img.shape
        scale = float(self.image_size) / max(H, W)
        img = imresize(img, scale)
        img = np.invert(img)
        img = img.astype(np.float32)  # convert to float
        img /= 255.0  # convert to [0, 1]
        #img -= (self.image_mean / 255.0)
        return np.expand_dims(img, 0)

    def getitem(self, index):
        datum = self.data_split[index]
        img = self.images[index]
        labels = np.array([self.wtoi[r['label']] for r in datum['regions']])
        boxes = datum['gt_boxes']
        img = self.prep(img)
        oshape = (self.original_heights[index], self.original_widths[index])
        proposals = datum['region_proposals']
        img = torch.from_numpy(img).unsqueeze(0)
        boxes = torch.from_numpy(boxes).unsqueeze(0)
        proposals = torch.from_numpy(proposals).unsqueeze(0)
        out = (img, oshape, boxes, proposals, labels)
        return out


    def __getitem__(self, index):
        if self.train:
            ind = index % self.N
            # To get the same augmentation ratio on average as the H5 files, which
            # also contain originals, we need to sometimes skip augmentation
            N = float(len(self.data_split))
            skip = np.random.rand() < (N / 5000.0)
            # fix remove augment and embeddings
            if self.augment and not skip:
                (img, boxes, embeddings, labels) = self._augment(ind)

            else:
                datum = self.data_split[ind]
                img = self.images[ind]
                boxes = datum['gt_boxes']
                # change no embedding
                # embeddings = np.array([self.wtoe[r['label']] for r in datum['regions']])
            # note this resize the img and subtact the mean value of all the images
            img = self.prep(img)
            out = (img, boxes)

            if self.dtp_train:
                proposals = self.data_split[ind]['region_proposals']
                out += (proposals,)

        else:
            datum = self.data_split[index]
            img = self.images[index]
            # change remove embedding
            # embeddings = np.array([self.wtoe[r['label']] for r in datum['regions']])
            boxes = datum['gt_boxes']
            proposals = datum['region_proposals']

            img = self.prep(img)
            oshape = (self.original_heights[index], self.original_widths[index])
            # change remove embedding
            # out = (img, oshape, boxes, proposals, embeddings, labels)
            out = (img, oshape, boxes, proposals)

        return out

    def _augment(self, index):
        if np.random.rand() < self.aug_ratio:
            out = self.inplace_augment(index)
        else:
            out = self.full_page_augment(index)

        return out

    def full_page_augment(self, index, augment=True):
        tparams = copy.deepcopy(self.tparams)
        tparams['keep_size'] = False
        tparams['rotate'] = (-5, 5)
        tparams['hpad'] = (0, 12)
        tparams['vpad'] = (0, 12)

        m = int(np.median(self.medians))
        s = 3  # inter word space
        x, y = s, s  # Upper left corner of box
        gt_boxes = []
        gt_labels = []
        gt_embeddings = []
        si = np.random.randint(len(self.data_split))
        shape = (self.original_heights[si], self.original_widths[si])
        canvas = self.create_background(m + np.random.randint(0, 20) - 10, shape)
        maxy = 0
        for j in range(self.max_words):
            ind = np.random.randint(self.vocab_size)
            k = len(self.words_by_label[ind])
            while k == 0:
                ind = np.random.randint(self.vocab_size)
                k = len(self.words_by_label[ind])

            word = self.words_by_label[ind][np.random.randint(k)]

            # randomly transform word and place on canvas
            if augment:
                try:
                    tword = utils.augment(word, tparams, self.augment_mode)
                except:
                    tword = word
            else:
                tword = word

            h, w = tword.shape
            if x + w > shape[1]:  # done with row?
                x = s
                y = maxy + s

            if y + h > shape[0]:  # done with page?
                break

            x1, y1, x2, y2 = x, y, x + w, y + h
            canvas[y1:y2, x1:x2] = tword
            b = [x1, y1, x2, y2]
            gt_labels.append(ind)
            gt_boxes.append(b)
            gt_embeddings.append(self.wtoe[self.vocab[ind]])
            x = x2 + s
            maxy = max(maxy, y2)

        H, W = shape
        scale = float(self.image_size) / max(H, W)
        boxes = np.array(gt_boxes)
        scaled_boxes = torch.from_numpy(np.round(scale * (boxes + 1) - 1))
        gt_boxes = box_utils.x1y1x2y2_to_xcycwh(scaled_boxes).numpy()
        return canvas, gt_boxes, np.array(gt_embeddings), np.array(gt_labels)

    def create_background(self, m, shape, fstd=2, bstd=10):
        canvas = np.ones(shape) * m
        noise = np.random.randn(shape[0], shape[1]) * bstd
        noise = fi.gaussian(noise, fstd)  # low-pass filter noise
        canvas += noise
        canvas = np.round(canvas)
        canvas = np.minimum(canvas, 255)
        canvas = canvas.astype(np.uint8)
        return canvas

    def inplace_augment(self, index):
        tparams = copy.deepcopy(self.tparams)
        tparams['keep_size'] = True
        tparams['rotate'] = (0, 1)
        tparams['hpad'] = (0, 1)
        tparams['vpad'] = (0, 1)
        datum = self.data_split[index]
        img = self.images[index]
        out = img.copy()
        boxes = datum['gt_boxes']
        labels = np.array([self.wtoi[r['label']] for r in datum['regions']])
        boxes = datum['gt_boxes']
        embeddings = np.array([self.wtoe[r['label']] for r in datum['regions']])

        for i, r in enumerate(reversed(datum['regions'])):
            b = [r['x'], r['y'], r['x'] + r['width'], r['y'] + r['height']]
            try:  # Some random values for weird boxes give value errors, just handle and ignore
                b = utils.close_crop_box(img, b)
                word = img[b[1]:b[3], b[0]:b[2]]
                aug = utils.augment(word, tparams, self.augment_mode)
                out[b[1]:b[3], b[0]:b[2]] = aug
            except ValueError:
                continue

        return out, boxes, embeddings, labels


class RandomSampler(object):
    """Samples num_iters times the idxes from class
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, num_iters):
        # change: remove useless
        # self.num_samples = len(data_source)
        self.num_iters = num_iters

    def __iter__(self):
        return iter(torch.randperm(self.num_iters))

    def __len__(self):
        return self.num_iters


def create_db(data, db_file):
    sizes = []
    means = []
    for datum in data:
        img = imread(datum['id'])
        if img.ndim == 3:
            img = img_as_ubyte(rgb2gray(img))

        if datum['split'] == 'train':
            means.append(np.invert(img).mean())
        sizes.append(img.shape)
    sizes = np.array(sizes)
    max_shape = sizes.max(0)

    lock = Lock()
    q = Queue()
    f = h5py.File(db_file, 'w')
    image_dset = f.create_dataset('images', (len(data), 1, max_shape[0], max_shape[1]), dtype=np.uint8)
    image_heights = np.zeros(len(data), dtype=np.int32)
    image_widths = np.zeros(len(data), dtype=np.int32)

    for i, datum in enumerate(data):
        q.put((i, datum['id']))

    def worker():
        while True:
            i, filename = q.get()
            img = imread(filename)
            if img.ndim == 3:
                img = img_as_ubyte(rgb2gray(img))
            h, w = img.shape
            lock.acquire()
            if i % 1000 == 0:
                print('Writing image %d / %d' % (i, len(data)))

            image_heights[i] = h
            image_widths[i] = w
            image_dset[i, :, :h, :w] = img
            lock.release()
            q.task_done()

    print('adding images to hdf5.... (this might take a while)')
    num_workers = 6
    for i in range(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()
    q.join()

    f.create_dataset('image_heights', data=image_heights)
    f.create_dataset('image_widths', data=image_widths)
    f.create_dataset('image_mean', data=np.mean(means))
    f.close()
