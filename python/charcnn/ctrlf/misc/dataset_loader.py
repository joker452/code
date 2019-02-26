#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:10:04 2016

@author: tomas
"""
import os
import glob
import string
import json
from queue import Queue
from threading import Thread, Lock

import scipy as sp
import scipy.io
import numpy as np
from skimage.io import imread, imsave
import cv2

from . import utils

default_alphabet = string.ascii_lowercase + string.digits

def extract_regions(t_img, C_range, R_range):
    """
    Extracts region propsals for a given image
    """
    all_boxes = []    
    for R in R_range:
        for C in C_range:
            s_img = cv2.morphologyEx(t_img, cv2.MORPH_CLOSE, np.ones((R, C), dtype=np.ubyte))
            n, l_img, stats, centroids = cv2.connectedComponentsWithStats(s_img, connectivity=4)
            boxes = [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in stats]
            all_boxes += boxes
                
    return all_boxes

    
def find_regions(img, threshold_range, C_range, R_range):
    """
    Extracts DTP from an image using different thresholds and morphology kernels    
    """

    ims = []
    for t in threshold_range:
        ims.append((img < t).astype(np.ubyte))
    
    ab = []
    for t_img in ims:
        ab += extract_regions(t_img, C_range, R_range)
        
    return ab

def extract_dtp(data, C_range, R_range, multiple_thresholds=True):
    lock = Lock()
    q = Queue()
    for i, datum in enumerate(data):
        q.put((i, datum['id']))
        
    def worker():
        while True:
            i, filename = q.get()
            proposal_file = filename[:-4] + '_dtp.npz'

            if i % 1000 == 0:
                lock.acquire()
                print('Processing image %d / %d' % (i, len(data)))
                lock.release()
            
            if not os.path.exists(proposal_file):
                try:
                    img = imread(filename)

                    #extract regions
                    m = img.mean()
                    if multiple_thresholds:
                        threshold_range = np.arange(0.7, 1.01, 0.1) * m
                    else:
                        threshold_range = np.array([0.9]) * m
                    region_proposals = find_regions(img, threshold_range, C_range, R_range) 
                    region_proposals = utils.unique_boxes(region_proposals)
                    np.savez_compressed(proposal_file, region_proposals=region_proposals)
                except:
                    lock.acquire()
                    print('exception thrown with file', filename)
                    lock.release()

            q.task_done()
              
    num_workers = 6
    for i in range(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()
    q.join()
    
def load_washington_small(fold=1, root="data/washington/", alphabet=default_alphabet):
    data = load_washington(fold, root, alphabet)
    first = True #Keep the same validation page
    for datum in data:
        if datum['split'] == 'train':
            datum['split'] = 'test'
        elif datum['split'] == 'test':
            if first:
                first = False
                continue
            else:
                datum['split'] = 'train'
            
    return data

def load_washington(fold=1, root="data/washington/", alphabet=default_alphabet):
    output_json = root + 'washington_fold_%d.json' % fold
    if not os.path.exists(output_json):
        print("loading washington from files")
        files = sorted(glob.glob(os.path.join(root, 'gw_20p_wannot/*.tif')))
        gt_files = [f[:-4] + '_boxes.txt' for f in files]
        
        with open(os.path.join(root, 'gw_20p_wannot/annotations.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()

        texts = [l[:-1] for l in lines]        
        ntexts = [utils.replace_tokens(text.lower(), [t for t in text if t not in alphabet]) for text in texts]
        
        data = []
        ind = 0
        box_id = 0
        for i, (f, gtf) in enumerate(zip(files, gt_files)):
            # ignore lines
            with open(gtf, 'r') as ff:
                boxlines = ff.readlines()[1: ]

            img = imread(f)
            h, w = img.shape
            gt_boxes = []
            for line in boxlines:
                tmp = line.split()
                x1 = int(float(tmp[0]) * w)
                x2 = int(float(tmp[1]) * w)
                y1 = int(float(tmp[2]) * h)
                y2 = int(float(tmp[3]) * h)
                box = (x1, y1, x2, y2)
                gt_boxes.append(box)
                
            labels = ntexts[ind:ind + len(gt_boxes)]
            ind += len(gt_boxes)
           # labels = [str(l, errors='replace') for l in labels]
                      
            regions = []
            # picturewise
            for p, l in zip(gt_boxes, labels):
                r = {}
                r['id'] = box_id
                # str, picture_name
                r['image'] = f
                r['height'] = p[3] - p[1]
                r['width'] = p[2] - p[0]
                r['label'] = l
                r['x'] = p[0]
                r['y'] = p[1]
                regions.append(r)
                box_id += 1

            datum = {}
            datum['id'] = f
            datum['gt_boxes'] = gt_boxes
            datum['regions'] = regions
            data.append(datum)

        print("extracting DTP for washington")
        C_range=list(range(1, 40, 3)) #horizontal range
        R_range=list(range(1, 40, 3)) #vertical range
        extract_dtp(data, C_range, R_range, multiple_thresholds=True)
        for datum in data:        
            proposals = np.load(datum['id'][:-4] + '_dtp.npz')['region_proposals']
            datum['region_proposals'] = proposals.tolist()
            
        print("extracting DTP done")
        # indeces: ndarray cotaining a dict {'inds': array of index}
        inds = np.squeeze(np.load(root + 'indeces.npy').item()['inds'])
        data = [data[i] for i in inds]
        
        #Train/val/test on different partitions based on which fold we're using
        data = np.roll(data, 5 * (fold - 1)).tolist()

        for j, datum in enumerate(data):
            if j < 14:
                datum['split'] = 'train'
            elif j == 14:
                datum['split'] = 'val'
            else:
                datum['split'] = 'test'
         
        with open(output_json, 'w') as f:
            json.dump(data, f)
    
    else: #otherwise load the json
        with open(output_json) as f:
            data = json.load(f)
    # list, containing 20 dicts
    # each dict:
    # 'id': picture name, 'gt_boxes' list, x1, y1, x2, y2,
    # 'regions': list of dicts, 'id': box_id, 'image':picture_name, height', 'weight', 'label': character annotation 'x': left, 'y': top
    # 'region_proposals' list of list
    # 'split': train, test, val
    return data


