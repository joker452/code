import os
import cv2
from threading import Thread, Lock
from queue import Queue
import numpy as np
from PIL import Image, ImageDraw


def extract_regions(t_img, c_range, r_range, w_range, h_range):
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


def find_regions(img, threshold_range, c_range, r_range, w_range, h_range):
    """
    Extracts DTP from an image using different thresholds and morphology kernels
    """

    ims = []
    for t in threshold_range:
        ims.append((img < t).astype(np.ubyte))

    ab = []
    for t_img in ims:
        ab += extract_regions(t_img, c_range, r_range, w_range, h_range)

    return ab


def unique_boxes(boxes):
    tmp = np.array(boxes)
    ncols = tmp.shape[1]
    dtype = tmp.dtype.descr * ncols
    struct = tmp.view(dtype)
    uniq, index = np.unique(struct, return_index=True)
    tmp = uniq.view(tmp.dtype).reshape(-1, ncols)
    return tmp, index


def extract_dtp(out_dir, file_names, c_range, r_range, w_range, h_range, multiple_thresholds=True):
    # extract regions
    lock = Lock()
    q = Queue()
    for i, file_name in enumerate(file_names):
        q.put((i, file_name))

    def worker():
        while True:
            i, file_name = q.get()
            lock.acquire()
            print('Processing image %d / %d' % (i, len(file_names)))
            lock.release()
            try:
                img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                image = Image.fromarray(img)
                d = ImageDraw.Draw(image)

                m = img.mean()
                if multiple_thresholds:
                    threshold_range = np.arange(0.7, 1.2, 0.1) * m
                else:
                    threshold_range = np.array([0.9]) * m
                region_proposals = find_regions(img, threshold_range, c_range, r_range, w_range, h_range)
                region_proposals, _ = unique_boxes(region_proposals)
                total = 0
                regions = []
                for region in region_proposals:
                    x1, y1, x2, y2 = region

                    d.rectangle([x1, y1, x2, y2], outline="white")
                    total += 1
                    regions.append((x1, y1, x2, y2))
                save_name = os.path.normpath(file_name).split(os.sep)
                save_name = save_name[-1].split('.')[0]
                name = out_dir + save_name + '_dtp.npz'
                np.savez_compressed(name, regions=regions, total=total)
                image.save(os.path.join(out_dir, 'image', save_name + '.jpg'))
            except:
                lock.acquire()
                print('exception thrown with file', file_name)
                lock.release()
            q.task_done()

    num_workers = 8
    for i in range(num_workers):
        t = Thread(target=worker)
        t.daemon = True
        t.start()
    q.join()


if __name__ == '__main__':
    image_dir = '/data2/dengbowen/color_out'
    image_names = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    out_dir = "/data2/dengbowen/difangzhi_dtp/"
    c_range = list(range(1, 80, 5))
    r_range = list(range(1, 75, 5))
    w_range = (33, 284)
    h_range = (44, 310)
    extract_dtp(out_dir, image_names, c_range, r_range, w_range, h_range, multiple_thresholds=True)
