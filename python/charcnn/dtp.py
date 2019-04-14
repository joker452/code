import os
import cv2
from threading import Thread, Lock
from queue import Queue
import numpy as np
from PIL import Image, ImageDraw


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


def unique_boxes(boxes):
    tmp = np.array(boxes)
    ncols = tmp.shape[1]
    dtype = tmp.dtype.descr * ncols
    struct = tmp.view(dtype)
    uniq, index = np.unique(struct, return_index=True)
    tmp = uniq.view(tmp.dtype).reshape(-1, ncols)
    return tmp, index

def extract_dtp(out_dir, file_names, C_range, R_range, multiple_thresholds=True):
    # extract regions
    lock = Lock()
    q = Queue()
    for i, file_name in enumerate(file_names):
        q.put((i, file_name))

    def worker():
        while True:
            i, file_name = q.get()
            if i % 1000 == 0:
                lock.acquire()
                print('Processing image %d / %d' % (i, len(file_names)))
                lock.release()
            try:
                image = Image.open(file_name).convert('L')
                d = ImageDraw.Draw(image)
                img = np.asarray(image)

                m = img.mean()
                if multiple_thresholds:
                    threshold_range = np.arange(0.7, 1.01, 0.1) * m
                else:
                    threshold_range = np.array([0.9]) * m
                region_proposals = find_regions(img, threshold_range, C_range, R_range)
                region_proposals, _ = unique_boxes(region_proposals)
                total = 0
                regions = []
                for region in region_proposals:
                    x1, y1, x2, y2 = region
                    delta_x = x2 - x1
                    delta_y = y2 - y1
                    if 44 <= delta_y <= 310 and 33 <= delta_x <= 284:
                        d.rectangle([x1, y1, x2, y2], outline="black")
                        total += 1
                        regions.append((x1, y1, x2, y2))
                save_name = os.path.normpath(file_name).split(os.sep)
                save_name = save_name[-3] + '-' + save_name[-1]
                name = out_dir + save_name + '_dtp.npz'
                np.savez_compressed(name, regions=regions, total=total)
                image.save(os.path.join(out_dir, 'image',save_name))
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


if __name__ == '__main__':

    #root_dir = "/data2/dengbowen/work/samples/difangzhi/"
    root_dir = "d:/lunwen/data/difangzhi/"
    image_dirs = [root_dir + f.name + '/jpg/' for f in os.scandir(root_dir) if f.is_dir() and f.name != 'images']
    for image_dir in image_dirs:
        image_names = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        out_dir = "/data2/dengbowen/difangzhi_dtp/"
        C_range = list(range(1, 40, 3))
        R_range = list(range(1, 40, 3))
        extract_dtp(out_dir, image_names, C_range, R_range, multiple_thresholds=True)
