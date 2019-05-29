import os
import numpy as np
import torch
from PIL import Image, ImageDraw


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    all boxes form: x1, y1, x2, y2
    boxes: (N, 4) ndarray
    query_boxes: (K, 4) ndarray
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes

    from https://github.com/ruotianluo/pytorch-faster-rcnn/blob/master/lib/utils/bbox.py
    """

    boxes = torch.as_tensor(boxes, dtype=torch.float64)
    query_boxes = torch.as_tensor(query_boxes, dtype=torch.float64)
    # (N,)
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # (K,)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1])

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t())).clamp(
        min=0)

    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t())).clamp(
        min=0)

    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih

    overlaps = iw * ih / ua

    return overlaps


def evaluate(gt_dir, detect_dir, threshold):
    gt_box_files = os.listdir(gt_dir)
    re = []
    pr = []
    for i, file_name in enumerate(gt_box_files):
        gt_boxes = []
        detect_boxes = []
        with open(os.path.join(gt_dir, file_name), 'r', encoding='utf-8') as f:
            for line in f:
                gt_boxes.append(list(map(int, line.split()[0: 4])))
        try:
            with open(os.path.join(detect_dir, file_name), 'r', encoding='utf-8') as f:
                for line in f:
                    detect_boxes.append(list(map(int, line.split()[0: 4])))
        except:
            print("%s not found!" % os.path.join(detect_dir, file_name))
        input_boxes = np.asarray(detect_boxes[::-1])
        target_boxes = np.asarray(gt_boxes)
        B1 = input_boxes.shape[0]
        B2 = target_boxes.shape[0]
        ious = bbox_overlaps(input_boxes, target_boxes).view(1, B1, B2)
        # N x B2
        # find the input boxes having max overlap with each gt box
        target_max_iou, target_idx = torch.max(ious, dim=1)
        pos_mask = torch.gt(target_max_iou, threshold)
        TP = 0
        box_id, index = target_idx.sort()
        i = 0
        l = box_id.size(1)
        while i < l:
            current_max = target_max_iou[0][index[0][i].item()]
            j = i + 1
            while j < l and box_id[0][j] == box_id[0][i]:
                temp = target_max_iou[0][index[0][j].item()]
                current_max = temp if temp > current_max else current_max
                j += 1
            if current_max > threshold:
                TP += 1
            i = j
        recall = TP / B2
        precision = TP / B1
        re.append(recall)
        pr.append(precision)
        print(file_name + " TP:{} Total Detection:{} Total GT:{}".format(TP, B1, B2), end='')
        print(" Recall:{:.2%}".format(recall), end='')
        print(" Precision:{:.2%}".format(precision))
    print("*" * 30)
    print("Avg recall:{:.2%} Avg precision:{:.2%}".format(sum(re) / len(re), sum(pr) / len(pr)))


if __name__ == '__main__':
    evaluate('c:/users/deng/desktop/g', 'c:/users/deng/desktop/d', 0.5)
