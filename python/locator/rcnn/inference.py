import os
import sys
import torch
import logging
import easydict
import numpy as np
import rcnn_model as rcnn
import misc.box_utils as box_utils
from opts import parse_args
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from misc.boxIoU import bbox_overlaps
from misc.rcnn_dataset import RcnnDataset


def inference():
    logger_name = "locator"
    logger = logging.getLogger(logger_name)
    opt = parse_args()
    logger.info("load testset")
    # trailing slash is necessary!
    # change the 4th parameter
    testset = RcnnDataset(opt, opt.dtp_test, True, logger, '/mnt/data1/dengbowen/python/locator/rcnn/dataset/train/',
                          'difangzhi.json', 'difangzhi.h5')
    logger.info("finish loading dataset")
    image_mean = testset.image_mean
    # all have been resized!
    testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    model = rcnn.RcnnNet(opt, logger)

    if not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, using CPU mode')
        opt.gpu = False

    device = torch.device("cuda:0" if opt.gpu else "cpu")
    model.to(device)

    args = easydict.EasyDict()
    args.score_threshold = opt.score_threshold
    args.score_nms_overlap = opt.score_nms_overlap
    args.out_path = opt.out_path
    args.gpu = opt.gpu
    args.max_proposals = opt.max_proposals
    args.use_external_proposals = opt.dtp_test
    args.rpn_nms_thresh = opt.test_rpn_nms_thresh
    args.numpy = False
    args.verbose = False

    try:
        model_dict = torch.load(opt.model_path)
        model.load_state_dict(model_dict)
    except Exception:
        logger.critical("cannot load model")
        sys.exit(-1)

    if not os.path.exists('./test_out/'):
        os.makedirs('./test_out/')
    model.eval()
    outputs = []
    for data in testloader:
        # make sure use float32 instead of float64
        if opt.dtp_test:
            (img, gt_boxes, external_proposals, (H, W)) = data
            input = (img, gt_boxes[0].float(), external_proposals[0].float())
        else:
            (img, gt_boxes, (H, W)) = data
            input = (img, gt_boxes[0].float())
        if args.max_proposals == -1:
            if opt.dtp_test:
                model.setTestArgs({'rpn_nms_thresh': args.rpn_nms_thresh, 'max_proposals': external_proposals.size(1)})
            else:
                model.setTestArgs({'rpn_nms_thresh': args.rpn_nms_thresh,
                                   'max_proposals': 1000})
        else:
            model.setTestArgs({'rpn_nms_thresh': args.rpn_nms_thresh, 'max_proposals': args.max_proposals})
        out = model.evaluate(input, args)
        outputs.append(out)
    score_nms_overlap = args.score_nms_overlap  # For wordness scores
    score_threshold = args.score_threshold
    thresholds = [0.25, 0.5, 0.75]
    re = [[], [], []]
    pr = [[], [], []]
    counter = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            if opt.dtp_test:
                roi_scores, proposals, eproposal_scores = outputs[i]
                (img, gt_boxes, external_proposals, (H, W)) = data
                external_proposals = box_utils.xcycwh_to_x1y1x2y2(external_proposals[0].float())
            else:
                roi_scores, proposals = outputs[i]
                (img, gt_boxes, (H, W)) = data
            scale = 900 / max(H.item(), W.item())
            gt_boxes = box_utils.xcycwh_to_x1y1x2y2(gt_boxes[0].float())

            # convert to probabilities with sigmoid
            scores = 1 / (1 + torch.exp(-roi_scores))
            if args.verbose:
                logger.info("in postprocessing at the very first")
                logger.info("rpn size {}".format(proposals.size()))
                if opt.dtp_test:
                    logger.info("dtp size {}".format(external_proposals.size()))
            if opt.dtp_test:
                eproposal_scores = 1 / (1 + torch.exp(-eproposal_scores))
                scores = torch.cat((scores, eproposal_scores), 0)
                proposals = torch.cat((proposals, external_proposals), 0)

            # Since slicing empty array doesn't work in torch, we need to do this explicitly
            if opt.dtp_test:
                nrpn = len(roi_scores)
                rpn_proposals = proposals[:nrpn]
                dtp_proposals = proposals[nrpn:]

            threshold_pick = torch.squeeze(scores > score_threshold)
            scores = scores[threshold_pick]
            tmp = threshold_pick.view(-1, 1).expand(threshold_pick.size(0), 4)
            proposals = proposals[tmp].view(-1, 4)

            if opt.dtp_test:
                rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
                dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)

            if args.verbose:
                logger.info("in postprocessing after score prune")
                logger.info("total size ".format(proposals.size()))
                if opt.dtp_test:
                    logger.info("rpn size ".format(rpn_proposals.size()))
                    logger.info("dtp size ".format(dtp_proposals.size()))

            dets = torch.cat([proposals.float(), scores.view(-1, 1)], 1)
            if dets.size(0) <= 1:
                continue

            pick = box_utils.nms(dets, score_nms_overlap)
            tt = torch.zeros(len(dets)).byte().cuda()
            tt[pick] = 1

            proposals = proposals[pick]

            if opt.dtp_test:
                nrpn = rpn_proposals.size(0)
                tmp = tt.view(-1, 1).expand(tt.size(0), 4)
                rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
                dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)

            if args.verbose:
                logger.info("in postprocessing after nms")
                logger.info("total size ".format(proposals.size()))
                if opt.dtp_test:
                    logger.info("rpn size ".format(rpn_proposals.size()))
                    logger.info("dtp size ".format(dtp_proposals.size()))
            B1 = proposals.shape[0]
            B2 = gt_boxes.shape[0]
            ious = bbox_overlaps(proposals, gt_boxes).view(1, B1, B2)
            # N x B2
            # find the input boxes having max overlap with each gt box
            target_max_iou, target_idx = torch.max(ious, dim=1)
            for m, threshold in enumerate(thresholds):
                TP = 0
                box_id, index = target_idx.sort()
                j = 0
                l = box_id.size(1)
                while j < l:
                    current_max = target_max_iou[0][index[0][j].item()]
                    k = j + 1
                    while k < l and box_id[0][k] == box_id[0][j]:
                        temp = target_max_iou[0][index[0][k].item()]
                        current_max = temp if temp > current_max else current_max
                        k += 1
                    if current_max > threshold:
                        TP += 1
                    j = k
                recall = TP / B2
                precision = TP / B1

                re[m].append(recall)

                pr[m].append(precision)
            proposals = proposals.cpu().numpy()
            with open(os.path.join('test_out', '{}.txt'.format(counter)), 'w', encoding='utf-8') as f:
                for box in proposals:
                    x1, y1, x2, y2 = int(box[0] / scale), int(box[1] / scale), int(box[2] / scale), int(box[3] / scale)
                    f.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "\n")
            counter = counter + 1
            if counter == 201:
                counter = 301
    for i in range(3):
        try:
            re[i] = sum(re[i]) / len(re[i])
            pr[i] = sum(pr[i]) / len(pr[i])
        except ZeroDivisionError:
            re[i] = 0
            pr[i] = 0
    print("Threshold25 Recall:{:.2%}, Precision:{:.2%}".format(re[0], pr[0]))
    print("Threshold50 Recall:{:.2%}, Precision:{:.2%}".format(re[1], pr[1]))
    print("Threshold75 Recall:{:.2%}, Precision:{:.2%}".format(re[2], pr[2]))


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    inference()
