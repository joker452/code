"""
Created on Thu Nov  9 14:56:02 2017

@author: tomas
"""
import os
import torch
import numpy as np
import misc.box_utils as box_utils
from PIL import Image, ImageDraw
from misc.boxIoU import bbox_overlaps

np.errstate(divide='ignore', invalid='ignore')


def extract_features(model, loader, args):
    outputs = []
    model.eval()
    for i, data in enumerate(loader):
        if args.use_external_proposals:
            (img, gt_boxes, external_proposals) = data
            input = (img, gt_boxes[0].float(), external_proposals[0].float())
        else:
            (img, gt_boxes) = data
            input = (img, gt_boxes[0].float())
        if args.max_proposals == -1:
            if args.use_external_proposals:
                model.setTestArgs({'rpn_nms_thresh': args.rpn_nms_thresh, 'max_proposals': external_proposals.size(1)})
            else:
                model.setTestArgs({'rpn_nms_thresh': args.rpn_nms_thresh,
                                   'max_proposals': 1000})
        else:
            model.setTestArgs({'rpn_nms_thresh': args.rpn_nms_thresh, 'max_proposals': args.max_proposals})

        # out is cpu tensor by default
        out = model.evaluate(input, args)
        outputs.append(out)

    model.train()
    return outputs


def mAP(model, loader, args, logger, i):
    features = extract_features(model, loader, args)
    overlap_thresholds = [0.25, 0.5, 0.75]
    res = postprocessing(features, loader, args, logger, i, overlap_thresholds)
    return res


def postprocessing(features, loader, args, logger, it, thresholds):
    score_nms_overlap = args.score_nms_overlap  # For wordness scores
    score_threshold = args.score_threshold
    re = [[], [], []]
    pr = [[], [], []]
    with torch.no_grad():
        for i, data in enumerate(loader):
            if args.use_external_proposals:
                roi_scores, proposals, eproposal_scores = features[i]
                (img, gt_boxes, external_proposals) = data
                external_proposals = box_utils.xcycwh_to_x1y1x2y2(external_proposals[0].float())
            else:
                roi_scores, proposals = features[i]
                (img, gt_boxes) = data

            # boxes are xcycwh from dataloader, convert to x1y1x2y2

            gt_boxes = box_utils.xcycwh_to_x1y1x2y2(gt_boxes[0].float())

            # convert to probabilities with sigmoid
            scores = 1 / (1 + torch.exp(-roi_scores))
            if args.verbose:
                logger.info("in postprocessing at the very first")
                logger.info("rpn size {}".format(proposals.size()))
                if args.use_external_proposals:
                    logger.info("dtp size {}".format(external_proposals.size()))
            if args.use_external_proposals:
                eproposal_scores = 1 / (1 + torch.exp(-eproposal_scores))
                scores = torch.cat((scores, eproposal_scores), 0)
                proposals = torch.cat((proposals, external_proposals), 0)

            # Since slicing empty array doesn't work in torch, we need to do this explicitly
            if args.use_external_proposals:
                nrpn = len(roi_scores)
                rpn_proposals = proposals[:nrpn]
                dtp_proposals = proposals[nrpn:]

            threshold_pick = torch.squeeze(scores > score_threshold)
            scores = scores[threshold_pick]
            tmp = threshold_pick.view(-1, 1).expand(threshold_pick.size(0), 4)
            proposals = proposals[tmp].view(-1, 4)

            if args.use_external_proposals:
                rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
                dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)

            if args.verbose:
                logger.info("in postprocessing after score prune")
                logger.info("total size ".format(proposals.size()))
                if args.use_external_proposals:
                    logger.info("rpn size ".format(rpn_proposals.size()))
                    logger.info("dtp size ".format(dtp_proposals.size()))

            dets = torch.cat([proposals.float(), scores.view(-1, 1)], 1)
            if dets.size(0) <= 1:
                continue

            pick = box_utils.nms(dets, score_nms_overlap)
            tt = torch.zeros(len(dets)).byte().cuda()
            tt[pick] = 1

            proposals = proposals[pick]

            if args.use_external_proposals:
                nrpn = rpn_proposals.size(0)
                tmp = tt.view(-1, 1).expand(tt.size(0), 4)
                rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
                dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)

            if args.verbose:
                logger.info("in postprocessing after nms")
                logger.info("total size ".format(proposals.size()))
                if args.use_external_proposals:
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

            img = img.numpy().squeeze()
            img = img + args.image_mean / 255
            img = img * 255
            img = img.astype(np.uint8)
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            with open(os.path.join(args.out_path, 'it{}-{}.txt'.format(it, i)), 'w', encoding='utf-8') as f:
                for box in proposals:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    d.rectangle([x1, y1, x2, y2], outline='white')
                    f.write(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2) + "\n")
            im.save(os.path.join(args.out_path, "image", "it{}-{}.png".format(it, i)))
    for i in range(3):
        try:
            re[i] = sum(re[i]) / len(re[i])
            pr[i] = sum(pr[i]) / len(pr[i])
        except ZeroDivisionError:
            re[i] = 0
            pr[i] = 0
    return re, pr
