#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:56:02 2017

@author: tomas
"""
import torch
import numpy as np
from PIL import Image, ImageDraw

np.errstate(divide='ignore', invalid='ignore')
import misc.box_utils as box_utils
from misc.boxIoU import bbox_overlaps

def my_unique(tensor1d):
    """ until pytorch adds this """
    t, idx = np.unique(tensor1d.cpu().numpy(), return_inverse=True)
    return t.shape[0]


def recall_torch(proposals, gt_boxes, ot):
    if proposals.nelement() == 0:
        return 0.0
    overlap = bbox_overlaps(proposals, gt_boxes)
    vals, inds = overlap.max(dim=1)
    i = vals >= ot
    covered = my_unique(inds[i])
    recall = float(covered) / float(gt_boxes.size(0))
    return recall


def recalls(proposals, gt_boxes, overlap_thresholds, entry, key):
    for ot in overlap_thresholds:
        entry['%s_recall_%d' % (key, ot * 100)] = recall_torch(proposals, gt_boxes, ot)


def extract_features(model, loader, args, numpy=True):
    outputs = []
    model.eval()
    for i, data in enumerate(loader):
        (img, gt_boxes, external_proposals) = data

        if args.max_proposals == -1:
            model.setTestArgs({'rpn_nms_thresh': args.rpn_nms_thresh, 'max_proposals': external_proposals.size(1)})
        else:
            model.setTestArgs({'rpn_nms_thresh': args.rpn_nms_thresh, 'max_proposals': args.max_proposals})

        input = (img, gt_boxes[0].float(), external_proposals[0].float())
        # out is cpu tensor by default
        out = model.evaluate(input, args.gpu, numpy)
        outputs.append(out)

    model.train()
    return outputs


def mAP(model, loader, args, logger, i, verbose):
    features = extract_features(model, loader, args, args.numpy)
    overlap_thresholds = [0.25, 0.5, 0.75]

    args.use_external_proposals = True
    res_true = postprocessing(features, loader, args, logger, i, verbose, overlap_thresholds)
    # total_recall = np.mean([e['%d_total_recall_25' % recall] for e in res_true.log])
    # rpn_recall = np.mean([e['%d_rpn_recall_25' % recall] for e in res_true.log])
    # pargs = (res_true.mAP_qbe_25 * 100, res_true.mAP_qbs_25 * 100, total_recall * 100, rpn_recall * 100)
    # rs3 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, rpn_recall: %.1f, With DTP 25%% overlap' % pargs
    # total_recall = np.mean([e['%d_total_recall_50' % recall] for e in res_true.log])
    # rpn_recall = np.mean([e['%d_rpn_recall_50' % recall] for e in res_true.log])
    # pargs = (res_true.mAP_qbe_50 * 100, res_true.mAP_qbs_50 * 100, total_recall * 100, rpn_recall * 100)
    # rs4 = 'QbE mAP: %.1f, QbS mAP: %.1f, recall: %.1f, rpn_recall: %.1f, With DTP 50%% overlap' % pargs
    # log = '--------------------------------\n'
    # log += '[%s set iter %d] %s\n' % (split, it + 1, rs3)
    # log += '[%s set iter %d] %s\n' % (split, it + 1, rs4)
    # log += '--------------------------------\n'
    # return log, res, res_true
    return res_true


def postprocessing(features, loader, args, logger, it, verbose, thresholds):
    score_nms_overlap = args.score_nms_overlap  # For wordness scores
    score_threshold = args.score_threshold
    re = [[], [], []]
    pr = [[], [], []]
    with torch.no_grad():
        for i, data in enumerate(loader):
            roi_scores, eproposal_scores, proposals = features[i]
            (img, gt_boxes, external_proposals) = data

            # boxes are xcycwh from dataloader, convert to x1y1x2y2
            external_proposals = box_utils.xcycwh_to_x1y1x2y2(external_proposals[0].float())
            gt_boxes = box_utils.xcycwh_to_x1y1x2y2(gt_boxes[0].float())


            # convert to probabilities with sigmoid
            scores = 1 / (1 + torch.exp(-roi_scores))
            if verbose:
                logger.debug("in postprocessing at the very first")
                print("rpn size", proposals.size())
                print("dtp size", external_proposals.size())
            if args.use_external_proposals:
                eproposal_scores = 1 / (1 + torch.exp(-eproposal_scores))
                scores = torch.cat((scores, eproposal_scores), 0)
                proposals = torch.cat((proposals, external_proposals), 0)
            if verbose:
                print("proposal size", proposals.size())
            # calculate the different recalls before NMS
            # entry = {}
            # recalls(proposals, gt_boxes, overlap_thresholds, entry, '1_total')

            # Since slicing empty array doesn't work in torch, we need to do this explicitly
            if args.use_external_proposals:
                nrpn = len(roi_scores)
                rpn_proposals = proposals[:nrpn]
                dtp_proposals = proposals[nrpn:]
            #     recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '1_dtp')
            #     recalls(rpn_proposals, gt_boxes, overlap_thresholds, entry, '1_rpn')

            threshold_pick = torch.squeeze(scores > score_threshold)
            scores = scores[threshold_pick]
            tmp = threshold_pick.view(-1, 1).expand(threshold_pick.size(0), 4)
            proposals = proposals[tmp].view(-1, 4)

            # recalls(proposals, gt_boxes, overlap_thresholds, entry, '2_total')

            if args.use_external_proposals:
                rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
                dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)
            # recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '2_dtp')
            # recalls(rpn_proposals, gt_boxes, overlap_thresholds, entry, '2_rpn')
            if verbose:
                logger.debug("in postprocessing after score prune")
                print("total ", proposals.size())
                print("rpn ", rpn_proposals.size())
                print("dtp ", dtp_proposals.size())

            dets = torch.cat([proposals.float(), scores.view(-1, 1)], 1)
            if dets.size(0) <= 1:
                continue

            pick = box_utils.nms(dets, score_nms_overlap)
            tt = torch.zeros(len(dets)).byte().cuda()
            tt[pick] = 1

            proposals = proposals[pick]

            # scores = scores[pick]
            # recalls(proposals, gt_boxes, overlap_thresholds, entry, '3_total')
            if args.use_external_proposals:
                nrpn = rpn_proposals.size(0)
                tmp = tt.view(-1, 1).expand(tt.size(0), 4)
                rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
                dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)
            # recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '3_dtp')
            # recalls(rpn_proposals, gt_boxes, overlap_thresholds, entry, '3_rpn')
            if verbose:
                logger.debug("in postprocessing after nms")
                print("total ", proposals.size())
                print("rpn ", rpn_proposals.size())
                print("dtp ", dtp_proposals.size())
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
                #print(recall, precision)
                re[m].append(recall)
                #(m, re[m])
                pr[m].append(precision)
            r = rpn_proposals.cpu().numpy()
            proposals = proposals.cpu().numpy()

            img = img.numpy().squeeze()
            img = img + 26.828410879159787 / 255
            img = img * 255
            img = img.astype(np.uint8)
            im1 = img.copy()
            im = Image.fromarray(img)
            d = ImageDraw.Draw(im)
            for box in r:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                d.rectangle([x1, y1, x2, y2], outline='white')
            im.save("/mnt/data1/dengbowen/ctrlf/out3/rpn/it{}-{}.png".format(it, i))
            im = Image.fromarray(im1)
            d = ImageDraw.Draw(im)
            for box in proposals:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                d.rectangle([x1, y1, x2, y2], outline='white')
            im.save("/mnt/data1/dengbowen/ctrlf/out3/dtp/it{}-{}.png".format(it, i))
    for i in range(3):
        re[i] = sum(re[i]) / len(re[i])
        pr[i] = sum(pr[i]) / len(pr[i])
    return re, pr
