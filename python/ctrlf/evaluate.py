#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:56:02 2017

@author: tomas
"""
from contextlib import closing
from multiprocessing import Pool as PyPool
import easydict
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
        out = model.evaluate(input, args.gpu, numpy)
        outputs.append(out)

    model.train()
    return outputs


def mAP(model, loader, args, logger, i):
    features = extract_features(model, loader, args, args.numpy)
    recall = 3
    args.overlap_thresholds = [0.25, 0.5]

    args.use_external_proposals = True
    res_true = mAP_eval(features, loader, args, logger, i)
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


def mAP_eval(features, loader, args, logger, i):
    d = postprocessing(features, loader, args, logger, i)

    return d


def postprocessing(features, loader, args, logger, kk):
    score_nms_overlap = args.score_nms_overlap  # For wordness scores
    score_threshold = args.score_threshold

    for li, data in enumerate(loader):
        roi_scores, eproposal_scores, proposals = features[li]
        (img, gt_boxes, external_proposals) = data

        # boxes are xcycwh from dataloader, convert to x1y1x2y2
        external_proposals = box_utils.xcycwh_to_x1y1x2y2(external_proposals[0].float())
        gt_boxes = box_utils.xcycwh_to_x1y1x2y2(gt_boxes[0].float())




        roi_scores = roi_scores.cuda()
        eproposal_scores = eproposal_scores.cuda()

        proposals = proposals.cuda()

        external_proposals = external_proposals.cuda()

        # convert to probabilities with sigmoid
        scores = 1 / (1 + torch.exp(-roi_scores))
        logger.debug("in postprocessing at the very first")
        print("rpn ", proposals.size())
        print("dtp ", external_proposals.size())
        if args.use_external_proposals:
            eproposal_scores = 1 / (1 + torch.exp(-eproposal_scores))
            scores = torch.cat((scores, eproposal_scores), 0)
            proposals = torch.cat((proposals, external_proposals), 0)

        print("eproposal_scores size", eproposal_scores.size())
        print("roi scores size", roi_scores.size())
        print("proposal size", proposals.size())
        # calculate the different recalls before NMS
        # entry = {}
        # recalls(proposals, gt_boxes, overlap_thresholds, entry, '1_total')

        # Since slicing empty array doesn't work in torch, we need to do this explicitly
        if args.use_external_proposals:
            nrpn = len(roi_scores)
            print(nrpn)
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
        # logger.debug("in postprocessing after score prune")
        #print(score_threshold)
        #print(threshold_pick)
        # print("total ", proposals.size())
        # print("rpn ", rpn_proposals.size())
        # print("dtp ", dtp_proposals.size())


        dets = torch.cat([proposals.float(), scores.view(-1, 1)], 1)
        if dets.size(0) <= 1:
            continue

        pick = box_utils.nms(dets, score_nms_overlap)
        tt = torch.zeros(len(dets)).byte().cuda()
        tt[pick] = 1

        proposals = proposals[pick]

        #scores = scores[pick]
    # recalls(proposals, gt_boxes, overlap_thresholds, entry, '3_total')
        if args.use_external_proposals:
            nrpn = rpn_proposals.size(0)
            tmp = tt.view(-1, 1).expand(tt.size(0), 4)
            rpn_proposals = rpn_proposals[tmp[:nrpn]].view(-1, 4)
            dtp_proposals = dtp_proposals[tmp[nrpn:]].view(-1, 4)
        # recalls(dtp_proposals, gt_boxes, overlap_thresholds, entry, '3_dtp')
        # recalls(rpn_proposals, gt_boxes, overlap_thresholds, entry, '3_rpn')
        logger.debug("in postprocessing after nms")
        print("total ", proposals.size())
        print("rpn ", rpn_proposals.size())
        print("dtp ", dtp_proposals.size())
        r = rpn_proposals.cpu().numpy()
        dtp = dtp_proposals.cpu().numpy()

        img = img.numpy().squeeze()
        img = img + 34.42953730765813 / 255
        img = img * 255
        img = img.astype(np.uint8)
        im1 = img.copy()
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        for i, box in enumerate(r):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            d.rectangle([x1, y1, x2, y2], outline='white')
        im.save("/mnt/data1/dengbowen/ctrlf/out/rpn/it{}-{}.png".format(kk, li))
        im = Image.fromarray(im1)
        d = ImageDraw.Draw(im)
        for i, box in enumerate(dtp):
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            d.rectangle([x1, y1, x2, y2], outline='white')
        im.save("/mnt/data1/dengbowen/ctrlf/out/dtp/it{}-{}.png".format(kk, li))
    # overlap = bbox_overlaps(proposals, gt_boxes)
    # overlaps.append(overlap)
    # max_gt_overlap, amax_gt_overlap = overlap.max(dim=1)
    # proposal_labels = torch.Tensor([gt_labels[i] for i in amax_gt_overlap])
    # proposal_labels = proposal_labels.cuda()
    # mask = overlap.sum(dim=1) == 0
    # proposal_labels[mask] = loader.dataset.get_vocab_size() + 1
    #
    # max_overlaps.append(max_gt_overlap)
    # amax_overlaps.append(amax_gt_overlap + n_gt)
    # n_gt += len(gt_boxes)

    # Artificially make a huge image containing all the boxes to be able to
    # perform nms on distance to query
    # proposals[:, 0] += offset[1]
    # proposals[:, 1] += offset[0]
    # proposals[:, 2] += offset[1]
    # proposals[:, 3] += offset[0]
    # joint_boxes.append(proposals)
    # offset[0] += img.shape[0]
    # offset[1] += img.shape[1]
    #
    # db_targets.append(proposal_labels)
    # db.append(embeddings)
    # log.append(entry)


#
# db = torch.cat(db, dim=0)
# db_targets = torch.cat(db_targets, dim=0)
# joint_boxes = torch.cat(joint_boxes, dim=0)
# max_overlaps = torch.cat(max_overlaps, dim=0)
# amax_overlaps = torch.cat(amax_overlaps, dim=0)
# all_gt_boxes = torch.cat(all_gt_boxes, dim=0)
#
#
# assert db.shape[0] == db_targets.shape[0]
#
#
# db_targets = db_targets.cpu()
# joint_boxes = joint_boxes.cpu()
# max_overlaps = max_overlaps.cpu()
# amax_overlaps = amax_overlaps.cpu()
#
#
# db_targets = db_targets.numpy()
# joint_boxes = joint_boxes.numpy()
# max_overlaps = max_overlaps.numpy()
# amax_overlaps = amax_overlaps.numpy()
#
# # A hack for some printing compatability
# if not args.use_external_proposals:
#     keys = []
#     for i in range(1, 4):
#         for ot in args.overlap_thresholds:
#             keys += ['%d_dtp_recall_%d' % (i, ot * 100),
#                      '%d_rpn_recall_%d' % (i, ot * 100)]
#
#     for entry in log:
#         for key in keys:
#             if key not in entry:
#                 entry[key] = entry['1_total_recall_50']

    return rpn_proposals, dtp_proposals
