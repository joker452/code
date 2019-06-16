import os
import json
import torch
import logging
import easydict
import datetime
import numpy as np
import torch.optim as optim
import rcnn_model as ctrlf
from evaluate import mAP
from opts import parse_args
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from misc.rcnn_dataset import RcnnDataset, RandomSampler


def train():
    logger_name = "rcnn-locator"
    logger = logging.getLogger(logger_name)
    opt = parse_args()
    logger.info("load trainset")
    # trailing slash is necessary!
    trainset = RcnnDataset(opt, opt.dtp_train, True, logger, '/mnt/data1/dengbowen/python/locator/rcnn/dataset/train/',
                           'difangzhi.json', 'difangzhi.h5')
    logger.info("load testset")
    testset = RcnnDataset(opt, opt.dtp_test, False, logger, '/mnt/data1/dengbowen/python/locator/rcnn/dataset/test/',
                          'test.json', 'test.h5')
    logger.info("finish loading dataset")

    # all have been resized!
    # out = (img, boxes) dtp_train
    # out = (img, oshape, boxes, proposals) not dtp_train
    train_sampler = RandomSampler(None, opt.max_iters)
    trainloader = DataLoader(trainset, batch_size=1, sampler=train_sampler, pin_memory=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)

    # initialize the Ctrl-F-Net model object
    model = rcnn.RcnnNet(opt, logger)

    show = not opt.quiet
    if show:
        logger.info("number of parameters in rcnn-net:{}".format(model.num_parameters()))

    if not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, using CPU mode')
        opt.gpu = False

    device = torch.device("cuda:0" if opt.gpu else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), opt.learning_rate, (opt.beta1, opt.beta2), opt.epsilon,
                           opt.weight_decay)

    keys = ['box-decay', 'end-reg', 'end-obj', 'mid-reg', 'mid-obj', 'total_loss']
    running_losses = {k: 0.0 for k in keys}

    it = 0
    args = easydict.EasyDict()
    args.score_threshold = opt.score_threshold
    args.score_nms_overlap = opt.score_nms_overlap
    args.out_path = opt.out_path
    args.gpu = opt.gpu
    args.use_external_proposals = opt.dtp_test
    args.max_proposals = opt.max_proposals
    args.rpn_nms_thresh = opt.test_rpn_nms_thresh
    args.numpy = False
    args.verbose = False
    args.image_mean = testset.image_mean

    trainlog = ''

    if opt.pretrained:
        try:
            model_dict = torch.load(opt.model_path)
            model.load_state_dict(model_dict)
        except Exception:
            logger.critical("cannot load model")

    if not os.path.exists('checkpoints/rcnn/'):
        os.makedirs('checkpoints/rcnn/')
    if not os.path.exists('./log/'):
        os.makedirs('./log/')
    if not os.path.exists('./out/dtp'):
        os.makedirs('./out/dtp')
    if not os.path.exists('./out/rpn'):
        os.makedirs('./out/rpn')
    model.train()
    start_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
    for data in trainloader:
        optimizer.zero_grad()
        # test(data[0].detach().cpu().numpy(), data[1].detach().cpu().numpy(), it)
        # [1, 1, 1720, 1032]  [1, 260, 4] [1, 260, 108]
        # make sure use float32 instead of float64
        img, gt_boxes = data[0].float(), data[1].float()
        img = img.to(device)
        gt_boxes = gt_boxes.to(device)
        input = (img, gt_boxes)
        if opt.dtp_train:
            proposals = data[2].float()
            proposals = proposals.to(device)
            input += (proposals,)
        losses, predict_boxes = model(input)
        optimizer.step()
        # print statistics
        running_losses = {k: v + losses[k] for k, v in running_losses.items()}
        if it % opt.print_every == opt.print_every - 1:
            running_losses = {k: v / opt.print_every for k, v in running_losses.items()}
            loss_string = "[iter %5d] " % (it + 1)
            for k, v in running_losses.items():
                loss_string += "%s: %.2f | " % (k, v)
            trainlog += loss_string
            if show:
                logger.info(loss_string)
            running_losses = {k: 0.0 for k, v in running_losses.items()}

        if it % opt.eval_every == opt.eval_every - 1:
            # predict_boxes_view = predict_boxes.detach().cpu().numpy()
            # test(data[0].cpu().numpy().squeeze(), predict_boxes_view, it)
            re, pr = mAP(model, testloader, args, logger, it)
            with open("./log/res_" + start_time + ".txt", "a", encoding="utf-8") as f:
                f.write("test at {} Threshold25 Recall:{:.2%}, Precision:{:.2%}\n".format(it, re[0], pr[0]))
                f.write("test at {} Threshold50 Recall:{:.2%}, Precision:{:.2%}\n".format(it, re[1], pr[1]))
                f.write("test at {} Threshold75 Recall:{:.2%}, Precision:{:.2%}\n".format(it, re[2], pr[2]))
                f.write("-" * 30 + "\n")
            if re[1] > 0.8 and pr[1] > 0.8:
                torch.save(model.state_dict(),
                           os.path.join('./checkpoints/rcnn',
                                        datetime.datetime.now().strftime("%m_%d_%H:%M") +
                                        'rcnn_{:.2f}_{:.2f}.pth.tar'.format(re[1], pr[1])))
        it += 1
    d = {}
    d['trainlog'] = trainlog
    with open('./log/' + start_time + '.json', 'w') as f:
        json.dump(d, f)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)
    train()
