import os
import torch
from PIL import Image, ImageDraw
import logging
import easydict
import datetime
import time
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from misc.rcnn_dataset import RcnnDataset, RandomSampler
import ctrlfnet_model as ctrlf
from train_opts import parse_args
from evaluate import mAP

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s, %(levelname)s], %(message)s')
logger_name = "ctrlf-debugger"
logger = logging.getLogger(logger_name)
opt = parse_args()
print("load trainset")
trainset = RcnnDataset(opt, True, True, logger, '/mnt/data1/dengbowen/no_train/', 'difangzhi.json',
                       'difangzhi.h5')
print("load testset")
testset = RcnnDataset(opt, True, False, logger, '/mnt/data1/dengbowen/no_test/', 'test.json',
                      'test.h5')
print("finish loading dataset")

# all have been resized!
# out = (img, boxes) train
# out = (img, oshape, boxes, proposals) not train
#   valset = RcnnDataset(opt, False, logger, '/data2/dengbowen/work/samples/difangzhi_for_ctrlf/', 'difangzhi.json')
# testset = datasets.Dataset(opt, 'test', logger)
train_sampler = RandomSampler(None, opt.max_iters)
trainloader = DataLoader(trainset, batch_size=1, sampler=train_sampler, pin_memory=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)


# initialize the Ctrl-F-Net model object
model = ctrlf.CtrlFNet(opt, logger)

show = not opt.quiet
if show:
    print("number of parameters in ctrlfnet:", model.num_parameters())

model.cuda()
optimizer = optim.Adam(model.parameters(), opt.learning_rate, (opt.beta1, opt.beta2), opt.epsilon, opt.weight_decay)
keys = ['box-decay', 'end-reg', 'end-obj', 'mid-reg', 'mid-obj', 'total_loss']
running_losses = {k: 0.0 for k in keys}

it = 0
args = easydict.EasyDict()
args.nms_overlap = opt.query_nms_overlap
args.score_threshold = opt.score_threshold
args.num_queries = -1
args.score_nms_overlap = opt.score_nms_overlap
args.overlap_threshold = 0.5
args.gpu = True
args.use_external_proposals = int(opt.external_proposals)
args.max_proposals = opt.max_proposals
args.rpn_nms_thresh = opt.test_rpn_nms_thresh
args.num_workers = 6
args.numpy = False
args.num_workers = 6

trainlog = ''
start = time.time()

if opt.weights:
    opt.save_id += '_pretrained'

if not os.path.exists('checkpoints/ctrlfnet/'):
    os.makedirs('checkpoints/ctrlfnet/')

def test(img, boxes, i):
    img += 34.42953730765813
    img = img * 255
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for box in boxes:
        xc, yc, w, h = box[0], box[1], box[2], box[3]
        d.rectangle([xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2], outline='white')
    im.save("/data2/dengbowen/work/samples/ctrlf-out/predict{}.png".format(i))


def test_gt(img, boxes, i):
    img += 34.42953730765813
    img = img * 255
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for box in boxes:
        xc, yc, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        d.rectangle([xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2], outline='white')
    im.save("/data2/dengbowen/work/samples/ctrlf-out/gt{}.png".format(i))
    print(i)

model.train()
for data in trainloader:
    optimizer.zero_grad()
    #test(data[0].detach().cpu().numpy(), data[1].detach().cpu().numpy(), it)
    # [1, 1, 1720, 1032]  [1, 260, 4] [1, 260, 108]
    # make sure use float32 instead of float64
    img, gt_boxes = data[0].float(), data[1].float()
    # logger.debug(img.size())
    if args.gpu:
        img = img.cuda()
        gt_boxes = gt_boxes.cuda()


    input = (img, gt_boxes)
    if opt.dtp_train:
        proposals = data[2].float()
        if args.gpu:
            proposals = proposals.cuda()
        input += (proposals, )
    losses, predict_boxes = model(input)
    # # logger.debug(losses)
    optimizer.step()
    torch.cuda.empty_cache()
    # print statistics
    running_losses = {k: v + losses[k] for k, v in running_losses.items()}
    if it % opt.print_every == opt.print_every - 1:
        running_losses = {k: v / opt.print_every for k, v in running_losses.items()}
        loss_string = "[iter %5d] " % (it + 1)
        for k, v in running_losses.items():
            loss_string += "%s: %.2f | " % (k, v)
        trainlog += loss_string
        if show:
            print(loss_string)
        running_losses = {k: 0.0 for k, v in running_losses.items()}

    if it % opt.eval_every == opt.eval_every - 1:
        # predict_boxes_view = predict_boxes.detach().cpu().numpy()
        # test(data[0].cpu().numpy().squeeze(), predict_boxes_view, it)
        re, pr = mAP(model, testloader, args, logger, it, True)
        with open("/mnt/data1/dengbowen/res6.txt", "a", encoding="utf-8") as f:
            f.write("test at {} Threshold25 Recall:{:.2%}, Precision:{:.2%}\n".format(it, re[0], pr[0]))
            f.write("test at {} Threshold50 Recall:{:.2%}, Precision:{:.2%}\n".format(it, re[1], pr[1]))
            f.write("test at {} Threshold75 Recall:{:.2%}, Precision:{:.2%}\n".format(it, re[2], pr[2]))
            f.write("-"* 30 + "\n")
        if re[1] > 0.85 and pr[1] > 0.9:
            torch.save(model.state_dict(), os.path.join('./checkpoints/ctrlfnet', datetime.datetime.now().strftime("%m-%d %H:%M:%S") +
                                                   'rcnn_{:.2f}_{:.2f}.pth.tar'.format(re[1], pr[1])))
        #print("test at {} Recall:{}, Precision:{}".format(it, re, pr))
    # if it % opt.reduce_lr_every == opt.reduce_lr_every - 1:
    #     optimizer.param_groups[0]['lr'] /= 10.0

    it += 1

# if show:
#     if opt.val_dataset != 'iam':
#         model.load_weights(out_name)
#         log, rf, rt = mAP(model, testloader, args, it)
#         print(log)
#
#     d = {}
#     d['opt'] = opt
#     d['loss_history'] = loss_history
#     d['map_history'] = mAPs
#     d['trainlog'] = trainlog
#     d['testlog'] = log
#     with open(out_name + '.json', 'w') as f:
#         json.dump(d, f)
#
#     duration = time.time(
