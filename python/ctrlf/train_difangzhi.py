import warnings

warnings.filterwarnings("ignore")
import os
from PIL import Image, ImageDraw
import logging
import easydict
import time
import numpy as np
import torch
from misc.dataloader import DataLoader
import torch.optim as optim
import misc.datasets as datasets
from misc.rcnn_dataset import RcnnDataset, RandomSampler
import ctrlfnet_model as ctrlf
from train_opts import parse_args
from evaluate import mAP
import misc.h5_dataset as h5_dataset

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s, %(levelname)s], %(message)s')
logger_name = "ctrlf-debugger"
logger = logging.getLogger(logger_name)
opt = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

if opt.h5:
    trainset = h5_dataset.H5Dataset(opt, split=0)
    valset = h5_dataset.H5Dataset(opt, split=1)
    testset = h5_dataset.H5Dataset(opt, split=2)
    opt.num_workers = 0
else:
    if opt.dataset.find('iiit_hws') > -1:
        trainset = datasets.SegmentedDataset(opt, 'train')
    else:
        trainset = RcnnDataset(opt, True, logger, '/data2/dengbowen/work/samples/no/', 'difangzhi.json',
                               'difangzhi.h5')
    # all have been resized!
    # out = (img, boxes) train
    # out = (img, oshape, boxes, proposals) not train
#   valset = RcnnDataset(opt, False, logger, '/data2/dengbowen/work/samples/difangzhi_for_ctrlf/', 'difangzhi.json')
#testset = datasets.Dataset(opt, 'test', logger)
train_sampler = RandomSampler(opt.max_iters, True)
test_sampler = RandomSampler(opt.max_iters, False)
trainloader = DataLoader(trainset, batch_size=1, sampler=train_sampler, num_workers=0)
valloader = DataLoader(trainset, batch_size=1, shuffle=test_sampler, num_workers=0)
print("data load finish")

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.set_default_tensor_type('torch.FloatTensor')
torch.cuda.device(opt.gpu)

# initialize the Ctrl-F-Net model object
model = ctrlf.CtrlFNet(opt, logger)

show = not opt.quiet
if show:
    print("number of parameters in ctrlfnet:", model.num_parameters())

model.cuda()
optimizer = optim.Adam(model.parameters(), opt.learning_rate, (opt.beta1, opt.beta2), opt.epsilon, opt.weight_decay)
keys = ['bd', 'ebr', 'eo', 'mbr', 'mo', 'total_loss']
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
loss_history, mAPs = [], []
if opt.eval_first_iteration:
    rpn_proposals, dtp_proposals = mAP(model, valloader, args, logger, 0)
    # trainlog += log
    # if show:
    #     print(log)
    # best_score = (rt.mAP_qbe_50 + rt.mAP_qbs_50) / 2
    # mAPs.append((it, [rt.mAP_qbe_50, rt.mAP_qbs_50]))

if opt.weights:
    opt.save_id += '_pretrained'

if not os.path.exists('checkpoints/ctrlfnet/'):
    os.makedirs('checkpoints/ctrlfnet/')

oargs = ('ctrlfnet', opt.dataset, opt.fold, opt.save_id)
out_name = 'checkpoints/%s/%s_%d_fold%s_best_val.pt' % oargs


def test(img, boxes, i):
    img = img * 255
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for box in boxes:
        xc, yc, w, h = box[0], box[1], box[2], box[3]
        d.rectangle([xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2], outline='black')
    im.save("/data2/dengbowen/work/samples/ctrlf-out/{}.png".format(i))

def test_gt(img, boxes, i):
    img = img * 255
    img = img.astype(np.uint8)
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for box in boxes:
        xc, yc, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        d.rectangle([xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2], outline='white')
    im.save("/data2/dengbowen/work/samples/ctrlf-out/{}.png".format(i))
    print(i)
for data in trainloader:
    optimizer.zero_grad()
    # test(data[0].cpu().numpy().squeeze(), data[1].cpu().numpy().squeeze(), it)

    losses, predict_boxes = model.forward_backward(logger, data, True)
    # # logger.debug(losses)
    optimizer.step()
    # print statistics
    running_losses = {k: v + losses[k] for k, v in running_losses.items()}
    if it % opt.print_every == opt.print_every - 1:
        running_losses = {k: v / opt.print_every for k, v in running_losses.items()}
        loss_string = "[iter %5d] " % (it + 1)
        for k, v in running_losses.items():
            loss_string += "%s: %.5f | " % (k, v)
        trainlog += loss_string
        if show:
            print(loss_string)
        logger.debug("running_losses: {}".format(running_losses))
        vals = [val for val in list(running_losses.values())]
        loss_history.append((it, vals))
        running_losses = {k: 0.0 for k, v in running_losses.items()}

    if it % opt.eval_every == opt.eval_every - 1:
        predict_boxes_view = predict_boxes.detach().cpu().numpy()
        test(data[0].cpu().numpy().squeeze(), predict_boxes_view, it)
    # mAP(model, valloader, args, logger, it)
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
#     duration = time.time() - start
#     print("training model took %0.2f hours" % (duration / 3600))
