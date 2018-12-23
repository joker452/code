import os
import ast
import tqdm
import time
import logging
import datetime
import argparse
import torch.autograd
import torch.cuda
import torch.nn as nn
import torch.optim
from datasets.casia_alexnet import CASIA
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
from models.alexnet import AlexNet
from tensorboardX import SummaryWriter


def count_parameter(m):
    return sum(parameter.numel() for parameter in m.parameters() if parameter.requires_grad)


def init_net(m):
    if isinstance(m, nn.Conv2d):
        if m.weight is not None:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.normal_(m.weight, 0, 0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train():
    logger = logging.getLogger('CASIA-classifier::train')
    logger.info('--- Running CASIA-classifier Training ---')

    # train arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', '-lrs', type=float, default=1e-2, help='learning rate for the network')
    parser.add_argument('--momentum1', '-mom1', type=float, default=0.9, help='The beta1 for Adam. Default: 0.9')
    parser.add_argument('--momentum2', '-mom2', type=float, default=0.999, help='The beta2 for Adam. Default: 0.999')
    parser.add_argument('--eps', type=float, default=1e-8, help='The epsilon for Adam. Default: 1e-8')
    parser.add_argument('--display', type=int, default=500, help='The number of iterations after which to display \
                        the loss values. Default: 500')
    parser.add_argument('--max_epoch', '-epoch', type=int, default=30, help='The number of max epoch while training. \
                        Default: 100')
    parser.add_argument('--batch_size', '-is', type=int, default=128, help='The batch size of the data. Default: 128')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.00005, help='The weight decay for Adam \
                        Default: 0.00005')
    parser.add_argument('--gpu', type=ast.literal_eval, default=True, help='Whether to use gpu or not')
    parser.add_argument('--pretrained', action='store_true', default=False, help='Flag for whether loading weight from \
                        model trained on Imagenet')
    parser.add_argument('--model_path', default="", help="path to the saved model")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, using CPU mode')
        args.gpu = False

    # print out the used arguments
    logger.info('###########################################')
    logger.info('Experiment Parameters:')
    for key, value in vars(args).items():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')

    # load dataset
    logger.info('Loading dataset CASIA')

    train_image_dir = "/home/dengbowen/character/images/train"
    train_label_dir = "/home/dengbowen/character/labels/train"
    test_image_dir = "/home/dengbowen/cmp/images"
    test_label_dir = "/home/dengbowen/cmp/labels"
    label_files = ['./datasets/Char4037-list.txt', './datasets/Char3319-list.txt']
    char_class = 7356
    train_set = CASIA(train_image_dir, train_label_dir, label_files, char_class)
    test_set = CASIA(test_image_dir, test_label_dir, label_files, char_class)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # make directory for saving models
    if not os.path.isdir('./output'):
        os.mkdir('./output')
        logger.info("Make output directory")
    else:
        logger.info("Output directory already exists")

    # initialize network, optimizer, scheduler, loss and writer
    cnn = AlexNet(char_class)
    optimizer = torch.optim.Adam(cnn.parameters(), args.learning_rate, (args.momentum1, args.momentum2), args.eps,
                                 args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    loss = nn.CrossEntropyLoss()
    model_name = cnn.get_class_name()
    writer = SummaryWriter(comment=datetime.datetime.now().strftime("%m-%d %H:%M:%S") + model_name)

    # load weight and optimizer state from saved model
    if args.pretrained:
        if os.path.isfile(args.model_path):
            logger.info("Loading checkpoint from {:s}".format(args.model_path))
            checkpoint = torch.load(args.model_path)
            cnn.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        cnn.apply(init_net)

    # move to GPU if specified
    device = torch.device("cuda:0" if args.gpu else "cpu")
    if args.gpu and torch.cuda.device_count() > 1:
        cnn.apply_parallel()
    cnn.to(device)
    # count parameters
    logger.info("{:s} has {} trainable parameters".format(model_name, count_parameter(cnn)))
    # run training
    logger.info('Training {:s}'.format(model_name))
    # initialize the iteration counter and the threshold for save a model
    iteration = 0
    threshold = 80
    for epoch in range(args.max_epoch):
        scheduler.step()
        type(cnn)
        epoch_start = time.time()
        best_accuracy = 0.0
        logger.info('Learning rate:{}'.format(optimizer.defaults['lr']))
        for batch_id, (image, label) in enumerate(train_loader):
            iteration += 1
            if (batch_id + 1) % args.display == 0:
                logger.info("Test at epoch{}, batch{}".format(epoch, batch_id + 1))
                top1, top5 = evaluate_cnn(cnn, test_loader, device)
                writer.add_scalar('top1', top1, iteration)
                writer.add_scalar('top5', top5, iteration)
                logger.info("Top1 accuracy:{}%, Top5 accuracy:{}% at epoch{}, batch{}".format(top1, top5, epoch,
                                                                                              batch_id + 1))
                if top1 >= threshold and top1 > best_accuracy:
                    state = {'state_dict': cnn.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, os.path.join('./output', datetime.datetime.now().strftime("%m-%d %H:%M:%S") +
                                                   '_{:s}_{:.2f}.pth.tar'.format(model_name, top1)))
                    best_accuracy = top1
            batch_start = time.time()
            image = image.to(device)
            label = label.to(device)
            output = cnn(image)
            loss_val = loss(output, label)
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_end = time.time()
            writer.add_scalar('loss', loss_val.item(), iteration)
            logger.info("batch{} loss: {} ".format(batch_id, loss_val.item()))
            logger.info("batch{} takes {:.2f}s".format(batch_id, batch_end - batch_start))
        epoch_end = time.time()
        logger.info("epoch{} takes {:2d}min {:.2f}s".format(epoch, int(epoch_end - epoch_start) // 60,
                                                            (epoch_end - epoch_start) % 60))
        logger.info('{:s} training finished'.format(model_name))


def evaluate_cnn(cnn, dataset_loader, device):
    # set the CNN in eval mode
    cnn.eval()
    top1 = 0.0
    top5 = 0.0
    total = 0
    with torch.no_grad():
        for sample_idx, (image, label) in enumerate(tqdm.tqdm(dataset_loader)):
            total += label.size()[0]
            image = image.to(device)
            label = label.to(device).view(1, -1)
            outputs = cnn(image)
            _, predicts = outputs.topk(k=5, dim=1)
            predicts = predicts.t()
            correct = predicts.eq(label.expand_as(predicts))
            top1 += correct[:1].view(-1).float().sum(0, keepdim=True).item()
            top5 += correct[:5].view(-1).float().sum(0, keepdim=True).item()
    # set the CNN in train model
    cnn.train()
    return 100 * top1 / total, 100 * top5 / total


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    train()
