import os
import ast
import tqdm
import logging
import argparse
import torch.autograd
import torch.cuda
import datetime
import torch.nn as nn
import torch.optim
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision
from tensorboardX import SummaryWriter
from torchvision.transforms import transforms
from models import resnet
import torch.backends.cudnn as cudnn


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
    parser.add_argument('--display', type=int, default=200, help='The number of iterations after which to display \
                        the loss values. Default: 500')
    parser.add_argument('--batch_size', '-is', type=int, default=64, help='The batch size of the data. Default: 128')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.00005, help='The weight decay for Adam \
                        Default: 0.00005')
    parser.add_argument('--gpu', type=ast.literal_eval, default=True, help='Whether to use gpu or not')

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
    char_class = 3755
    train_set = torchvision.datasets.ImageFolder("/data1/yg/chinese/cas_jpg96/train",
                                                 transforms.Compose([transforms.Resize((224, 224)),
                                                                     transforms.ToTensor()]))
    test_set = torchvision.datasets.ImageFolder("/data1/yg/chinese/cas_icdar_jpg96/",
                                                transforms.Compose([transforms.Resize((224, 224)),
                                                                    transforms.ToTensor()]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    arch = '50'
    nl_type = 'cgnl'
    nl_nums = 0
    pool_size = 7
    cnn = resnet.model_hub(arch,
                           pretrained=False,
                           nl_type=nl_type,
                           nl_nums=nl_nums,
                           pool_size=pool_size)

    # change the fc layer
    cnn._modules['fc'] = torch.nn.Linear(in_features=2048,
                                         out_features=char_class)

    torch.nn.init.kaiming_normal_(cnn._modules['fc'].weight,
                                  mode='fan_out', nonlinearity='relu')
    print(cnn)

    # optimizer
    optimizer = torch.optim.SGD(
        cnn.parameters(),
        args.learning_rate,
        momentum=0.9,
        weight_decay=1e-4)
    writer = SummaryWriter(log_dir='runs/' + datetime.datetime.now().strftime("%m-%d-%H-%M") +
                                   optimizer.__class__.__name__ + str(args.learning_rate))

    # cudnn
    cudnn.benchmark = True
    # make directory for saving models
    if not os.path.isdir('./output/pretrain'):
        os.makedirs('./output/pretrain')
        logger.info("Make output directory")
    else:
        logger.info("Output directory already exists")

    # initialize network, optimizer, scheduler, loss and writer

    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    loss = nn.CrossEntropyLoss().cuda()
    # move to GPU if specified
    device = torch.device("cuda:0" if args.gpu else "cpu")
    if torch.cuda.device_count() > 1:
        cnn = torch.nn.DataParallel(cnn).to(device)
    else:
        cnn = cnn.to(device)
    # initialize the iteration counter and the threshold for save a model
    model_name = 'ResNet'
    start_time = datetime.datetime.now().strftime("%m-%d-%H-%M")
    iteration = 0
    threshold = 80
    while iteration < 9000:
        scheduler.step()
        best_accuracy = 0.0
        logger.info('Learning rate:{}'.format(optimizer.defaults['lr']))

        for batch_id, (image, label) in enumerate(train_loader):
            iteration += 1
            if (batch_id + 1) % args.display == 0:
                logger.info("Test at epoch{}, batch{}".format(epoch, batch_id + 1))
                top1, top5 = evaluate_cnn(cnn, test_loader, device)
                logger.info("Top1 accuracy:{}%, Top5 accuracy:{}% at epoch{}, batch{}".format(top1, top5, epoch,
                                                                                              batch_id + 1))
                writer.add_scalar('test/top1', top1, iteration)
                writer.add_scalar('test/top5', top5, iteration)
                if top1 >= threshold and top1 > best_accuracy:
                    state = {'state_dict': cnn.state_dict()}
                    torch.save(state,
                               os.path.join('./output', '_{:s}_{:.2f}.pth.tar'.format(model_name + start_time, top1)))
                    best_accuracy = top1
            image = image.to(device)
            label = label.to(device)
            output = cnn(image)
            loss_val = loss(output, label)
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad()
            out = torch.cat((output.data, torch.ones(len(output), 1, device=device)), 1)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iteration)
            writer.add_scalar('train/loss', loss_val.item(), iteration)

            logger.info("batch{} loss: {} ".format(batch_id, loss_val.item()))


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
