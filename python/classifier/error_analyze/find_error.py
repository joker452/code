import torch
import pickle
import torchvision.transforms as tr
from models.charcnn import CharCnn
from datasets.casia_charcnn import CASIA
from torch.utils.data.dataloader import DataLoader

with open('../utils/index2char.pkl', 'rb') as f:
    d = pickle.load(f)

def evaluate_cnn(cnn, dataset_loader, device):
    # set the CNN in eval mode
    cnn.eval()
    error = 0
    with torch.no_grad():
        for sample_idx, (image, label) in enumerate(dataset_loader):
            if error > 100:
                break
            image = image.to(device)
            label = label.to(device).view(1, -1)
            outputs = cnn(image)
            _, predicts = outputs.topk(k=5, dim=1)
            correct = predicts.eq(label.expand_as(predicts))

            # if top1 is not correct
            if correct[:1].view(-1).float().sum(0, keepdim=True).item() < 1:
                error += 1
                image.squeeze_()
                image.unsqueeze_(0)
                prediction = list(map(lambda x: bytes.fromhex(d[str(x)][2: ]).decode('gbk'),
                                               predicts.cpu().numpy()[0]))
                print("Prediction: ", prediction)
                gt = bytes.fromhex(d[str(label.cpu().numpy()[0, 0])][2: ]).decode('gbk')
                print("GT: ", gt)
                tr.ToPILImage()(image.cpu()).save('/home/dengbowen/images/{}-{}.png'.format(prediction, gt))


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('../output/CharCnn.pth.tar')
char_class = 7356
cnn = CharCnn(char_class)
cnn.apply_parallel()
cnn.load_state_dict(checkpoint['state_dict'])
cnn.to(device)
test_image_dir = "/mnt/data1/dengbowen/cmp/images"
test_label_dir = "/mnt/data1/dengbowen/cmp/labels"
label_files = ['/mnt/data1/dengbowen/code/python/charcnn/datasets/Char4037-list.txt',
               '/mnt/data1/dengbowen/code/python/charcnn/datasets/Char3319-list.txt']
test_set = CASIA(test_image_dir, test_label_dir, label_files, char_class)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
evaluate_cnn(cnn, test_loader, device)
