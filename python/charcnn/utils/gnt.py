import os
import numpy as np
from PIL import Image


def mkdir(dir_name):
    if not os.path.isdir(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError:
            print('Can not make directory for {}'.format(dir_name))
            raise OSError
        else:
            print("Make directory for {}".format(dir_name))
    else:
        print("{} already exists".format(dir_name))


data_dir = "/home/dengbowen/offline"
# for competition
cmp = True
# for HWDB1.0
train_data_file = "/home/dengbowen/offline/CASIA-HWDB1.0/trainlist.txt"
test_data_file = "/home/dengbowen/offline/CASIA-HWDB1.0/testlist.txt"
# for HWDB1.1 and HWDB1.2
train_list = (501, 740)
test_list = (741, 800)
if train_data_file != "":
    with open(train_data_file, 'r') as f:
        train_file_names = f.read().split('\n')
else:
    train_file_names = [str(i) for i in range(train_list[0], train_list[1] + 1)]
if test_data_file != "":
    with open(test_data_file, 'r') as f:
        test_file_names = f.read().split('\n')
else:
    test_file_names = [str(i) for i in range(test_list[0], test_list[1] + 1)]
images_train_dir = '.' + os.sep + 'character' + os.sep + 'images' + os.sep + 'train'
images_test_dir = '.' + os.sep + 'character' + os.sep + 'images' + os.sep + 'test'
labels_train_dir = '.' + os.sep + 'character' + os.sep + 'labels' + os.sep + 'train'
labels_test_dir = '.' + os.sep + 'character' + os.sep + 'labels' + os.sep + 'test'

if cmp:
    train_file_names = [file_name[: -6] for file_name in os.listdir(data_dir) if file_name.endswith('.gnt')]
    test_file_names = []
    images_train_dir = '.' + os.sep + 'cmp' + os.sep + 'images'
    labels_train_dir = '.' + os.sep + 'cmp' + os.sep + 'labels'
mkdir(images_train_dir)
mkdir(images_test_dir)
mkdir(labels_train_dir)
mkdir(labels_test_dir)
for file_name in train_file_names:
    with open(os.path.join(data_dir, file_name + '-f.gnt'), 'rb') as f:
        counter = 0
        while True:
            header_size = 10
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size:
                break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tag_code = str(hex(header[5] + (header[4] << 8)))
            tag_code = tag_code[0:2] + tag_code[-2:] + tag_code[2: 4]
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                print('size mismatch')
                break
            counter = counter + 1
            image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
            im = Image.fromarray(image)
            im.convert('L').save(os.path.join(images_train_dir, file_name + '-' + str(counter) + '.png'))
            with open(os.path.join(labels_train_dir, file_name + '-' + str(counter) + '.txt'), 'w',
                      encoding='utf-8') as w:
                w.write(tag_code)
for file_name in test_file_names:
    with open(os.path.join(data_dir, file_name + '-f.gnt'), 'rb') as f:
        counter = 0
        while True:
            header_size = 10
            header = np.fromfile(f, dtype='uint8', count=header_size)
            if not header.size:
                break
            sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
            tag_code = str(hex(header[5] + (header[4] << 8)))
            tag_code = tag_code[0:2] + tag_code[-2:] + tag_code[2: 4]
            width = header[6] + (header[7] << 8)
            height = header[8] + (header[9] << 8)
            if header_size + width * height != sample_size:
                print('size mismatch')
                break
            if header_size != sample_size:
                counter = counter + 1
                image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                im = Image.fromarray(image)
                try:
                    im.convert('L').save(os.path.join(images_test_dir, file_name + '-' + str(counter) + '.png'))
                except SystemError:
                    print(file_name, counter)
                with open(os.path.join(labels_test_dir, file_name + '-' + str(counter) + '.txt'), 'w',
                          encoding='utf-8') as w:
                    w.write(tag_code)
