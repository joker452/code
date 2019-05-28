import os
import sys
import pickle
import struct
import numpy as np
from PIL import Image

sys.path.append('./')
from util import mkdir


def gnt2image(data_dir, file_numbers, image_dir, label_dir, is_only3755, d):
    """
    :param str data_dir: directory containing all the gnt files
    :param list file_numbers: all of the writer indexes
    :param str image_dir: output directory for images
    :param str label_dir: output directory for labels
    :param bool is_only3755: is True, only generates images for charaters in GB2312-80 level 1 set
    :param dict d: GB2312-80 set 1 dictionary
    :return None
    """
    for file_number in file_numbers:
        with open(os.path.join(data_dir, file_number + '-c.gnt'), 'rb') as f:
            counter = 0
            while True:
                header_size = 10
                header = np.fromfile(f, dtype='uint8', count=header_size)
                if not header.size:
                    break
                sample_size = struct.unpack("<I", header[:4])[0]
                # in consistence with label, note symbols are stored in little endian
                tag_code = hex(struct.unpack(">H", header[4: 6])[0])
                width = struct.unpack("<H", header[6: 8])[0]
                height = struct.unpack("<H", header[8:])[0]
                if is_only3755:
                    if tag_code not in d:
                        # pass the data for this image
                        np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                        continue
                counter = counter + 1
                image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                im = Image.fromarray(image)
                im.convert('L').save(os.path.join(image_dir, file_number + '-' + str(counter) + '.png'))
                with open(os.path.join(label_dir, file_number + '-' + str(counter) + '.txt'), 'w',
                          encoding='utf-8') as w:
                    w.write(tag_code)


if __name__ == '__main__':
    with open('./gbcode3755.pkl', 'rb') as f:
        d = pickle.load(f)
    data_dir = '/data2/dengbowen/1.1'
    number_file = './1.1trainlist.txt'
    with open(number_file, 'r') as f:
        file_numbers = f.read().split('\n')
    image_dir = '/data2/dengbowen/character/3755images'
    label_dir = '/data2/dengbowen/character/3755labels'
    mkdir(image_dir)
    mkdir(label_dir)
    gnt2image(data_dir, file_numbers, image_dir, label_dir, True, d)
