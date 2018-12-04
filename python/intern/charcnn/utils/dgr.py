import os
import pickle
import numpy as np
import logging
from PIL import Image

dgr_dir = '/home/dengbowen/offline'
dir_length = len(dgr_dir)
file_names = [files for files in os.listdir(dgr_dir) if files.endswith('dgr')]


def bytes2int(x):
    result = 0
    for index, byte in enumerate(x):
        result = result + (byte << (index * 8))
    return result
cmp = False
# different for HWDB2.0-2.2 and cmp
images_dir = '/home/dengbowen' + os.sep + 'text_cmp' + os.sep + 'images' if cmp  else '/home/dengbowen' + os.sep + 'text' + os.sep + 'images'
labels_dir = '/home/dengbowen' + os.sep + 'text_cmp' + os.sep + 'labels' if cmp else '/home/dengbowen' + os.sep + 'text' + os.sep + 'labels'
print(images_dir)
logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M%S', level=logging.INFO)
logger = logging.getLogger('Data prepare')
if not os.path.exists(images_dir):
    try:
        os.makedirs(images_dir)
    except OSError:
        logger.warning('Can not make images directory!')
        raise OSError
    else:
        logger.info("Make images directory")
else:
    logger.info("images directory already exists")
if not os.path.exists(labels_dir):
    try:
        os.makedirs(labels_dir)
    except OSError:
        logger.warning('Can not make labels directory')
        raise OSError
    else:
        logger.info("Make images directory")
else:
    logger.info("labels dir already exists")
freq_dict = {}
for file in file_names:
    with open(os.path.join(dgr_dir, file), 'rb') as f:
        file_name = file.split('.')[0]
        header = np.fromfile(f, dtype='uint8', count=4)
        header_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        # jump over format code and illustration
        np.fromfile(f, dtype='uint8', count=header_size - 28)
        code_type = bytes(np.fromfile(f, dtype='uint8', count=20)).decode()
        if code_type[:2] == 'GB':
            code_type = 'gbk'
        else:
            print('Code type{} in file {}'.format(code_type, file))
        code_length = bytes2int(np.fromfile(f, dtype='uint8', count=2))
        bits_per_pixel = bytes2int(np.fromfile(f, dtype='uint8', count=2))
        if bits_per_pixel != 8:
            logger.warning('Binary image in file {}'.format(file))
        image_h = bytes2int(np.fromfile(f, dtype='uint8', count=4))
        image_w = bytes2int(np.fromfile(f, dtype='uint8', count=4))
        line_num = bytes2int(np.fromfile(f, dtype='uint8', count=4))
        text_image = np.full((image_h, image_w), 255, dtype='uint8')
        with open(os.path.join(labels_dir, file_name) + '.txt', 'w', encoding='utf-8') as w:
            for line in range(line_num):
                chars_num = bytes2int(np.fromfile(f, dtype='uint8', count=4))
                for chars in range(chars_num):
                    char_code = np.fromfile(f, dtype='uint8', count=code_length)
                    top = bytes2int(np.fromfile(f, dtype='uint8', count=2))
                    left = bytes2int(np.fromfile(f, dtype='uint8', count=2))
                    char_h = bytes2int(np.fromfile(f, dtype='uint8', count=2))
                    char_w = bytes2int(np.fromfile(f, dtype='uint8', count=2))
                    try:
                        if char_code[0] != 255:
                            label = hex(bytes(char_code)[1]) + hex(bytes(char_code)[0])[2: ]
                            if label not in freq_dict.keys():
                                freq_dict[label] = 1
                            else:
                                freq_dict[label] += 1
                            w.write(label + ' ' + str(top) + ' ' + str(left) + ' ' + str(char_h) + ' ' + str(char_w) + ' ')
                    except UnicodeDecodeError:
                        logger.warning(file_name, line, chars, tmp)
                    if bits_per_pixel == 1:
                        char = np.fromfile(f, dtype='uint8', count=char_h * ((char_w + 7) / 8)).reshape(char_h, (
                                char_w + 7) / 8)
                    else:
                        char = np.fromfile(f, dtype='uint8', count=char_h * char_w).reshape(char_h, char_w)
                    try:
                      text_image[top: top + char_h, left: left + char_w] = char
                      if top + char_h >= image_h or left + char_w >= image_w:
                          # some annotation may be out of boundary
                          logger.warning("File{:s}, line{:d}, char{:d}".format(file_name, line, chars))
                    except ValueError:
                      logger.warning("File{:s}, line{:d}, char{:d}".format(file_name, line, chars))
                w.write('\n')
        im = Image.fromarray(text_image)

        im.convert('L').save(os.path.join(images_dir, file_name + '.png'))
with open('./freq.pkl', 'wb') as f:
    pickle.dump(freq_dict, f)
