import numpy as np
from PIL import Image
import os
import struct


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


def dgr2image(data_dir, file_names, image_dir, label_dir):
    code_types = set()
    code_lengths = set()
    for file_name in file_names:
        with open(os.path.join(data_dir, file_name), 'rb') as f:
            header_size = struct.unpack("<I", np.fromfile(f, dtype='uint8', count=4))[0]
            # skip format code and illustration
            np.fromfile(f, dtype='uint8', count=header_size-28)
            code_type = hex(np.fromfile(f, dtype='uint8', count=20)[0])
            code_length = struct.unpack("<H", np.fromfile(f, dtype='uint8', count=2))[0]
            bits_per_pixel = struct.unpack("<H", np.fromfile(f, dtype='uint8', count=2))[0]
            if code_type not in code_types:
                code_types.add(code_type)
            if code_length not in code_lengths:
                code_lengths.add(code_length)
            if bits_per_pixel != 8:
                print('Binary image in file {}'.format(file_name))
            height = struct.unpack("<I", np.fromfile(f, dtype='uint8', count=4))[0]
            width = struct.unpack("<I", np.fromfile(f, dtype='uint8', count=4))[0]
            line_number = struct.unpack("<I", np.fromfile(f, dtype='uint8', count=4))[0]
            text_image = np.full((height, width), 255, dtype='uint8')
            with open(os.path.join(label_dir, file_name) + '.txt', 'w', encoding='utf-8') as w:
                for line in range(line_number):
                    chars_num = struct.unpack("<I", np.fromfile(f, dtype='uint8', count=4))[0]
                    for chars in range(chars_num):
                        if code_length == 1:
                            label = hex(struct.unpack("<B", np.fromfile(f, dtype='uint8', count=code_length))[0])
                        elif code_length == 2:
                            label = hex(struct.unpack("<H", np.fromfile(f, dtype='uint8', count=code_length))[0])
                        else:
                            label = hex(struct.unpack("<I", np.fromfile(f, dtype='uint8', count=code_length))[0])
                        top = struct.unpack("<H", np.fromfile(f, dtype='uint8', count=2))[0]
                        left = struct.unpack("<H", np.fromfile(f, dtype='uint8', count=2))[0]
                        char_h = struct.unpack("<H", np.fromfile(f, dtype='uint8', count=2))[0]
                        char_w = struct.unpack("<H", np.fromfile(f, dtype='uint8', count=2))[0]
                        try:
                            w.write(label + ' ' + str(top) + ' ' + str(left) + ' ' + str(char_h) + ' ' + str(char_w) + ' ')
                        except UnicodeDecodeError:
                            print(file_name, line, chars)
                        if bits_per_pixel == 1:
                            char = np.fromfile(f, 'uint8', char_h * ((char_w + 7) / 8)).reshape(char_h,
                                                (char_w + 7) / 8)
                        else:
                            char = np.fromfile(f, 'uint8', char_h * char_w).reshape(char_h, char_w)
                        try:
                          text_image[top: top + char_h, left: left + char_w] = char
                          if top + char_h >= height or left + char_w >= width:
                              # some annotation may be out of boundary
                              print("File{:s}, line{:d}, char{:d}".format(file_name, line, chars))
                        except ValueError:
                            print("File{:s}, line{:d}, char{:d}".format(file_name, line, chars))
                    w.write('\n')
            im = Image.fromarray(text_image)
            im.convert('L').save(os.path.join(image_dir, file_name + '.png'))
    print(code_types)
    print(code_lengths)


if __name__ == '__main__':
    data_dir = '/data2/dengbowen/data'
    file_names = [files for files in os.listdir(data_dir) if files.endswith('dgr')]
    file_names.sort()
    image_dir = '/data2/dengbowen/document/cmp/images'
    label_dir = '/data2/dengbowen/document/cmp/labels'
    mkdir(image_dir)
    mkdir(label_dir)
    dgr2image(data_dir, file_names, image_dir, label_dir)
