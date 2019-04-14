import os
import cv2
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


def make_dataset(root_dir):
    total = 0
    data_dir = os.listdir(root_dir)
    for data in data_dir:
        out_dir = os.path.join(root_dir, data, "out")
        mkdir(out_dir)
        mkdir(os.path.join(out_dir, "images"))
        rgn_dir = os.path.join(root_dir, data, "work")
        img_dir = os.path.join(root_dir, data, "img")
        rgn = os.listdir(rgn_dir)
        for file in rgn:
            img = cv2.imread(os.path.join(img_dir, file[: -4] + '.jpg'), -1)
            if img is None:
                print(os.path.join(img_dir, file[: -4] + '.jpg'))
                continue
            labels = []
            try:
                with open(os.path.join(img_dir, file[: -4] + '.txt'), 'r', encoding='utf-16-le') as f:
                    old_label = f.read()
                    old_label = old_label.replace(chr(65279), '')
                    old_label = old_label.replace(' ', '')
                    old_label = old_label.replace('\n', '')
            except:
                print(file[: -4], "not exist")
            with open(os.path.join(rgn_dir, file), 'rb') as f:
                # better way to check?
                #debug_time = 0
                #left_debug = []
                #right_debug = []
                char_start =  columns = char_in_column = 0
                left_most = 100000
                right_most = -1
                top_most = 100000
                bottom_most = -1
                prev_left = prev_right = None
                char_num = struct.unpack("<I", f.read(4))[0]
                if len(old_label) != char_num:
                    print(len(old_label), char_num, file[: -4], "size mismatch")
                for i in range(char_num):
                    total += 1
                    left = struct.unpack("<I", f.read(4))[0]
                    top = struct.unpack("<I", f.read(4))[0]
                    right = struct.unpack("<I", f.read(4))[0]
                    bottom = struct.unpack("<I", f.read(4))[0]
                    if prev_left is None:
                        prev_left = left
                    if prev_right is None:
                        prev_right = right
                    # skip iAttribute and iField
                    # left_debug.append(left)
                    # right_debug.append(right)
                    # debug_time += 1
                    # if debug_time >= 100:
                    #     print(left_debug)
                    #     print(right_debug)
                    #     exit(0)
                    f.read(8)
                    if prev_left > left + 200 and prev_right > right + 200:
                        char_column = img[top_most: bottom_most, left_most: right_most]
                        right_most = right
                        columns += 1
                        file_name = "{}-{}.jpg".format(file[: -4], columns)
                        cv2.imwrite(os.path.join(out_dir, file_name), char_column)
                        try:
                            labels.append(os.path.join(out_dir, file_name) + " " + old_label[char_start: char_start + char_in_column] + "\n")
                        except:
                            print(file[: -4], "out of index")
                        char_start += char_in_column
                        char_in_column = 0
                    prev_left = left
                    prev_right = right
                    char_in_column += 1
                    if left_most > left:
                        left_most = left
                    if right_most < right:
                        right_most = right
                    if top_most > top:
                        top_most = top
                    if bottom_most < bottom:
                        bottom_most = bottom
                if (char_start <= char_num):
                    char_column = img[top_most: bottom_most, left_most: right_most]
                    columns += 1
                    file_name = "{}-{}.jpg".format(file[: -4], columns)
                    cv2.imwrite(os.path.join(out_dir, file_name), char_column)
                    try:
                        labels.append(os.path.join(out_dir, file_name) + " " + old_label[char_start: char_start + char_in_column] + "\n")
                    except:
                        print(file[: -4], "out of index")
                print(columns)
            with open(os.path.join(out_dir, "labels.txt"), 'a', encoding='utf-8') as f:
                f.writelines(labels)


if __name__ == '__main__':
    make_dataset('d:/lunwen/data/dunhuang/')
