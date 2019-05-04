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
            if not img:
                print(os.path.join(img_dir, file[: -4] + '.jpg'))
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
                char_num = struct.unpack("<I", f.read(4))[0]
                if len(old_label) != char_num:
                    print(len(old_label), char_num, file[: -4], "size mismatch")
                for i in range(char_num):
                    total += 1
                    left = struct.unpack("<I", f.read(4))[0]
                    top = struct.unpack("<I", f.read(4))[0]
                    right = struct.unpack("<I", f.read(4))[0]
                    bottom = struct.unpack("<I", f.read(4))[0]
                    # skip iAttribute and iField
                    f.read(8)

                    char = img[top: bottom, left: right]
                    file_name = "{}-{}.jpg".format(file[: -4], i)
                    cv2.imwrite(os.path.join(out_dir, file_name), char)
                    try:
                        labels.append(os.path.join(out_dir, file_name) + " " + old_label[i] + "\n")
                    except:
                        print(file[: -4], "out of index")

            with open(os.path.join(out_dir, "labels.txt"), 'w', encoding='utf-8') as f:
                f.writelines(labels)


if __name__ == '__main__':
    make_dataset('/mnt/data1/dengbowen/dunhuang')
