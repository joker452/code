import os
import cv2
import sys
import json
import pickle
import untangle
import numpy as np

sys.path.append('./')
from util import mkdir, compare
from functools import cmp_to_key


def write_char(f, position, class_id, next_char_class, nnext_char_class):
    for pos in position:
        f.write(str(pos) + " ")
    f.write(str(class_id) + " ")
    f.write(str(next_char_class) + " " + str(nnext_char_class) + "\n")


def make_dataset(root_dir):
    not_wanted = ('╳', '阝', '═', '︺', '︹', '━', '', '│', '□', '○', '。', '、', '\u3000', '\ue002')
    data_dir = [f.name for f in os.scandir(root_dir) if f.is_dir() and f.name != 'out']
    page_id = 1
    out_dir = os.path.join(root_dir, "out")
    train_dir = os.path.join(out_dir, "train")
    test_dir = os.path.join(out_dir, "test")
    mkdir(out_dir)
    mkdir(train_dir)
    mkdir(test_dir)
    with open('./difangzhi_freq.json', 'r', encoding='utf-8') as f:
        d1 = json.load(f)
    d2 = d1.copy()
    for k in d2.keys():
        d2[k] = d2[k] // 2
        if k not in not_wanted:
            mkdir(os.path.join(test_dir, k))
            mkdir(os.path.join(train_dir, k))
    with open('./char2index.pkl', 'rb') as f:
        index_dict = pickle.load(f)

    counter = 0
    for data in data_dir:
        img_dir = os.path.join(root_dir, data, "jpg")
        xml_dir = os.path.join(root_dir, data, "xml")
        doc_list = os.listdir(xml_dir)
        for i in range(len(doc_list)):
            doc = untangle.parse(os.path.join(xml_dir, doc_list[i]))
            page_name = doc.page['id'].replace("xml", "jpg")
            if hasattr(doc.page, 'text'):
                positions = [text['image_position'] for text in doc.page.text]
                characters = [text.cdata for text in doc.page.text]
            else:
                # for images containing multi-text
                characters = []
                positions = []
                if hasattr(doc.page, 'text_line'):
                    for text_line in doc.page.text_line:
                        pos1 = pos2 = char1 = char2 = None
                        if hasattr(text_line, 'text'):
                            pos1 = [text['image_position'] for text in text_line.text if text['image_position']]
                            char1 = [text.cdata for text in text_line.text if text['image_position']]
                        if hasattr(text_line, 'multi_text'):
                            pos2 = [text['image_position'] for multi_text in text_line.multi_text
                                    for text in multi_text.text if text['image_position']]
                            char2 = [text.cdata for multi_text in text_line.multi_text
                                     for text in multi_text.text if text['image_position']]
                        if pos1:
                            if pos2:
                                positions += pos1 + pos2
                                characters += char1 + char2
                            else:
                                positions += pos1
                                characters += char1
                        elif pos2:
                            positions += pos2
                            characters += char2

            labels = []
            for k in range(len(characters)):
                positions[k] = positions[k].split(';')
                characters[k] = characters[k].replace('\n', '')
                for char_index in range(len(characters[k])):
                    # 去掉无用字符
                    char = characters[k][char_index]
                    if char not in not_wanted:
                        coordinates = list(map(int, positions[k][char_index].split(',')))
                        labels.append({'text': char, 'coordinates': coordinates})

            labels.sort(key=cmp_to_key(compare))
            img = cv2.imread(os.path.join(img_dir, page_name), -1)
            cv2.imwrite(os.path.join(out_dir, '{}.jpg'.format(page_id)), img)
            img = cv2.morphologyEx(img, 6, cv2.getStructuringElement(1, (33, 33), (16, 16)))
            img = img.mean(2)[:, :, None]
            img = np.where(img < 50, (0, 0, 0), (255, 255, 255)).astype(np.uint8)
            for k in range(len(labels)):
                position = labels[k]['coordinates']
                label = labels[k]['text']
                char = img[position[1]: position[3], position[0]: position[2]]
                file_name = '{}-{}-{}.jpg'.format(data, page_name[: -4], counter)
                counter += 1
                if d1[label] != 1 and d2[label] > 0:
                    d2[label] = d2[label] - 1
                    # put it in train
                    cv2.imwrite(os.path.join(train_dir, label, file_name), char)
                else:
                    cv2.imwrite(os.path.join(test_dir, label, file_name), char)

            with open(os.path.join(out_dir, "{}.txt".format(page_id)), "a", encoding='utf-8') as f:
                char_num = len(labels)
                for k in range(char_num - 2):
                    position = labels[k]['coordinates']
                    class_id = index_dict[labels[k]['text']]
                    next_class_id = index_dict[labels[k + 1]['text']]
                    nnext_class_id = index_dict[labels[k + 2]['text']]
                    write_char(f, position, class_id, next_class_id, nnext_class_id)
                # -2
                write_char(f, labels[char_num - 2]['coordinates'],
                           index_dict[labels[char_num - 2]['text']],
                           index_dict[labels[char_num - 1]['text']], -1)
                # -1
                write_char(f, labels[char_num - 1]['coordinates'],
                           index_dict[labels[char_num - 1]['text']], -1, -1)
            page_id += 1


if __name__ == '__main__':
    root_dir = "d:/project/lunwen/data/difangzhi"
    make_dataset(root_dir)
