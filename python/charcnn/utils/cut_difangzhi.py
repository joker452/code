import os
import cv2
from PIL import Image
import json
import untangle


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
    not_wanted = ('╳', '阝', '═', '︺', '︹', '━', '', '│', '□', '○', '。', '、', '\u3000', '\ue002')
    with open('./difangzhi_freq.json', 'r') as f:
        d1 = json.load(f)
    d2 = d1.copy()
    test_dir = './test'
    train_dir = './train'
    mkdir(test_dir)
    mkdir(train_dir)
    for k in d2.keys():
        d2[k] = d2[k] // 2
        if k not in not_wanted:
            mkdir(os.path.join(test_dir, k))
            mkdir(os.path.join(train_dir, k))
    prefix = '/data2/dengbowen/work/samples/difangzhi'
    total = 0
    train = 0
    test = 0
    data_dir = [f.name for f in os.scandir(root_dir) if f.is_dir() and f.name != 'data']

    # f1 = open('train.txt', 'w', encoding='utf-8')
    # f2 = open('test.txt', 'w', encoding='utf-8')
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
                    pos1 = pos2 = char1 = char2 = None
                    for text_line in doc.page.text_line:
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

            img = cv2.imread(os.path.join(img_dir, page_name), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for k in range(len(labels)):
                position = labels[k]['coordinates']
                label = labels[k]['text']
                char = img[position[1]: position[3], position[0]: position[2]]
                total += 1
                file_name = '{}-{}-{}.jpg'.format(data, page_name[: -4], k)
                im = Image.fromarray(char)
                if d1[label] != 1 and d2[label] > 0:
                    d2[label] = d2[label] - 1
                    # put it in train
                    train += 1
                    path = os.path.join(train_dir, label, file_name)
                    if not os.path.exists(path):
                        im.save(os.path.join(train_dir, label, file_name))
                    #path = os.path.join(prefix, 'train', label, file_name).replace('\\', '/')
                    # f1.write(path + " " + label + "\n")
                else:
                    test += 1
                    path = os.path.join(test_dir, label, file_name)
                    if not os.path.exists(path):
                        im.save(os.path.join(test_dir, label, file_name))
                    #path = os.path.join(prefix, 'test', label, file_name).replace('\\', '/')
                    # f2.write(path + " " + label + "\n")
    #
    # f1.close()
    # f2.close()
    print(total)
    print(train)
    print(test)


if __name__ == '__main__':
    root_dir = "d:/lunwen/data/difangzhi"
    make_dataset(root_dir)
