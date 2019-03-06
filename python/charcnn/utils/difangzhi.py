import os
import cv2
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
    total = 0
    data_dir = [f.name for f in os.scandir(root_dir) if f.is_dir() and f.name != 'images']
    char_h_max = char_w_max = 0
    char_h_min = char_w_min = 10000
    for data in data_dir:
        out_dir = os.path.join(root_dir, data, "out")
        mkdir(out_dir)
        mkdir(os.path.join(out_dir, "images"))
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
                    if char != '\u3000' and char != '\ue002':
                        coordinates = list(map(int, positions[k][char_index].split(',')))
                        labels.append({'text': char, 'coordinates': coordinates})

            img = cv2.imread(os.path.join(img_dir, page_name), -1)
            with open(os.path.join(out_dir, "labels.txt"), "a", encoding='utf-8') as f:
                for k in range(len(labels)):
                    total += 1
                    position = labels[k]['coordinates']
                    char = img[position[1]: position[3], position[0]: position[2]]
                    h = position[3] - position[1]
                    w = position[2] - position[0]
                    if char_h_max < h:
                        char_h_max = h
                    if char_h_min > h:
                        char_h_min = h
                    if char_w_max < w:
                        char_w_max = w
                    if char_w_min > w:
                        char_w_min = w
                    file_name = '{}-{}-{}.jpg'.format(data, page_name[: -4], k)
                    cv2.imwrite(os.path.join(out_dir, "images", file_name), char)
                    f.write(file_name + " " + labels[k]['text'] + "\n")
    print("char_h_max:{} char_h_min:{}".format(char_h_max, char_h_min))
    print("char_w_max:{} char_w_min:{}".format(char_w_max, char_w_min))
    print(total)


if __name__ == '__main__':
    root_dir = "d:/lunwen/data/difangzhi"
    make_dataset(root_dir)
