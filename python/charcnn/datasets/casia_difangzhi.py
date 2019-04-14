import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CASIA(Dataset):

    def __init__(self, label_file):
        self.image_list = []
        self.label_list = []
        with open(label_file, 'r') as f:
            file_lines = f.read().split('\n')
            for line in file_lines:
                image_path, label = line.split()
                self.image_list.append(image_path)
                self.label_list.append(label)
        print(len(self.image_list))
        if len(self.image_list) != len(self.label_list):
            raise ValueError("length of image_list and label_list don't equal")

        char_set = set(self.label_list)
        self.char2index = dict(zip(char_set, range(len(char_set))))
        # change this line for different size of input
        self.transforms = transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def get_image(img_path):
        # crop the character from the original image
        img = cv2.imread(img_path, 0)
        black_index = np.where(img < 255)
        try:
            min_x = min(black_index[0])
            max_x = max(black_index[0])
            min_y = min(black_index[1])
            max_y = max(black_index[1])
        except ValueError:
            print(img_path)
            img = cv2.resize(img, dsize=(108, 108), interpolation=cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img[min_x: max_x, min_y: max_y], dsize=(108, 108), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(img)

    def __getitem__(self, index):
        image = self.get_image(self.image_list[index])
        image = self.transforms(image)

        char_code = self.label_list[index]
        # label = torch.zeros(3926, dtype=torch.float)
        # label[self.char2index[char_code]] = 1
        label = torch.tensor(self.char2index[char_code], dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.image_list)
