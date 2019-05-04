import os
import subprocess
root_dir = "/data2/dengbowen/work/samples/dunhuang/"
# .../BD00963
data_dir = [root_dir + dir for dir in os.listdir(root_dir) if dir != 'data']
counter = 0
for dir in data_dir:
    # .../BD00963/out
    img_dir = os.path.join(dir, 'out')
    img_names = [f for f in os.listdir(img_dir) if f != 'labels.txt' and f != 'images']
    for img in img_names:
        # .../BD000963/out/xxx.jpg
        img_path = os.path.join(img_dir, img)
        subprocess.call('cp {} {}'.format(img_path, os.path.join('/data2/dengbowen/work/samples/dunhuang/data', img)),
                        shell=True)
        counter += 1
    label_path = os.path.join(img_dir, 'labels.txt')
    subprocess.call('cp {} {}'.format(label_path, '/data2/dengbowen/work/samples/dunhuang/data/{}labels.txt')\
                        .format(img_dir.split('/')[-2]), shell=True)
print(counter)