cmake_minimum_required(VERSION 3.12)
project(intern)

set(CMAKE_CXX_STANDARD 14)

add_executable(intern
        charcnn/datasets/__init__.py
        charcnn/datasets/casia_alexnet.py
        charcnn/datasets/casia_charcnn.py
        charcnn/datasets/casia_googlenet.py
        charcnn/datasets/casia_resnetx.py
        charcnn/datasets/casia_semi.py
        charcnn/datasets/Char3319-list.txt
        charcnn/datasets/Char4037-list.txt
        charcnn/models/alexnet.py
        charcnn/models/charcnn.py
        charcnn/models/dgm.py
        charcnn/models/googlenet.py
        charcnn/models/resnetx.py
        charcnn/models/vae.py
        "charcnn/runs/Jul23_09-12-50_user-ubuntu07-23 09_12_50CharCnn/events.out.tfevents.1532351570.user-ubuntu"
        "charcnn/runs/Jul23_09-54-09_user-ubuntu07-23 09_54_09GoogLeNet/events.out.tfevents.1532354049.user-ubuntu"
        "charcnn/runs/Jul23_09-57-54_user-ubuntu07-23 09_57_54ResNetX/events.out.tfevents.1532354274.user-ubuntu"
        "charcnn/runs/Jul23_10-01-35_user-ubuntu07-23 10_01_35AlexNet/events.out.tfevents.1532354495.user-ubuntu"
        charcnn/utils/fonts/DroidSansFallbackFull.ttf
        charcnn/utils/fonts/fangzheng_heiti.TTF
        charcnn/utils/fonts/mingliu.ttc
        charcnn/utils/fonts/NotoSansHans-Black.otf
        charcnn/utils/fonts/NotoSansHans-Bold.otf
        charcnn/utils/fonts/NotoSansHans-DemiLight.otf
        charcnn/utils/fonts/NotoSansHans-Light.otf
        charcnn/utils/fonts/NotoSansHans-Medium.otf
        charcnn/utils/fonts/NotoSansHans-Regular.otf
        charcnn/utils/fonts/NotoSansHans-Thin-Windows.otf
        charcnn/utils/char2index.pkl
        charcnn/utils/dgr.py
        charcnn/utils/freq.pkl
        charcnn/utils/gen_printed_char.py
        charcnn/utils/gnt.py
        charcnn/utils/pallete
        charcnn/utils/segmentation.py
        charcnn/test.py
        charcnn/train_alexnet.py
        charcnn/train_charcnn.py
        charcnn/train_googlenet.py
        charcnn/train_resnetx.py)
