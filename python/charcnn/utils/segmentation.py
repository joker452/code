import numpy as np
from PIL import Image


start_threshold = 2
end_threshold = 150
min_height = 20

def segment(image_path):
    '''
    :param image_path: path to the image
    :return: dictionary the value of which are the whole image (ndarray), image_lines(ndarray),
             and the y coordinate of each image_line left upper point
    '''
    result = {'image_lines': [], 'row_start': []}
    im = np.asarray(Image.open(image_path), dtype='uint8')
    mask = np.where(im != 255, True, False)
    row_sum = mask.sum(axis=1)
    start = 0
    counter = 0
    wave = np.full(im.shape, 0, dtype='uint8')
    for i, value in enumerate(row_sum):
        wave[i, : value + 1] = 255
    Image.fromarray(wave).save('wave.png')
    for i, value in enumerate(row_sum):
        if value > start_threshold and start == 0:
            start = i
        elif start != 0 and value < end_threshold:
            if (i - start) > min_height:
                end = i
                counter = counter + 1
                result['image_lines'].append(im[start: end + 1, :])
                result['row_start'].append(start)
            start = 0
        else:
            pass
    result['image'] = im
    return result

