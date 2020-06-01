# to have a nice progress bar
from random import randrange

import cv2
import pandas
import torch
from imageio import imread
from tqdm import tqdm


def preprocess(image_data_path='../data/bees/bee_imgs/bee_imgs/', csv_path='../data/bees/bee_data.csv'):
    data = pandas.read_csv(csv_path)
    data.head()
    train_img = []
    # this could be done differently if I didn't have a dataset with classifications
    for img_name in tqdm(data['file']):
        image_path = image_data_path + img_name
        img = imread(image_path)
        img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
        h, w, c = img.shape
        try:
            rh = randrange(0, h - 100)
        except ValueError:
            rh = 0
        try:
            rw = randrange(0, w - 100)
        except ValueError:
            rw = 0

        crop = img[rw:rw + 100, rh:rh + 100]
        cropped = cv2.copyMakeBorder(crop, 62, 62, 62, 62,
                                     borderType=cv2.BORDER_CONSTANT)
        # cv2.imshow('cropped', cropped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        train_img.append(torch.from_numpy(cropped))

    return train_img
