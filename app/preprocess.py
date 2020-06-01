# to have a nice progress bar
from random import randrange

import cv2
import pandas
from imageio import imread
from tqdm import tqdm

data = pandas.read_csv('../data/bees/bee_data.csv')
data.head()
train_img = []
for img_name in tqdm(data['file']):
    image_path = '../data/bees/bee_imgs/bee_imgs/' + img_name
    img = imread(image_path)
    img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    h, w, c = img.shape
    print(img.shape)
    rh = randrange(0, h - 100)
    rw = randrange(0, w - 100)
    cropped = cv2.copyMakeBorder(img[rw:rw + 100, rh:rh + 100], h - (rh + 100), rh, rw, w - (rw + 100),
                                 borderType=cv2.BORDER_CONSTANT)
    # cv2.imshow('cropped', cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    train_img.append(cropped)
