import cv2
import numpy as np
import torch


def preprocess(img):
    img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    max_x = img.shape[1] - 100
    max_y = img.shape[0] - 100

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = img[y: y + 100, x: x + 100]
    cropped = cv2.copyMakeBorder(crop, 62, 62, 62, 62,
                                 borderType=cv2.BORDER_CONSTANT)

    # cv2.imshow('img', cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ar = torch.from_numpy(cropped)
    # cv2 returns HxWxC format but I need CxHxW
    return ar.permute(2, 0, 1)
