from PIL import Image
import torchvision.transforms as T


class ZoomIn(object):

    def __init__(self, r):
        self.r = r

    def __call__(self, img):
        newsize = (int(img.width * self.r), int(img.height * self.r))
        return img.resize(newsize)


def preprocess(path):
    augs = T.Compose([
        ZoomIn(1.5),
        T.RandomCrop(100),
        T.Pad(62),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(path).convert("RGB")
    img = augs(img)
    return img
