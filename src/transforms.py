# src/transforms.py

import random
import torchvision.transforms.functional as F

class Compose(object):
    """Simple wrapper to apply multiple transforms in sequence."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomHorizontalFlip(object):
    """Flip the image and bounding boxes horizontally with a given probability."""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # image shape is (C, H, W)
            # Flip the image horizontally
            image = F.hflip(image)
            # Flip the bounding boxes
            _, _, width = image.shape  # after flip, but width is the same
            boxes = target["boxes"]
            # boxes are [x_min, y_min, x_max, y_max]
            # new x_min = width - old x_max, etc.
            boxes[:, [0,2]] = width - boxes[:, [2,0]]
            target["boxes"] = boxes
        return image, target

def get_transforms(train=False):
    # You can add more transforms here if you want
    transforms = []
    if train:
        transforms.append(RandomHorizontalFlip(prob=0.5))
    return Compose(transforms)
