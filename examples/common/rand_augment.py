from typing import Dict, List, Sequence
import albumentations as A
import cv2
import numpy as np


def rand_augment(num_ops: int = 3, magnitude: int = 9, p: float = 1.0, mode="all", cut_out = False, **kwargs):
    '''
    Taken from https://towardsdatascience.com/augmentation-methods-using-albumentations-and-pytorch-35cd135382f8
    '''
    # Magnitude(M) search space
    #  Shift is normalized space, based on 331px images
    shift_x = np.linspace(0,0.45,10)
    shift_y = np.linspace(0,0.45,10)
    rot = np.linspace(0,30,10)
    shear = np.linspace(0,10,10)
    sola = np.linspace(0,256,10)
    post = [4,4,5,5,6,6,7,7,8,8]
    cont = [np.linspace(-0.8,-0.1,10),np.linspace(0.1,2,10)]
    bright = np.linspace(0.1,0.7,10)
    shar = np.linspace(0.1,0.9,10)
    cut = np.linspace(0,60,10)
    # Transformation search space

    border = cv2.BORDER_CONSTANT

    Aug_geo = [
        lambda: A.ShiftScaleRotate(
            shift_limit_x=shift_x[magnitude],
            rotate_limit=0,
            shift_limit_y=0,
            shift_limit=shift_x[magnitude],
            p=p,
            border_mode=border
        ),
        lambda: A.ShiftScaleRotate(
            shift_limit_y=shift_y[magnitude],
            rotate_limit=0,
            shift_limit_x=0,
            shift_limit=shift_y[magnitude],
            p=p,
            border_mode=border
        ),
        lambda: A.Affine(rotate=rot[magnitude], p=p, mode=border),
        lambda: A.Affine(shear=shear[magnitude], p=p, mode=border),
    ]

    Aug_color = [#0 - geometrical
        #4 - Color Based
        lambda: A.InvertImg(p=p),
        lambda: A.Equalize(p=p),
        lambda: A.Solarize(threshold=sola[magnitude], p=p),
        lambda: A.Posterize(num_bits=post[magnitude], p=p),
        lambda: A.RandomBrightnessContrast(brightness_limit=bright[magnitude], contrast_limit=[cont[0][magnitude], cont[1][magnitude]], p=p),
        lambda: A.Sharpen(alpha=shar[magnitude], lightness=shar[magnitude], p=p)
    ]
    # Sampling from the Transformation search space
    if mode == "geo":
        ops = np.random.choice(Aug_geo, num_ops)
    elif mode == "color":
        ops = np.random.choice(Aug_color, num_ops)
    else:
        ops = np.random.choice(Aug_geo + Aug_color, num_ops)

    ops = [ctor() for ctor in ops]

    if cut_out:
        ops.append(A.Cutout(num_holes=8, max_h_size=int(cut[magnitude]),   max_w_size=int(cut[magnitude]), p=p))
    transforms = A.Compose(ops, **kwargs)
    return transforms, ops


class RandAugment(A.DualTransform):
    def __init__(self, num_ops: int = 3, magnitude: int = 9, p: float = 1.0, mode="all", cut_out = False, **kwargs):
        super().__init__(p=p)

        self.num_ops = num_ops
        self.magnitude = magnitude
        self.mode = mode
        self.cut_out = cut_out
        self.extra = kwargs

    def __call__(self, *args, **kwargs):
        tx, ops = rand_augment(self.num_ops, self.magnitude, p=1.0, mode=self.mode, cut_out=self.cut_out, **self.extra)

        return tx(*args, **kwargs)
