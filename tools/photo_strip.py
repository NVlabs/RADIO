# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import argparse
import math
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Compute SSL embedding rank estimates')
    parser.add_argument('-i', '--input-dir', default='vis_denoise', type=str)
    parser.add_argument('-o', '--output-dir', type=str, required=True)

    args = parser.parse_args()

    files = [
        'vis_15.jpg',
        'vis_17.jpg',
        'vis_10.jpg',
        'vis_12.jpg',
        'vis_18.jpg',
    ]

    size = 256

    for dirpath, dirnames, filenames in os.walk(args.input_dir):
        paths = [None for _ in files]

        for filename in filenames:
            try:
                idx = files.index(filename)
                paths[idx] = os.path.join(dirpath, filename)
            except ValueError:
                continue

        if any(g is None for g in paths):
            continue

        imgs = [cv2.imread(f) for f in paths]

        for i, img in enumerate(imgs):
            height, width = img.shape[:2]

            maxdim = max(height, width)
            scale = size / maxdim
            rs_height = int(round(height * scale))
            rs_width = int(round(width * scale))

            img = cv2.resize(img, dsize=(rs_width, rs_height), interpolation=cv2.INTER_AREA)

            le_width = (size - img.shape[1]) // 2
            le_height = (size - img.shape[0]) // 2
            im2 = np.full((size, size, 3), 0, dtype=img.dtype)
            im2[le_height:le_height+img.shape[0], le_width:le_width+img.shape[1]] = img
            imgs[i] = im2

        buff = np.full((size, len(files) * size + (2 * (len(files) - 1)), 3), 0, dtype=im2.dtype)

        for i, img in enumerate(imgs):
            c = i * (size + 2)
            buff[:, c:c+size] = img

        outdir = dirpath.replace(args.input_dir, args.output_dir)
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, 'img.jpg')
        cv2.imwrite(outfile, buff)

if __name__ == '__main__':
    main()
