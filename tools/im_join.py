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
    parser.add_argument('-i', '--input-dirs', nargs='+', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-c', '--cols', type=int, default=0, help='The number of columns in the mosaic')
    parser.add_argument('--header', type=int, default=128, help='The height for the cell headers')
    parser.add_argument('--header-color', type=int, default=255, help='The color of the header')

    args = parser.parse_args()

    all_files = []
    for indir in args.input_dirs:
        dir_files = [os.path.join(indir, f) for f in os.listdir(indir)]
        dir_files.sort()
        all_files.append(dir_files)

    # Transpose, so we're iterating over grouped images
    all_files = list(zip(*all_files))

    os.makedirs(args.output_dir, exist_ok=True)

    num_cols = args.cols
    if not num_cols:
        num_cols = int(math.ceil(math.sqrt(len(args.input_dirs))))
    num_rows = int(math.ceil(len(args.input_dirs) / num_cols))

    for group in all_files:
        fname = os.path.basename(group[0])

        imgs = [cv2.imread(f) for f in group]

        max_width = 0
        max_height = 0
        for im in imgs:
            max_width = max(max_width, im.shape[1])
            max_height = max(max_height, im.shape[0])

        for i in range(len(imgs)):
            im = imgs[i]
            if im.shape[0] != max_height or im.shape[1] != max_width:
                im_ratio = im.shape[0] / im.shape[1]
                max_ratio = max_height / max_width

                if im_ratio > max_ratio:
                    # Height varies faster than width
                    scale = max_height / im.shape[0]
                else:
                    scale = max_width / im.shape[1]
                rs_height = int(round(im.shape[0] * scale))
                rs_width = int(round(im.shape[1] * scale))
                im = cv2.resize(im, dsize=(rs_width, rs_height), interpolation=cv2.INTER_CUBIC)

                if im.shape[0] != max_height or im.shape[1] != max_width:
                    le_width = (max_width - im.shape[1]) // 2
                    le_height = (max_height - im.shape[0]) // 2
                    im2 = np.full((max_height, max_width, 3), 0, dtype=im.dtype)
                    im2[le_height:le_height+im.shape[0], le_width:le_width+im.shape[1]] = im
                    im = im2
            imgs[i] = im

        buff = np.full((num_rows * (max_height + args.header) + (2 * (num_rows - 1)), num_cols * max_width + (2 * (num_cols - 1)), 3), 255, dtype=imgs[0].dtype)

        for i, img in enumerate(imgs):
            c = i % num_cols
            r = i // num_cols

            c = c * (max_width + 2)
            r = r * (max_height + 2 + args.header)

            buff[r:r+args.header, c:c+max_width] = args.header_color

            r += args.header

            buff[r:r+max_height, c:c+max_width] = img

        cv2.imwrite(os.path.join(args.output_dir, fname), buff)


if __name__ == '__main__':
    main()
