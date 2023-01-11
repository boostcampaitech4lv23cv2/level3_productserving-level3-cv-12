import json
from os import path as osp
import os

import numpy as np
from PIL import Image, ImageDraw

import argparse

from tqdm import tqdm


def get_im_parse_agnostic(im_parse):
    parse_array = np.array(im_parse)
    parse_upper = ((parse_array == 4).astype(np.uint8))
    parse_bottom = ((parse_array == 6).astype(np.uint8) +
                    (parse_array == 8).astype(np.uint8))
    # 원래는 목 부분을 지우는데 dress code는 목에 대한 segmentation class가 없어 face를 선택하여 face를 지움
    parse_neck = (parse_array == 11).astype(np.uint8)

    parse_arms = ((parse_array == 14).astype(np.uint8) +
                  (parse_array == 15).astype(np.uint8))
    parse_legs = ((parse_array == 12).astype(np.uint8) +
                  (parse_array == 13).astype(np.uint8))

    agnostic = im_parse.copy()

    # mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_bottom * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arms * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_legs * 255), 'L'))

    return agnostic


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help="dataset dir")
    parser.add_argument('--output_path', type=str, help="output dir")

    args = parser.parse_args()
    data_path = args.data_path
    output_path = args.output_path
    
    os.makedirs(output_path, exist_ok=True)
    
    for im_name in tqdm(os.listdir(osp.join(data_path, 'images'))):
        if '_1.jpg' in im_name:
            continue

        # load parsing image
        parse_name = im_name.replace('_0.jpg', '_4.png')
        im_parse = Image.open(osp.join(data_path, 'label_maps', parse_name))

        agnostic = get_im_parse_agnostic(im_parse)
        
        agnostic.save(osp.join(output_path, parse_name))
