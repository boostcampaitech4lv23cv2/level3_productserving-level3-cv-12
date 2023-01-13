import json
from os import path as osp
import os

import numpy as np
from PIL import Image, ImageDraw

import argparse
from tqdm import tqdm


def get_img_agnostic_onlypose(img, pose_data):
    for pair in [[3,4], [6,7]]:
        pointx, pointy = pose_data[pair[1]]+(pose_data[pair[1]]-pose_data[pair[0]])*0.3
        pointx, pointy = int(pointx), int(pointy)

    r = 10
    img = np.array(img)
    img = Image.fromarray(img)
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    # mask line LHip-to-RHip
    agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], 'black', width=r*10)
    # mask line LKnee-to-RKnee
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'black', width=r*45)

    # mask circle waist
    for i in [8, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')

    # mask line leg
    for i in [9, 10, 12, 13]:
        if (pose_data[i - 1, 0] < 0.0 and pose_data[i - 1, 1] < 0.0) or (pose_data[i, 0] < 0.0 and pose_data[i, 1] < 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'black', width=r*10)
        pointx, pointy = pose_data[i]
        if i in [10, 13]:
            pass    #agnostic_draw.ellipse((pointx-r, pointy-r, pointx+r, pointy+r), 'black', 'black')
        else:
            agnostic_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')

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

        img = Image.open(osp.join(data_path, 'images', im_name)).convert("RGB")

        json_name = im_name.replace('_0.jpg', '_2.json')
        with open(osp.join(data_path, 'keypoints', json_name)) as f:
            json_file = json.load(f)
            pose_data = json_file['keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data[:, :2] * 2

        agnostic = get_img_agnostic_onlypose(img, pose_data)
        
        agnostic.save(osp.join(output_path, im_name))
