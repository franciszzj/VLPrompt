import os
import sys
import cv2
import json
import copy
import mmcv
from tqdm import tqdm
import random
import numpy as np
import seaborn as sns
from skimage.segmentation import find_boundaries
from panopticapi.utils import rgb2id, id2rgb

file_client = mmcv.FileClient(**dict(backend='disk'))


frequency_dict = {
    'over': 39950,
    'in front of': 11433,
    'beside': 45859,
    'on': 54222,
    'in': 10051,
    'attached to': 20967,
    'hanging from': 3935,
    'on back of': 146,
    'falling off': 10,
    'going down': 118,
    'painted on': 192,
    'walking on': 7256,
    'running on': 1174,
    'crossing': 205,
    'standing on': 18754,
    'lying on': 1518,
    'sitting on': 5444,
    'flying over': 869,
    'jumping over': 81,
    'jumping from': 179,
    'wearing': 2982,
    'holding': 10466,
    'carrying': 2385,
    'looking at': 5351,
    'guiding': 90,
    'kissing': 17,
    'eating': 1283,
    'drinking': 117,
    'feeding': 70,
    'biting': 162,
    'catching': 135,
    'picking': 15,
    'playing with': 353,
    'chasing': 39,
    'climbing': 12,
    'cleaning': 22,
    'playing': 1934,
    'touching': 979,
    'pushing': 78,
    'pulling': 323,
    'opening': 8,
    'cooking': 19,
    'talking to': 428,
    'throwing': 183,
    'slicing': 139,
    'driving': 616,
    'riding': 2061,
    'parked on': 6795,
    'driving on': 5785,
    'about to hit': 572,
    'kicking': 75,
    'swinging': 743,
    'entering': 55,
    'exiting': 28,
    'enclosing': 600,
    'leaning on': 831,
}


def vis(palette, object_classes, predicate_classes, meta_info, coco_path, output_path):

    # filter target relations
    '''
    target_relation_list = ['pulling', 'pushing', 'guiding', 'entering', 'existing']
    target = False
    for idx, triplet in enumerate(meta_info['relations']):
        rel_label = triplet[2]
        if predicate_classes[rel_label] in target_relation_list:
            target = True
    if not target:
        return
    # '''
    # if '000000079966' not in meta_info['file_name']:
    #     return

    file_name = meta_info['file_name'].split('/')[-1]
    image_path = os.path.join(coco_path, meta_info['file_name'])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    seg_path = os.path.join(coco_path, meta_info['pan_seg_file_name'])
    seg_bytes = file_client.get(seg_path)
    seg = mmcv.imfrombytes(seg_bytes,
                           flag='color',
                           channel_order='rgb').squeeze()
    seg_id = rgb2id(seg)
    boundaries = find_boundaries(seg_id, mode='thick')

    new_seg = copy.deepcopy(image)
    boundaries_for_object_list = []
    for idx, object in enumerate(meta_info['segments_info']):
        object_id = object['id']
        object_label = object['category_id']
        index = np.where(seg_id == object_id)
        # whole image
        new_seg[index] = [int(x * 255)
                          for x in palette[object_label * 5 + random.randint(0, 4)]]
        # for each object
        mask_for_object = np.zeros_like(seg_id)
        mask_for_object[index] = 1
        boundaries_for_object = find_boundaries(mask_for_object, mode='inner')
        boundaries_for_object_list.append(boundaries_for_object)

    # plot whole image
    '''
    new_image = image * 0.5 + new_seg * 0.5
    new_image = new_image.astype(np.uint8)
    new_image[boundaries] = [0, 0, 0]
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('{}/{}_pan_seg.jpg'.format(output_path, file_name.split('.')[0]), new_image)
    # '''

    # plot relations
    h, w, _ = image.shape
    merged_image = np.zeros((h * 4, w * 5, 3), dtype=np.uint8)
    for idx, triplet in enumerate(meta_info['relations']):
        # if idx >= 20:
        #     break
        sub_idx = triplet[0]
        obj_idx = triplet[1]
        rel_label = triplet[2]
        if frequency_dict[predicate_classes[rel_label]] > 500:
            continue
        # if predicate_classes[rel_label] not in target_relation_list:
        #     continue
        # if object_classes[meta_info['segments_info'][obj_idx]['category_id']] != 'elephant':
        #     continue
        sub_boundaries = boundaries_for_object_list[sub_idx]
        obj_boundaries = boundaries_for_object_list[obj_idx]
        image_for_relation = copy.deepcopy(image)
        image_for_relation = image_for_relation * 0.5 + new_seg * 0.5
        image_for_relation = image_for_relation.astype(np.uint8)
        image_for_relation[obj_boundaries] = [0, 0, 255]
        image_for_relation[sub_boundaries] = [255, 0, 0]
        sub_size = cv2.getTextSize(
            object_classes[meta_info['segments_info'][sub_idx]['category_id']], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        obj_size = cv2.getTextSize(
            object_classes[meta_info['segments_info'][obj_idx]['category_id']], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        rel_size = cv2.getTextSize(
            predicate_classes[rel_label], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        image_for_relation = cv2.rectangle(image_for_relation, (10, 10), (
            10 + sub_size[0][0], 10 + sub_size[0][1]), (255, 255, 255), -1)
        image_for_relation = cv2.rectangle(image_for_relation, (10, 70), (
            10 + obj_size[0][0], 70 + obj_size[0][1]), (255, 255, 255), -1)
        image_for_relation = cv2.rectangle(image_for_relation, (10, 40), (
            10 + rel_size[0][0], 40 + rel_size[0][1]), (255, 255, 255), -1)
        image_for_relation = cv2.putText(image_for_relation, object_classes[meta_info['segments_info'][sub_idx]['category_id']], (
            10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        image_for_relation = cv2.putText(image_for_relation, object_classes[meta_info['segments_info'][obj_idx]['category_id']], (
            10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        image_for_relation = cv2.putText(image_for_relation, predicate_classes[rel_label], (
            10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        image_for_relation = image_for_relation.astype(np.uint8)
        image_for_relation = cv2.cvtColor(
            image_for_relation, cv2.COLOR_RGB2BGR)
        # merged_image[(idx // 5) * h: (idx // 5 + 1) * h, (idx % 5) * w: (idx % 5 + 1) * w, :] = image_for_relation
        # save each relation
        # '''
        cv2.imwrite('{}/{}_rel_{}.jpg'.format(output_path,
                    file_name.split('.')[0], idx), image_for_relation)
        # '''
    # save all relations (top 20)
    # cv2.imwrite('{}/{}'.format(output_path, file_name), merged_image)
    # cv2.imwrite('{}/{}_rel_all.jpg'.format(output_path, file_name.split('.')[0]), merged_image)


def main(psg_file, coco_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load PSG
    psg = json.load(open(psg_file))
    data = psg['data']
    thing_classes = psg['thing_classes']
    stuff_classes = psg['stuff_classes']
    predicate_classes = psg['predicate_classes']
    object_classes = thing_classes + stuff_classes
    test_image_ids = psg['test_image_ids']

    # Palette
    palette = sns.color_palette('hls', len(object_classes) * 5)

    for i, d in tqdm(enumerate(data), total=len(data), desc='Visualizing', ncols=100):
        if d['image_id'] in test_image_ids:
            vis(palette, object_classes, predicate_classes, d, coco_path, output_path)


if __name__ == '__main__':
    psg_file = '/users/k21163430/workspace/KingsSGG/data/psg/psg.json'
    coco_path = '/users/k21163430/workspace/KingsSGG/data/coco/'
    output_path = './vis/'
    main(psg_file, coco_path, output_path)
